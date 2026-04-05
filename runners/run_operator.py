from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.methods import get_directional_weight_builder, weighted_mean
from src.models.operator import build_operator_model
from src.tasks.operator import OperatorDataConfig, build_operator_dataloaders
from src.utils import (
    append_jsonl,
    build_pair_metadata,
    canonicalize_method_name,
    count_parameters,
    make_run_dir,
    save_curves,
    save_json,
    save_live_loss_plot,
    set_seed,
    uses_peer_model,
)

OPERATOR_MODEL_CHOICES = [
    "fno",
    "deeponet",
    "gnot",
    "neuralop_fno",
    "neuralop_tfno",
    "neuralop_uno",
    "uno",
]
OPERATOR_METHOD_CHOICES = ["independent", "dml", "ssml"]


def parse_args():
    p = argparse.ArgumentParser(description="Run operator-learning experiment")
    p.add_argument("--dataset", type=str, default="burgers", choices=["burgers", "darcy", "navier_stokes"])
    p.add_argument(
        "--model",
        type=str,
        default="fno",
        choices=OPERATOR_MODEL_CHOICES,
    )
    p.add_argument(
        "--peer-model",
        type=str,
        default=None,
        choices=OPERATOR_MODEL_CHOICES,
    )
    p.add_argument("--method", type=str, default="dml", choices=OPERATOR_METHOD_CHOICES)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="results/experiments")
    p.add_argument("--download", action="store_true")
    p.add_argument("--regression-imitation-loss", type=str, default="mse", choices=["mse", "mae", "huber"])
    p.add_argument("--lambda-imitation", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--imitation-decay-start-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-end-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-min-scale", type=float, default=1.0)
    p.add_argument("--hetero-ssml-one-way", action="store_true")
    p.add_argument("--ssml-student-only", action="store_true")
    p.add_argument("--ssml-freeze-peer", action="store_true")
    p.add_argument("--operator-weight-granularity", type=str, default="sample", choices=["sample", "element"])
    p.add_argument("--relay-stage-epochs", type=str, default="")
    p.add_argument("--relay-hint-mode", type=str, default="full", choices=["full", "coarse", "hotspot"])
    p.add_argument("--relay-taper-schedule", type=str, default="linear", choices=["linear", "cosine", "constant"])
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--peer-init-checkpoint", type=str, default=None)
    p.add_argument("--live-plot-interval", type=int, default=20)
    return p.parse_args()


def unpack_batch(batch):
    if isinstance(batch, dict):
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        if "input" in batch and "output" in batch:
            return batch["input"], batch["output"]
        raise KeyError("Dict batch must contain either ('x','y') or ('input','output').")
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Unsupported batch format.")


def build_regression_imitation_loss_fn(
    imitation_loss_name: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if imitation_loss_name == "mse":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return reduce_loss_per_sample(F.mse_loss(pred, peer_pred.detach(), reduction="none"))

        return _loss

    if imitation_loss_name == "mae":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return reduce_loss_per_sample(F.l1_loss(pred, peer_pred.detach(), reduction="none"))

        return _loss

    if imitation_loss_name == "huber":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return reduce_loss_per_sample(F.smooth_l1_loss(pred, peer_pred.detach(), reduction="none"))

        return _loss

    raise ValueError(f"Unsupported regression imitation loss: {imitation_loss_name}")


def reduce_loss_per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
    if loss_tensor.ndim <= 1:
        return loss_tensor.reshape(-1)
    return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)


def build_regression_elementwise_loss_fn(
    imitation_loss_name: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if imitation_loss_name == "mse":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(pred, peer_pred.detach(), reduction="none")

        return _loss

    if imitation_loss_name == "mae":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(pred, peer_pred.detach(), reduction="none")

        return _loss

    if imitation_loss_name == "huber":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return F.smooth_l1_loss(pred, peer_pred.detach(), reduction="none")

        return _loss

    raise ValueError(f"Unsupported regression imitation loss: {imitation_loss_name}")


def compute_effective_lambda(
    base_lambda: float,
    *,
    epoch: int,
    method: str,
    warmup_epochs: int,
    decay_start_epoch: int,
    decay_end_epoch: int,
    decay_min_scale: float,
) -> float:
    if method == "independent" or base_lambda <= 0.0:
        return 0.0
    if method == "ssml" and epoch <= warmup_epochs:
        return 0.0
    if decay_start_epoch < 0 or decay_end_epoch <= decay_start_epoch:
        return base_lambda
    if epoch <= decay_start_epoch:
        return base_lambda
    if epoch >= decay_end_epoch:
        return base_lambda * decay_min_scale

    progress = (epoch - decay_start_epoch) / max(decay_end_epoch - decay_start_epoch, 1)
    scale = 1.0 + (decay_min_scale - 1.0) * progress
    return base_lambda * scale


def parse_relay_stage_epochs(spec: str) -> tuple[int, int, int]:
    if not spec.strip():
        return (0, 0, 0)
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if len(values) != 3:
        raise ValueError(
            f"relay-stage-epochs must contain exactly three comma-separated integers, got: {spec!r}"
        )
    return tuple(max(value, 0) for value in values)


def compute_relay_curriculum(
    epoch: int,
    stage_epochs: tuple[int, int, int],
    taper_schedule: str,
) -> tuple[str, float]:
    stage0_epochs, stage1_epochs, stage2_epochs = stage_epochs
    if stage0_epochs + stage1_epochs + stage2_epochs <= 0:
        return "disabled", 1.0
    if epoch <= stage0_epochs:
        return "control", 0.0
    if epoch <= stage0_epochs + stage1_epochs:
        return "relay", 1.0
    if epoch <= stage0_epochs + stage1_epochs + stage2_epochs:
        if stage2_epochs <= 0:
            return "finetune", 0.0
        if stage2_epochs == 1:
            progress = 1.0
        else:
            progress = (epoch - stage0_epochs - stage1_epochs - 1) / max(stage2_epochs - 1, 1)
        progress = float(max(0.0, min(1.0, progress)))
        if taper_schedule == "linear":
            scale = 1.0 - progress
        elif taper_schedule == "cosine":
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            scale = 1.0
        return "taper", float(max(scale, 0.0))
    return "finetune", 0.0


def build_low_frequency_operator_hint(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().reshape(tensor.shape[0], 1, -1)
    if flat.shape[-1] <= 2:
        return tensor.detach()
    kernel = max(3, min(int(flat.shape[-1] // 32) * 2 + 1, min(int(flat.shape[-1]), 31)))
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, int(flat.shape[-1]) if int(flat.shape[-1]) % 2 == 1 else max(int(flat.shape[-1]) - 1, 1))
    if kernel <= 1:
        return tensor.detach()
    pad = kernel // 2
    smoothed = F.avg_pool1d(F.pad(flat, (pad, pad), mode="replicate"), kernel_size=kernel, stride=1)
    return smoothed.reshape_as(tensor).detach()


def build_operator_teacher_hint(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    hint_mode: str,
    margin: float,
) -> tuple[torch.Tensor, float]:
    if hint_mode == "full":
        return teacher_pred.detach(), 1.0
    if hint_mode == "coarse":
        return build_low_frequency_operator_hint(teacher_pred), 0.0
    if hint_mode == "hotspot":
        student_error = F.mse_loss(student_pred.detach(), target.detach(), reduction="none")
        teacher_error = F.mse_loss(teacher_pred.detach(), target.detach(), reduction="none")
        hotspot_mask = (student_error - teacher_error) > margin
        hotspot_ratio = float(hotspot_mask.to(dtype=torch.float32).mean().item()) if hotspot_mask.numel() > 0 else 0.0
        hint = torch.where(hotspot_mask, teacher_pred.detach(), student_pred.detach())
        return hint, hotspot_ratio
    raise ValueError(f"Unsupported relay hint mode: {hint_mode}")


def choose_one_way_imitation(
    student_supervised_loss: torch.Tensor,
    peer_supervised_loss: torch.Tensor,
) -> tuple[bool, bool]:
    student_mean = student_supervised_loss.mean()
    peer_mean = peer_supervised_loss.mean()
    if torch.isclose(student_mean, peer_mean, rtol=1e-4, atol=1e-6):
        return False, False
    if float(student_mean.item()) > float(peer_mean.item()):
        return True, False
    return False, True


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str], label: str) -> Optional[str]:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    print(f"[operator] loaded {label}_checkpoint={path}")
    return str(path)


def train_one_epoch(
    model,
    peer_model: Optional[torch.nn.Module],
    loader,
    optimizer,
    peer_optimizer: Optional[torch.optim.Optimizer],
    device,
    supervised_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    supervised_elementwise_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    elementwise_imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_imitation: float,
    margin: float,
    method: str,
    relay_scale: float = 1.0,
    relay_stage: str = "disabled",
    relay_hint_mode: str = "full",
    hetero_ssml_one_way: bool = False,
    peer_update_disabled: bool = False,
    operator_weight_granularity: str = "sample",
):
    method = canonicalize_method_name(method)
    model.train()
    if peer_model is not None:
        if peer_update_disabled:
            peer_model.eval()
        else:
            peer_model.train()
    dml_weight_builder = get_directional_weight_builder("dml")
    ssml_weight_builder = get_directional_weight_builder("ssml")
    total_loss = 0.0
    total_relay_hotspot_ratio = 0.0
    total_count = 0
    for batch in loader:
        x, y = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if peer_optimizer is not None:
            peer_optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        supervised_loss_map = supervised_elementwise_loss_fn(pred, y)
        supervised_loss = reduce_loss_per_sample(supervised_loss_map)
        batch_relay_hotspot_ratio = 0.0
        relay_lambda = lambda_imitation * max(float(relay_scale), 0.0)

        if method == "independent":
            loss = supervised_loss.mean()
            loss.backward()
            optimizer.step()

        elif method == "dml":
            if peer_model is None:
                raise ValueError("peer_model is required when method='dml'")
            if not peer_update_disabled and peer_optimizer is None:
                raise ValueError("peer_optimizer is required when method='dml' unless peer updates are disabled")

            if peer_update_disabled:
                with torch.no_grad():
                    peer_pred = peer_model(x)
            else:
                peer_pred = peer_model(x)
            peer_supervised_loss_map = supervised_elementwise_loss_fn(peer_pred, y)
            peer_supervised_loss = reduce_loss_per_sample(peer_supervised_loss_map)
            student_weight_source = supervised_loss.detach()
            peer_weight_source = peer_supervised_loss.detach()
            if operator_weight_granularity == "element":
                student_weight_source = supervised_loss_map.detach()
                peer_weight_source = peer_supervised_loss_map.detach()
            w_student, w_peer = dml_weight_builder(
                student_weight_source,
                peer_weight_source,
                margin=margin,
            )
            if relay_lambda <= 0.0:
                w_student = torch.zeros_like(w_student)
                w_peer = torch.zeros_like(w_peer)
            student_hint_target, batch_relay_hotspot_ratio = build_operator_teacher_hint(
                pred,
                peer_pred,
                y,
                hint_mode=relay_hint_mode,
                margin=margin,
            )
            imitation_student_values = (
                elementwise_imitation_loss_fn(pred, student_hint_target)
                if operator_weight_granularity == "element"
                else imitation_loss_fn(pred, student_hint_target)
            )
            imitation_student = weighted_mean(imitation_student_values, w_student)

            loss = supervised_loss.mean() + relay_lambda * imitation_student
            if peer_update_disabled:
                loss.backward()
                optimizer.step()
            else:
                peer_hint_target, _ = build_operator_teacher_hint(
                    peer_pred,
                    pred,
                    y,
                    hint_mode=relay_hint_mode,
                    margin=margin,
                )
                imitation_peer_values = (
                    elementwise_imitation_loss_fn(peer_pred, peer_hint_target)
                    if operator_weight_granularity == "element"
                    else imitation_loss_fn(peer_pred, peer_hint_target)
                )
                imitation_peer = weighted_mean(imitation_peer_values, w_peer)
                peer_loss = peer_supervised_loss.mean() + relay_lambda * imitation_peer

                (loss + peer_loss).backward()
                optimizer.step()
                peer_optimizer.step()

        elif method == "ssml":
            if peer_model is None:
                raise ValueError("peer_model is required when method='ssml'")
            if not peer_update_disabled and peer_optimizer is None:
                raise ValueError("peer_optimizer is required when method='ssml' unless peer updates are disabled")
            if peer_update_disabled:
                with torch.no_grad():
                    peer_pred = peer_model(x)
            else:
                peer_pred = peer_model(x)
            peer_supervised_loss_map = supervised_elementwise_loss_fn(peer_pred, y)
            peer_supervised_loss = reduce_loss_per_sample(peer_supervised_loss_map)
            student_weight_source = supervised_loss.detach()
            peer_weight_source = peer_supervised_loss.detach()
            if operator_weight_granularity == "element":
                student_weight_source = supervised_loss_map.detach()
                peer_weight_source = peer_supervised_loss_map.detach()
            w_student, w_peer = ssml_weight_builder(
                student_weight_source,
                peer_weight_source,
                margin=margin,
            )
            if relay_lambda <= 0.0:
                w_student = torch.zeros_like(w_student)
                w_peer = torch.zeros_like(w_peer)
            elif hetero_ssml_one_way:
                student_imitates, peer_imitates = choose_one_way_imitation(
                    supervised_loss.detach(),
                    peer_supervised_loss.detach(),
                )
                if not student_imitates:
                    w_student = torch.zeros_like(w_student)
                if not peer_imitates:
                    w_peer = torch.zeros_like(w_peer)

            student_hint_target, batch_relay_hotspot_ratio = build_operator_teacher_hint(
                pred,
                peer_pred,
                y,
                hint_mode=relay_hint_mode,
                margin=margin,
            )
            imitation_student_values = (
                elementwise_imitation_loss_fn(pred, student_hint_target)
                if operator_weight_granularity == "element"
                else imitation_loss_fn(pred, student_hint_target)
            )
            imitation_term_student = weighted_mean(imitation_student_values, w_student)
            loss = supervised_loss.mean() + relay_lambda * imitation_term_student

            if peer_update_disabled:
                loss.backward()
                optimizer.step()
            else:
                peer_hint_target, _ = build_operator_teacher_hint(
                    peer_pred,
                    pred,
                    y,
                    hint_mode=relay_hint_mode,
                    margin=margin,
                )
                imitation_peer_values = (
                    elementwise_imitation_loss_fn(peer_pred, peer_hint_target)
                    if operator_weight_granularity == "element"
                    else imitation_loss_fn(peer_pred, peer_hint_target)
                )
                imitation_term_peer = weighted_mean(imitation_peer_values, w_peer)
                peer_loss = peer_supervised_loss.mean() + relay_lambda * imitation_term_peer

                (loss + peer_loss).backward()
                optimizer.step()
                peer_optimizer.step()
        else:
            raise ValueError(f"Unsupported method '{method}'")

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_relay_hotspot_ratio += batch_relay_hotspot_ratio * batch_size
        total_count += batch_size
    return {
        "train_mse": total_loss / total_count,
        "relay_hotspot_ratio": total_relay_hotspot_ratio / total_count,
        "relay_stage": relay_stage,
        "relay_scale": relay_scale,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for batch in loader:
        x, y = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        mse = F.mse_loss(pred, y)
        mae = F.l1_loss(pred, y)
        batch_size = x.size(0)
        total_mse += float(mse.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_count += batch_size
    return total_mse / total_count, total_mae / total_count


def main():
    args = parse_args()
    args.method = canonicalize_method_name(args.method)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = build_operator_dataloaders(
        OperatorDataConfig(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            download=args.download,
        )
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    meta = data["meta"]

    peer_model_name = (args.peer_model or args.model) if uses_peer_model(args.method) else None
    pair_meta = build_pair_metadata(args.model, peer_model_name)
    model = build_operator_model(args.model, args.dataset, meta).to(device)
    peer_model = None
    peer_optimizer = None
    if uses_peer_model(args.method):
        peer_model = build_operator_model(pair_meta["peer_model"], args.dataset, meta).to(device)
    loaded_init_checkpoint = load_model_checkpoint(model, args.init_checkpoint, "init")
    loaded_peer_init_checkpoint = None
    if peer_model is not None:
        loaded_peer_init_checkpoint = load_model_checkpoint(peer_model, args.peer_init_checkpoint, "peer_init")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if peer_model is not None:
        peer_optimizer = torch.optim.AdamW(peer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    peer_update_disabled = args.method == "ssml" and (args.ssml_student_only or args.ssml_freeze_peer)
    if peer_model is not None and peer_update_disabled:
        for param in peer_model.parameters():
            param.requires_grad_(False)
        peer_optimizer = None
    supervised_loss_fn = build_regression_imitation_loss_fn("mse")
    supervised_elementwise_loss_fn = build_regression_elementwise_loss_fn("mse")
    imitation_loss_fn = build_regression_imitation_loss_fn(args.regression_imitation_loss)
    elementwise_imitation_loss_fn = build_regression_elementwise_loss_fn(args.regression_imitation_loss)
    relay_stage_epochs = parse_relay_stage_epochs(args.relay_stage_epochs)

    run_dir = make_run_dir(
        args.output_dir,
        "operator",
        args.dataset,
        f"{pair_meta['pair_tag']}_{args.method}_{args.regression_imitation_loss}_seed{args.seed}",
    )
    print(f"[operator] run_dir={run_dir}")
    print(f"[operator] params={count_parameters(model)}")
    print(
        "[operator] "
        f"granularity={args.operator_weight_granularity} "
        f"student_only={args.ssml_student_only} "
        f"freeze_peer={args.ssml_freeze_peer} "
        f"peer_updates_disabled={peer_update_disabled} "
        f"relay_stage_epochs={relay_stage_epochs} "
        f"relay_hint_mode={args.relay_hint_mode} "
        f"relay_taper_schedule={args.relay_taper_schedule}"
    )

    epoch_log_path = Path(run_dir) / "epoch_metrics.jsonl"
    if epoch_log_path.exists():
        epoch_log_path.unlink()

    train_mse_curve = []
    val_mse_curve = []
    val_mae_curve = []
    peer_val_mse_curve = []
    peer_val_mae_curve = []
    best_val_mse = float("inf")
    best_peer_val_mse = float("inf")
    best_stage_val_mse = {
        "control": float("inf"),
        "relay": float("inf"),
        "taper": float("inf"),
        "finetune": float("inf"),
        "disabled": float("inf"),
    }
    hetero_ssml_one_way = args.hetero_ssml_one_way and pair_meta["is_heterogeneous_pair"]

    for epoch in range(1, args.epochs + 1):
        effective_lambda = compute_effective_lambda(
            args.lambda_imitation,
            epoch=epoch,
            method=args.method,
            warmup_epochs=args.warmup_epochs,
            decay_start_epoch=args.imitation_decay_start_epoch,
            decay_end_epoch=args.imitation_decay_end_epoch,
            decay_min_scale=args.imitation_decay_min_scale,
        )
        relay_stage, relay_scale = compute_relay_curriculum(
            epoch,
            relay_stage_epochs,
            args.relay_taper_schedule,
        )

        train_stats = train_one_epoch(
            model,
            peer_model,
            train_loader,
            optimizer,
            peer_optimizer,
            device,
            supervised_loss_fn=supervised_loss_fn,
            supervised_elementwise_loss_fn=supervised_elementwise_loss_fn,
            imitation_loss_fn=imitation_loss_fn,
            elementwise_imitation_loss_fn=elementwise_imitation_loss_fn,
            lambda_imitation=effective_lambda,
            margin=args.margin,
            method=args.method,
            relay_scale=relay_scale,
            relay_stage=relay_stage,
            relay_hint_mode=args.relay_hint_mode,
            hetero_ssml_one_way=hetero_ssml_one_way,
            peer_update_disabled=peer_update_disabled,
            operator_weight_granularity=args.operator_weight_granularity,
        )
        tr_mse = float(train_stats["train_mse"])
        va_mse, va_mae = evaluate(model, val_loader, device)
        peer_va_mse = None
        peer_va_mae = None
        if peer_model is not None:
            peer_va_mse, peer_va_mae = evaluate(peer_model, val_loader, device)

        train_mse_curve.append(tr_mse)
        val_mse_curve.append(va_mse)
        val_mae_curve.append(va_mae)
        best_val_mse = min(best_val_mse, va_mse)
        best_stage_val_mse[relay_stage] = min(best_stage_val_mse.get(relay_stage, float("inf")), va_mse)
        if peer_va_mse is not None and peer_va_mae is not None:
            peer_val_mse_curve.append(peer_va_mse)
            peer_val_mae_curve.append(peer_va_mae)
            best_peer_val_mse = min(best_peer_val_mse, peer_va_mse)

        if relay_stage_epochs[0] > 0 and epoch == relay_stage_epochs[0]:
            torch.save(model.state_dict(), run_dir / "relay_stage0_model.pt")
            if peer_model is not None:
                torch.save(peer_model.state_dict(), run_dir / "relay_stage0_peer_model.pt")
        if relay_stage_epochs[1] > 0 and epoch == relay_stage_epochs[0] + relay_stage_epochs[1]:
            torch.save(model.state_dict(), run_dir / "relay_stage1_model.pt")
            if peer_model is not None:
                torch.save(peer_model.state_dict(), run_dir / "relay_stage1_peer_model.pt")

        status = (
            f"[operator][epoch {epoch:03d}] lambda={effective_lambda:.4f} "
            f"relay_stage={relay_stage} relay_scale={relay_scale:.3f} "
            f"relay_hotspot={train_stats['relay_hotspot_ratio']:.4f} "
            f"train_mse={tr_mse:.8f} "
            f"val_mse={va_mse:.8f} val_mae={va_mae:.8f}"
        )
        if peer_va_mse is not None and peer_va_mae is not None:
            status += f" | peer_val_mse={peer_va_mse:.8f} peer_val_mae={peer_va_mae:.8f}"
        print(status)
        append_jsonl(
            epoch_log_path,
            {
                "epoch": epoch,
                "method": args.method,
                "dataset": args.dataset,
                "model": args.model,
                "peer_model": pair_meta["peer_model"],
                "lambda_imitation": effective_lambda,
                "relay_stage": relay_stage,
                "relay_scale": relay_scale,
                "relay_stage_epochs": list(relay_stage_epochs),
                "relay_hint_mode": args.relay_hint_mode,
                "relay_taper_schedule": args.relay_taper_schedule,
                "operator_weight_granularity": args.operator_weight_granularity,
                "hetero_ssml_one_way": hetero_ssml_one_way,
                "ssml_student_only": args.ssml_student_only,
                "ssml_freeze_peer": args.ssml_freeze_peer,
                "train_mse": tr_mse,
                "relay_hotspot_ratio": train_stats["relay_hotspot_ratio"],
                "val_mse": va_mse,
                "val_mae": va_mae,
                "peer_val_mse": peer_va_mse,
                "peer_val_mae": peer_va_mae,
            },
        )

        if epoch % args.live_plot_interval == 0 or epoch == args.epochs:
            save_curves(
                run_dir / "curves.npz",
                train_mse=train_mse_curve,
                val_mse=val_mse_curve,
                val_mae=val_mae_curve,
                train_mse1=train_mse_curve,
                val_mse1=val_mse_curve,
                val_mae1=val_mae_curve,
                val_mse2=peer_val_mse_curve,
                val_mae2=peer_val_mae_curve,
            )
            saved = save_live_loss_plot(
                run_dir=run_dir,
                task="operator",
                seed=args.seed,
            )
            if saved:
                print(f"[operator][epoch {epoch:03d}] updated live plot")
            else:
                print(f"[operator][epoch {epoch:03d}] live plot skipped")
    save_curves(
        run_dir / "curves.npz",
        train_mse=train_mse_curve,
        val_mse=val_mse_curve,
        val_mae=val_mae_curve,
        train_mse1=train_mse_curve,
        val_mse1=val_mse_curve,
        val_mae1=val_mae_curve,
        val_mse2=peer_val_mse_curve,
        val_mae2=peer_val_mae_curve,
    )
    summary = {
        "task": "operator",
        "dataset": args.dataset,
        "method": args.method,
        "model": args.model,
        "peer_model": pair_meta["peer_model"],
        "pair_tag": pair_meta["pair_tag"],
        "pair_type": pair_meta["pair_type"],
        "is_joint_training": pair_meta["is_joint_training"],
        "is_heterogeneous_pair": pair_meta["is_heterogeneous_pair"],
        "curve_mode": "pair" if peer_model is not None else "single",
        "model_idx": 1,
        "model1": args.model,
        "model2": pair_meta["peer_model"],
        "regression_imitation_loss": args.regression_imitation_loss,
        "lambda_imitation": args.lambda_imitation,
        "margin": args.margin,
        "warmup_epochs": args.warmup_epochs,
        "imitation_decay_start_epoch": args.imitation_decay_start_epoch,
        "imitation_decay_end_epoch": args.imitation_decay_end_epoch,
        "imitation_decay_min_scale": args.imitation_decay_min_scale,
        "relay_stage_epochs": list(relay_stage_epochs),
        "relay_hint_mode": args.relay_hint_mode,
        "relay_taper_schedule": args.relay_taper_schedule,
        "hetero_ssml_one_way": hetero_ssml_one_way,
        "ssml_student_only": args.ssml_student_only,
        "ssml_freeze_peer": args.ssml_freeze_peer,
        "peer_update_disabled": peer_update_disabled,
        "operator_weight_granularity": args.operator_weight_granularity,
        "dml_rule": "supervised_all_plus_soft_peer_better_imitation",
        "ssml_rule": "supervised_all_plus_hard_peer_better_imitation",
        "ssml_directionality": "hetero_weaker_to_stronger_only" if hetero_ssml_one_way else "bidirectional",
        "epochs": args.epochs,
        "seed": args.seed,
        "epoch_log_path": str(epoch_log_path),
        "best_val_mse": best_val_mse,
        "final_val_mse": val_mse_curve[-1],
        "final_val_mae": val_mae_curve[-1],
        "best_control_val_mse": None if math.isinf(best_stage_val_mse["control"]) else best_stage_val_mse["control"],
        "best_relay_val_mse": None if math.isinf(best_stage_val_mse["relay"]) else best_stage_val_mse["relay"],
        "best_taper_val_mse": None if math.isinf(best_stage_val_mse["taper"]) else best_stage_val_mse["taper"],
        "best_finetune_val_mse": None if math.isinf(best_stage_val_mse["finetune"]) else best_stage_val_mse["finetune"],
        "best_metric": best_val_mse,
        "best_metric_key": "mse",
        "final_metric": val_mse_curve[-1],
        "best_metric1": best_val_mse,
        "final_metric1": val_mse_curve[-1],
        "best_val_mse1": best_val_mse,
        "final_val_mse1": val_mse_curve[-1],
        "final_val_mae1": val_mae_curve[-1],
        "final_val1": val_mse_curve[-1],
        "num_parameters": count_parameters(model),
        "num_parameters1": count_parameters(model),
        "init_checkpoint": loaded_init_checkpoint,
        "peer_init_checkpoint": loaded_peer_init_checkpoint,
        "relay_stage0_model_path": str(run_dir / "relay_stage0_model.pt"),
        "relay_stage1_model_path": str(run_dir / "relay_stage1_model.pt"),
        "meta": meta,
    }
    if peer_model is not None:
        summary.update(
            {
                "best_metric2": best_peer_val_mse,
                "final_metric2": peer_val_mse_curve[-1],
                "best_val_mse2": best_peer_val_mse,
                "final_val_mse2": peer_val_mse_curve[-1],
                "final_val_mae2": peer_val_mae_curve[-1],
                "final_val2": peer_val_mse_curve[-1],
                "num_parameters2": count_parameters(peer_model),
            }
        )
    save_json(run_dir / "summary.json", summary)
    torch.save(model.state_dict(), run_dir / "model.pt")
    if peer_model is not None:
        torch.save(peer_model.state_dict(), run_dir / "peer_model.pt")
    print("[operator] done")


if __name__ == "__main__":
    main()
