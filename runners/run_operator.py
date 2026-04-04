from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.methods import get_directional_weight_builder, weighted_mean
from src.models.operator import build_operator_model
from src.tasks.operator import OperatorDataConfig, build_operator_dataloaders
from src.utils import (
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
            if lambda_imitation <= 0.0:
                w_student = torch.zeros_like(w_student)
                w_peer = torch.zeros_like(w_peer)
            imitation_student_values = (
                elementwise_imitation_loss_fn(pred, peer_pred)
                if operator_weight_granularity == "element"
                else imitation_loss_fn(pred, peer_pred)
            )
            imitation_student = weighted_mean(imitation_student_values, w_student)

            loss = supervised_loss.mean() + lambda_imitation * imitation_student
            if peer_update_disabled:
                loss.backward()
                optimizer.step()
            else:
                imitation_peer_values = (
                    elementwise_imitation_loss_fn(peer_pred, pred)
                    if operator_weight_granularity == "element"
                    else imitation_loss_fn(peer_pred, pred)
                )
                imitation_peer = weighted_mean(imitation_peer_values, w_peer)
                peer_loss = peer_supervised_loss.mean() + lambda_imitation * imitation_peer

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
            if lambda_imitation <= 0.0:
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

            imitation_student_values = (
                elementwise_imitation_loss_fn(pred, peer_pred)
                if operator_weight_granularity == "element"
                else imitation_loss_fn(pred, peer_pred)
            )
            imitation_term_student = weighted_mean(imitation_student_values, w_student)
            loss = supervised_loss.mean() + lambda_imitation * imitation_term_student

            if peer_update_disabled:
                loss.backward()
                optimizer.step()
            else:
                imitation_peer_values = (
                    elementwise_imitation_loss_fn(peer_pred, pred)
                    if operator_weight_granularity == "element"
                    else imitation_loss_fn(peer_pred, pred)
                )
                imitation_term_peer = weighted_mean(imitation_peer_values, w_peer)
                peer_loss = peer_supervised_loss.mean() + lambda_imitation * imitation_term_peer

                (loss + peer_loss).backward()
                optimizer.step()
                peer_optimizer.step()
        else:
            raise ValueError(f"Unsupported method '{method}'")

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / total_count


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
        f"peer_updates_disabled={peer_update_disabled}"
    )

    train_mse_curve = []
    val_mse_curve = []
    val_mae_curve = []
    peer_val_mse_curve = []
    peer_val_mae_curve = []
    best_val_mse = float("inf")
    best_peer_val_mse = float("inf")
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

        tr_mse = train_one_epoch(
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
            hetero_ssml_one_way=hetero_ssml_one_way,
            peer_update_disabled=peer_update_disabled,
            operator_weight_granularity=args.operator_weight_granularity,
        )
        va_mse, va_mae = evaluate(model, val_loader, device)
        peer_va_mse = None
        peer_va_mae = None
        if peer_model is not None:
            peer_va_mse, peer_va_mae = evaluate(peer_model, val_loader, device)

        train_mse_curve.append(tr_mse)
        val_mse_curve.append(va_mse)
        val_mae_curve.append(va_mae)
        best_val_mse = min(best_val_mse, va_mse)
        if peer_va_mse is not None and peer_va_mae is not None:
            peer_val_mse_curve.append(peer_va_mse)
            peer_val_mae_curve.append(peer_va_mae)
            best_peer_val_mse = min(best_peer_val_mse, peer_va_mse)

        status = (
            f"[operator][epoch {epoch:03d}] lambda={effective_lambda:.4f} "
            f"train_mse={tr_mse:.8f} "
            f"val_mse={va_mse:.8f} val_mae={va_mae:.8f}"
        )
        if peer_va_mse is not None and peer_va_mae is not None:
            status += f" | peer_val_mse={peer_va_mse:.8f} peer_val_mae={peer_va_mae:.8f}"
        print(status)

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
        "best_val_mse": best_val_mse,
        "final_val_mse": val_mse_curve[-1],
        "final_val_mae": val_mae_curve[-1],
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
