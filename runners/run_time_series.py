from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.methods import get_directional_weight_builder, mask_activation_ratio, weighted_mean
from src.models import build_time_series_model
from src.tasks import TimeSeriesDataConfig, build_time_series_dataloaders
from src.utils import (
    append_jsonl,
    count_parameters,
    make_run_dir,
    save_curves,
    save_json,
    save_live_loss_plot,
    set_seed,
)

TIME_SERIES_MODEL_CHOICES = ["dlinear", "transformer"]
TIME_SERIES_METHOD_CHOICES = ["independent", "naive", "dml", "studygroup"]


def parse_args():
    p = argparse.ArgumentParser(description="Run time-series forecasting experiment")
    p.add_argument(
        "--dataset",
        type=str,
        default="etth1",
        choices=["etth1", "etth2", "ettm1", "ettm2", "electricity", "weather", "traffic", "exchange_rate", "illness"],
    )
    p.add_argument("--model", type=str, default="dlinear", choices=TIME_SERIES_MODEL_CHOICES)
    p.add_argument("--peer-model", type=str, default=None, choices=TIME_SERIES_MODEL_CHOICES)
    p.add_argument("--method", type=str, default="dml", choices=TIME_SERIES_METHOD_CHOICES)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="results/experiments")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--feature-mode", type=str, default="multivariate", choices=["multivariate", "univariate"])
    p.add_argument("--target-column", type=str, default=None)
    p.add_argument("--regression-imitation-loss", type=str, default="mse", choices=["mse", "mae", "huber"])
    p.add_argument("--lambda-imitation", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--live-plot-interval", type=int, default=20)
    return p.parse_args()


def build_regression_imitation_loss_fn(
    imitation_loss_name: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def _reduce_per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
        if loss_tensor.ndim <= 1:
            return loss_tensor.reshape(-1)
        return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)

    if imitation_loss_name == "mse":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return _reduce_per_sample(F.mse_loss(pred, peer_pred.detach(), reduction="none"))

        return _loss

    if imitation_loss_name == "mae":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return _reduce_per_sample(F.l1_loss(pred, peer_pred.detach(), reduction="none"))

        return _loss

    if imitation_loss_name == "huber":
        def _loss(pred: torch.Tensor, peer_pred: torch.Tensor) -> torch.Tensor:
            return _reduce_per_sample(F.smooth_l1_loss(pred, peer_pred.detach(), reduction="none"))

        return _loss

    raise ValueError(f"Unsupported regression imitation loss: {imitation_loss_name}")


def _positive_activation_ratio(weights: torch.Tensor) -> float:
    if weights.numel() == 0:
        return 0.0
    return float((weights > 0).to(dtype=torch.float32).mean().item())


def train_one_epoch(
    model,
    peer_model: Optional[torch.nn.Module],
    loader,
    optimizer,
    peer_optimizer: Optional[torch.optim.Optimizer],
    device,
    supervised_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_imitation: float,
    margin: float,
    method: str,
):
    model.train()
    if peer_model is not None:
        peer_model.train()
    directional_weight_builder = get_directional_weight_builder(method)

    total_train_objective = 0.0
    total_supervised = 0.0
    total_imitation = 0.0
    total_mean_weight = 0.0
    total_active_ratio = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if peer_optimizer is not None:
            peer_optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        supervised_loss = supervised_loss_fn(pred, y)

        if method == "independent":
            loss = supervised_loss.mean()
            imitation_term_student = supervised_loss.new_tensor(0.0)
            w_student = torch.zeros_like(supervised_loss)
            loss.backward()
            optimizer.step()

        else:
            if peer_model is None or peer_optimizer is None:
                raise ValueError(f"peer_model and peer_optimizer are required when method='{method}'")
            peer_pred = peer_model(x)
            peer_supervised_loss = supervised_loss_fn(peer_pred, y)

            w_student, w_peer = directional_weight_builder(
                supervised_loss.detach(),
                peer_supervised_loss.detach(),
                margin=margin,
            )
            if lambda_imitation <= 0.0:
                w_student = torch.zeros_like(w_student)
                w_peer = torch.zeros_like(w_peer)

            sample_loss = imitation_loss_fn(pred, peer_pred)
            imitation_term_student = weighted_mean(sample_loss, w_student)
            loss = supervised_loss.mean() + lambda_imitation * imitation_term_student

            sample_loss_peer = imitation_loss_fn(peer_pred, pred)
            imitation_term_peer = weighted_mean(sample_loss_peer, w_peer)
            peer_loss = peer_supervised_loss.mean() + lambda_imitation * imitation_term_peer

            (loss + peer_loss).backward()
            optimizer.step()
            peer_optimizer.step()

        batch_size = x.size(0)
        total_train_objective += float(loss.item()) * batch_size
        total_supervised += float(supervised_loss.mean().item()) * batch_size
        total_imitation += float(imitation_term_student.item()) * batch_size
        total_mean_weight += mask_activation_ratio(w_student) * batch_size
        total_active_ratio += _positive_activation_ratio(w_student) * batch_size
        total_count += batch_size

    denom = max(total_count, 1)
    return {
        "train_total": total_train_objective / denom,
        "supervised_loss_mean": total_supervised / denom,
        "imitation_loss_mean": total_imitation / denom,
        "mean_imitation_weight": total_mean_weight / denom,
        "active_imitation_ratio": total_active_ratio / denom,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for x, y in loader:
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
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = build_time_series_dataloaders(
        TimeSeriesDataConfig(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sequence_length=args.seq_len,
            prediction_length=args.pred_len,
            feature_mode=args.feature_mode,
            target_column=args.target_column,
        )
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    meta = data["meta"]

    peer_model_name = args.peer_model or args.model
    model = build_time_series_model(
        model_name=args.model,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_features=int(meta["num_features"]),
        num_targets=int(meta["num_targets"]),
    ).to(device)
    peer_model = None
    peer_optimizer = None
    if args.method != "independent":
        peer_model = build_time_series_model(
            model_name=peer_model_name,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_features=int(meta["num_features"]),
            num_targets=int(meta["num_targets"]),
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if peer_model is not None:
        peer_optimizer = torch.optim.AdamW(peer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    supervised_loss_fn = build_regression_imitation_loss_fn("mse")
    imitation_loss_fn = build_regression_imitation_loss_fn(args.regression_imitation_loss)

    summary_peer_model = peer_model_name if peer_model is not None else None
    pair_tag = args.model if summary_peer_model in {None, args.model} else f"{args.model}__{summary_peer_model}"
    run_dir = make_run_dir(
        args.output_dir,
        "time_series",
        args.dataset,
        f"{pair_tag}_{args.method}_{args.regression_imitation_loss}_seed{args.seed}",
    )
    print(f"[time_series] run_dir={run_dir}")
    print(f"[time_series] params={count_parameters(model)}")

    epoch_log_path = Path(run_dir) / "epoch_metrics.jsonl"
    if epoch_log_path.exists():
        epoch_log_path.unlink()

    train_total_curve = []
    train_mse_curve = []
    train_sup_curve = []
    train_imitation_curve = []
    train_mean_weight_curve = []
    train_active_ratio_curve = []
    val_mse_curve = []
    val_mae_curve = []
    best_val_mse = float("inf")
    last_train_stats = {
        "train_total": 0.0,
        "supervised_loss_mean": 0.0,
        "imitation_loss_mean": 0.0,
        "mean_imitation_weight": 0.0,
        "active_imitation_ratio": 0.0,
    }

    for epoch in range(1, args.epochs + 1):
        effective_lambda = args.lambda_imitation
        if args.method == "independent":
            effective_lambda = 0.0
        elif args.method == "studygroup" and epoch <= args.warmup_epochs:
            effective_lambda = 0.0

        train_stats = train_one_epoch(
            model,
            peer_model,
            train_loader,
            optimizer,
            peer_optimizer,
            device,
            supervised_loss_fn=supervised_loss_fn,
            imitation_loss_fn=imitation_loss_fn,
            lambda_imitation=effective_lambda,
            margin=args.margin,
            method=args.method,
        )
        last_train_stats = train_stats
        va_mse, va_mae = evaluate(model, val_loader, device)
        train_total_curve.append(train_stats["train_total"])
        train_mse_curve.append(train_stats["supervised_loss_mean"])
        train_sup_curve.append(train_stats["supervised_loss_mean"])
        train_imitation_curve.append(train_stats["imitation_loss_mean"])
        train_mean_weight_curve.append(train_stats["mean_imitation_weight"])
        train_active_ratio_curve.append(train_stats["active_imitation_ratio"])
        val_mse_curve.append(va_mse)
        val_mae_curve.append(va_mae)
        best_val_mse = min(best_val_mse, va_mse)

        print(
            f"[time_series][epoch {epoch:03d}] "
            f"train_total={train_stats['train_total']:.8f} "
            f"train_sup={train_stats['supervised_loss_mean']:.8f} "
            f"train_imit={train_stats['imitation_loss_mean']:.8f} "
            f"mean_w={train_stats['mean_imitation_weight']:.4f} "
            f"active={train_stats['active_imitation_ratio']:.4f} "
            f"val_mse={va_mse:.8f} val_mae={va_mae:.8f}"
        )
        append_jsonl(
            epoch_log_path,
            {
                "epoch": epoch,
                "method": args.method,
                "model": args.model,
                "peer_model": summary_peer_model,
                "lambda_imitation": effective_lambda,
                "margin": args.margin,
                "train_total": train_stats["train_total"],
                "supervised_loss_mean": train_stats["supervised_loss_mean"],
                "imitation_loss_mean": train_stats["imitation_loss_mean"],
                "mean_imitation_weight": train_stats["mean_imitation_weight"],
                "active_imitation_ratio": train_stats["active_imitation_ratio"],
                "val_mse": va_mse,
                "val_mae": va_mae,
            },
        )

        if epoch % args.live_plot_interval == 0 or epoch == args.epochs:
            save_curves(
                run_dir / "curves.npz",
                train_total=train_total_curve,
                train_mse=train_mse_curve,
                train_sup=train_sup_curve,
                train_imitation_loss=train_imitation_curve,
                train_mean_imitation_weight=train_mean_weight_curve,
                train_active_imitation_ratio=train_active_ratio_curve,
                val_mse=val_mse_curve,
                val_mae=val_mae_curve,
            )
            saved = save_live_loss_plot(
                run_dir=run_dir,
                task="time_series",
                seed=args.seed,
            )
            if saved:
                print(f"[time_series][epoch {epoch:03d}] updated live plot")
            else:
                print(f"[time_series][epoch {epoch:03d}] live plot skipped")
    save_curves(
        run_dir / "curves.npz",
        train_total=train_total_curve,
        train_mse=train_mse_curve,
        train_sup=train_sup_curve,
        train_imitation_loss=train_imitation_curve,
        train_mean_imitation_weight=train_mean_weight_curve,
        train_active_imitation_ratio=train_active_ratio_curve,
        val_mse=val_mse_curve,
        val_mae=val_mae_curve,
    )
    save_json(
        run_dir / "summary.json",
        {
            "task": "time_series",
            "dataset": args.dataset,
            "method": args.method,
            "model": args.model,
            "peer_model": summary_peer_model,
            "mean_imitation_weight": last_train_stats["mean_imitation_weight"],
            "active_imitation_ratio": last_train_stats["active_imitation_ratio"],
            "supervised_loss_mean": last_train_stats["supervised_loss_mean"],
            "imitation_loss_mean": last_train_stats["imitation_loss_mean"],
            "curve_mode": "single",
            "model_idx": 1,
            "regression_imitation_loss": args.regression_imitation_loss,
            "lambda_imitation": args.lambda_imitation,
            "margin": args.margin,
            "warmup_epochs": args.warmup_epochs,
            "epochs": args.epochs,
            "seed": args.seed,
            "epoch_log_path": str(epoch_log_path),
            "best_val_mse": best_val_mse,
            "final_val_mse": val_mse_curve[-1],
            "final_val_mae": val_mae_curve[-1],
            "best_metric": best_val_mse,
            "best_metric_key": "mse",
            "final_metric": val_mse_curve[-1],
            "num_parameters": count_parameters(model),
            "meta": meta,
        },
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    print("[time_series] done")


if __name__ == "__main__":
    main()
