from __future__ import annotations

import argparse
from itertools import islice

import torch
import torch.nn.functional as F

from src.models.operator import build_operator_model
from src.tasks.operator import OperatorDataConfig, build_operator_dataloaders
from src.utils import count_parameters, make_run_dir, save_curves, save_json, save_live_loss_plot, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run operator-learning experiment")
    p.add_argument("--dataset", type=str, default="burgers", choices=["burgers", "darcy", "navier_stokes"])
    p.add_argument(
        "--model",
        type=str,
        default="fno",
        choices=[
            "fno",
            "deeponet",
            "gnot",
            "neuralop_fno",
            "neuralop_tfno",
            "neuralop_uno",
            "uno",
        ],
    )
    p.add_argument("--method", type=str, default="independent", choices=["independent", "naive", "dml", "studygroup"])
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
    p.add_argument("--live-plot-interval", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
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


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        x, y = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

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


def _iter_limited(loader, max_batches: int | None):
    if max_batches is not None and max_batches > 0:
        return islice(loader, max_batches)
    return loader


def main():
    args = parse_args()
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

    model = build_operator_model(args.model, args.dataset, meta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = make_run_dir(args.output_dir, "operator", args.dataset, f"{args.model}/{args.method}/seed{args.seed}")
    print(f"[operator] run_dir={run_dir}")
    print(f"[operator] params={count_parameters(model)}")

    train_mse_curve = []
    val_mse_curve = []
    val_mae_curve = []
    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_mse = train_one_epoch(
            model,
            _iter_limited(train_loader, args.max_train_batches),
            optimizer,
            device,
        )
        va_mse, va_mae = evaluate(model, _iter_limited(val_loader, args.max_val_batches), device)

        train_mse_curve.append(tr_mse)
        val_mse_curve.append(va_mse)
        val_mae_curve.append(va_mae)
        best_val_mse = min(best_val_mse, va_mse)

        print(
            f"[operator][epoch {epoch:03d}] train_mse={tr_mse:.8f} "
            f"val_mse={va_mse:.8f} val_mae={va_mae:.8f}"
        )

        if epoch % args.live_plot_interval == 0 or epoch == args.epochs:
            save_curves(
                run_dir / "curves.npz",
                train_mse=train_mse_curve,
                val_mse=val_mse_curve,
                val_mae=val_mae_curve,
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
    )
    save_json(
        run_dir / "summary.json",
        {
            "task": "operator",
            "dataset": args.dataset,
            "method": args.method,
            "model": args.model,
            "peer_model": args.model,
            "curve_mode": "single",
            "model_idx": 1,
            "regression_imitation_loss": args.regression_imitation_loss,
            "lambda_imitation": args.lambda_imitation,
            "margin": args.margin,
            "warmup_epochs": args.warmup_epochs,
            "epochs": args.epochs,
            "seed": args.seed,
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
    print("[operator] done")


if __name__ == "__main__":
    main()
