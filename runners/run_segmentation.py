from __future__ import annotations

import argparse
from itertools import islice
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models import build_segmentation_model
from src.tasks import SegmentationDataConfig, build_segmentation_dataloaders
from src.utils import count_parameters, make_run_dir, save_curves, save_json, save_live_loss_plot, set_seed

IGNORE_INDEX = 255


def parse_args():
    p = argparse.ArgumentParser(description="Run semantic segmentation experiment")
    p.add_argument("--train-dataset", type=str, default="voc", choices=["voc", "cityscapes"])
    p.add_argument("--val-dataset", type=str, default="voc", choices=["voc", "cityscapes"])
    p.add_argument("--model", type=str, default="unet", choices=["unet", "deeplabv3_resnet50"])
    p.add_argument("--method", type=str, default="independent", choices=["independent", "naive", "dml", "studygroup"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="results/experiments")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--download-voc", action="store_true")
    p.add_argument("--train-corruption", type=str, default=None)
    p.add_argument("--val-corruption", type=str, default=None)
    p.add_argument("--train-corruption-severity", type=int, default=1)
    p.add_argument("--val-corruption-severity", type=int, default=1)
    p.add_argument("--train-resolution-scale", type=float, default=1.0)
    p.add_argument("--val-resolution-scale", type=float, default=1.0)
    p.add_argument("--segmentation-imitation-loss", type=str, default="kl", choices=["kl", "js", "mse_logits"])
    p.add_argument("--lambda-imitation", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--live-plot-interval", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    return p.parse_args()


def model_forward(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, dict):
        if "out" not in out:
            raise KeyError("Segmentation model output dict must contain key 'out'.")
        return out["out"]
    return out


def update_iou_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    pred = logits.argmax(dim=1)
    valid = target != IGNORE_INDEX
    pred = pred[valid]
    target = target[valid]

    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection[c] = float((pred_c & target_c).sum().item())
        union[c] = float((pred_c | target_c).sum().item())
    pixel_acc = float((pred == target).float().mean().item()) if target.numel() > 0 else 0.0
    return intersection, union, pixel_acc


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model_forward(model, x)
        loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    pixel_acc_sum = 0.0
    pixel_acc_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model_forward(model, x)
        loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)

        inter, uni, pixel_acc = update_iou_stats(logits, y, num_classes)
        total_intersection += inter
        total_union += uni
        pixel_acc_sum += pixel_acc
        pixel_acc_count += 1

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    iou = total_intersection / np.maximum(total_union, 1e-8)
    miou = float(np.mean(iou))
    pixel_acc = float(pixel_acc_sum / max(pixel_acc_count, 1))
    return total_loss / total_count, miou, pixel_acc


def _iter_limited(loader, max_batches: int | None):
    if max_batches is not None and max_batches > 0:
        return islice(loader, max_batches)
    return loader


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = build_segmentation_dataloaders(
        SegmentationDataConfig(
            train_dataset=args.train_dataset,
            val_dataset=args.val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_height=args.height,
            input_width=args.width,
            download_voc=args.download_voc,
            train_corruption=args.train_corruption,
            val_corruption=args.val_corruption,
            train_corruption_severity=args.train_corruption_severity,
            val_corruption_severity=args.val_corruption_severity,
            train_resolution_scale=args.train_resolution_scale,
            val_resolution_scale=args.val_resolution_scale,
        )
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    train_meta = data["train_meta"]
    val_meta = data["val_meta"]

    sample_x, _ = next(iter(train_loader))
    in_channels = int(sample_x.shape[1])
    num_classes = int(val_meta["num_classes"])

    model = build_segmentation_model(args.model, num_classes=num_classes, in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_tag = f"{args.train_dataset}_to_{args.val_dataset}"
    run_dir = make_run_dir(args.output_dir, "segmentation", dataset_tag, f"{args.model}/{args.method}/seed{args.seed}")
    print(f"[segmentation] run_dir={run_dir}")
    print(f"[segmentation] params={count_parameters(model)}")

    train_loss_curve = []
    val_loss_curve = []
    val_miou_curve = []
    val_pixel_acc_curve = []
    best_val_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, _iter_limited(train_loader, args.max_train_batches), optimizer, device)
        va_loss, va_miou, va_pixel_acc = evaluate(
            model,
            _iter_limited(val_loader, args.max_val_batches),
            device,
            num_classes=num_classes,
        )
        train_loss_curve.append(tr_loss)
        val_loss_curve.append(va_loss)
        val_miou_curve.append(va_miou)
        val_pixel_acc_curve.append(va_pixel_acc)
        best_val_miou = max(best_val_miou, va_miou)

        print(
            f"[segmentation][epoch {epoch:03d}] "
            f"train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
            f"val_miou={va_miou:.6f} val_pixel_acc={va_pixel_acc:.6f}"
        )

        if epoch % args.live_plot_interval == 0 or epoch == args.epochs:
            save_curves(
                run_dir / "curves.npz",
                train_loss=train_loss_curve,
                val_loss=val_loss_curve,
                val_miou=val_miou_curve,
                val_pixel_acc=val_pixel_acc_curve,
            )
            saved = save_live_loss_plot(
                run_dir=run_dir,
                task="segmentation",
                seed=args.seed,
            )
            if saved:
                print(f"[segmentation][epoch {epoch:03d}] updated live plot")
            else:
                print(f"[segmentation][epoch {epoch:03d}] live plot skipped")
    save_curves(
        run_dir / "curves.npz",
        train_loss=train_loss_curve,
        val_loss=val_loss_curve,
        val_miou=val_miou_curve,
        val_pixel_acc=val_pixel_acc_curve,
    )
    save_json(
        run_dir / "summary.json",
        {
            "task": "segmentation",
            "dataset": dataset_tag,
            "train_dataset": args.train_dataset,
            "val_dataset": args.val_dataset,
            "method": args.method,
            "model": args.model,
            "peer_model": args.model,
            "curve_mode": "single",
            "model_idx": 1,
            "segmentation_imitation_loss": args.segmentation_imitation_loss,
            "lambda_imitation": args.lambda_imitation,
            "margin": args.margin,
            "warmup_epochs": args.warmup_epochs,
            "epochs": args.epochs,
            "seed": args.seed,
            "best_val_miou": best_val_miou,
            "final_val_miou": val_miou_curve[-1],
            "final_val_pixel_acc": val_pixel_acc_curve[-1],
            "final_val_loss": val_loss_curve[-1],
            "best_metric": best_val_miou,
            "best_metric_key": "miou",
            "final_metric": val_miou_curve[-1],
            "num_parameters": count_parameters(model),
            "train_meta": train_meta,
            "val_meta": val_meta,
        },
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    print("[segmentation] done")


if __name__ == "__main__":
    main()
