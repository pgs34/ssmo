from __future__ import annotations

import argparse
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.models import build_classification_model
from src.tasks import ClassificationDataConfig, build_classification_dataloaders
from src.utils import count_parameters, make_run_dir, save_curves, save_json, save_live_loss_plot, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run classification experiment")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, default="resnet18", choices=["simple_cnn", "simple_mlp", "resnet18", "vit_b16"])
    p.add_argument(
        "--method",
        type=str,
        default="dml",
        choices=["naive", "dml", "studygroup"],
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="results/experiments")
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-subset-size", type=int, default=None)
    p.add_argument("--val-subset-size", type=int, default=None)
    p.add_argument("--label-noise-type", type=str, default=None, choices=[None, "symmetric", "asymmetric"])
    p.add_argument("--label-noise-rate", type=float, default=0.0)
    p.add_argument("--classification-imitation-loss", type=str, default="kl", choices=["kl", "js", "mse_logits"])
    p.add_argument("--lambda-imitation", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--live-plot-interval", type=int, default=20)
    return p.parse_args()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def build_imitation_loss_fn(imitation_loss_name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if imitation_loss_name == "kl":
        def _loss(logits: torch.Tensor, peer_logits: torch.Tensor) -> torch.Tensor:
            teacher_prob = F.softmax(peer_logits, dim=1).detach()
            return F.kl_div(
                F.log_softmax(logits, dim=1),
                teacher_prob,
                reduction="none",
            ).sum(dim=1)

        return _loss

    if imitation_loss_name == "js":
        def _loss(logits: torch.Tensor, peer_logits: torch.Tensor) -> torch.Tensor:
            student_prob = torch.clamp(F.softmax(logits, dim=1), min=1e-8, max=1.0)
            teacher_prob = torch.clamp(F.softmax(peer_logits, dim=1).detach(), min=1e-8, max=1.0)
            mix = torch.clamp((student_prob + teacher_prob) * 0.5, min=1e-8, max=1.0)
            return 0.5 * (
                F.kl_div(torch.log(student_prob), mix, reduction="none").sum(dim=1)
                + F.kl_div(torch.log(teacher_prob), mix, reduction="none").sum(dim=1)
            )

        return _loss

    if imitation_loss_name == "mse_logits":
        def _loss(logits: torch.Tensor, peer_logits: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(logits, peer_logits.detach(), reduction="none").mean(dim=1)

        return _loss

    raise ValueError(f"Unsupported classification imitation loss: {imitation_loss_name}")


def train_one_epoch(
    model,
    peer_model: Optional[torch.nn.Module],
    loader,
    optimizer,
    peer_optimizer: Optional[torch.optim.Optimizer],
    device,
    supervised_loss_fn: Callable[..., torch.Tensor],
    imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_imitation: float,
    margin: float,
    method: str,
) -> tuple[float, float]:
    model.train()
    if peer_model is not None:
        peer_model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if peer_optimizer is not None:
            peer_optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        supervised_loss = supervised_loss_fn(logits, y)

        if method == "naive":
            # Supervised only: no imitation loss is used.
            loss = supervised_loss.mean()
            loss.backward()
            optimizer.step()

        elif method == "dml":
            if peer_model is None or peer_optimizer is None:
                raise ValueError("peer_model and peer_optimizer are required when method='dml'")
            # DML: peer-superior samples are weighted more, weakly proportional to the gap.
            peer_logits = peer_model(x)
            peer_supervised_loss = supervised_loss_fn(peer_logits, y)

            # Keep weighting non-differentiable to avoid crossing the two model graphs.
            w_student = torch.relu(peer_supervised_loss.detach() - supervised_loss.detach() - margin)
            sample_loss = imitation_loss_fn(logits, peer_logits)
            student_weight_sum = w_student.sum()
            imitation_term_student = (
                (sample_loss * w_student).sum() / (student_weight_sum + 1e-12)
                if float(student_weight_sum.item()) > 0.0
                else sample_loss.new_tensor(0.0)
            )
            loss = supervised_loss.mean() + lambda_imitation * imitation_term_student

            w_peer = torch.relu(supervised_loss.detach() - peer_supervised_loss.detach() - margin)
            sample_loss_peer = imitation_loss_fn(peer_logits, logits)
            peer_weight_sum = w_peer.sum()
            imitation_term_peer = (
                (sample_loss_peer * w_peer).sum() / (peer_weight_sum + 1e-12)
                if float(peer_weight_sum.item()) > 0.0
                else sample_loss_peer.new_tensor(0.0)
            )
            peer_loss = peer_supervised_loss.mean() + lambda_imitation * imitation_term_peer

            (loss + peer_loss).backward()
            optimizer.step()
            peer_optimizer.step()

        elif method == "studygroup":
            if peer_model is None or peer_optimizer is None:
                raise ValueError("peer_model and peer_optimizer are required when method='studygroup'")
            # Notebook-parity StudyGroup:
            # index1 = self correct / peer wrong, index2 = self wrong / peer correct, index3 = others.
            # self loss = KD(index2) + CE(index3), peer loss = KD(index1) + CE(index3)
            peer_logits = peer_model(x)
            peer_supervised_loss = supervised_loss_fn(peer_logits, y)

            pred_student = logits.argmax(dim=1)
            pred_peer = peer_logits.argmax(dim=1)
            correct_student = pred_student == y
            correct_peer = pred_peer == y

            index1 = correct_student & ~correct_peer
            index2 = ~correct_student & correct_peer
            index3 = ~(index1 | index2)

            sample_loss_student = imitation_loss_fn(logits, peer_logits)
            sample_loss_peer = imitation_loss_fn(peer_logits, logits)

            student_loss_terms = []
            if index2.any():
                student_loss_terms.append(lambda_imitation * sample_loss_student[index2].mean())
            if index3.any():
                student_loss_terms.append(supervised_loss[index3].mean())
            if student_loss_terms:
                loss = torch.stack(student_loss_terms).sum()
            else:
                loss = supervised_loss.new_tensor(0.0)

            peer_loss_terms = []
            if index1.any():
                peer_loss_terms.append(lambda_imitation * sample_loss_peer[index1].mean())
            if index3.any():
                peer_loss_terms.append(peer_supervised_loss[index3].mean())
            if peer_loss_terms:
                peer_loss = torch.stack(peer_loss_terms).sum()
            else:
                peer_loss = peer_supervised_loss.new_tensor(0.0)

            (loss + peer_loss).backward()
            optimizer.step()
            peer_optimizer.step()

        else:
            raise ValueError(f"Unsupported method '{method}'")

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy(logits, y) * batch_size
        total_count += batch_size
    return total_loss / total_count, total_acc / total_count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy(logits, y) * batch_size
        total_count += batch_size
    return total_loss / total_count, total_acc / total_count


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = build_classification_dataloaders(
        ClassificationDataConfig(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
            train_subset_size=args.train_subset_size,
            val_subset_size=args.val_subset_size,
            seed=args.seed,
            label_noise_type=args.label_noise_type,
            label_noise_rate=args.label_noise_rate,
        )
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    num_classes = int(data["meta"]["num_classes"])

    sample_x, _ = next(iter(train_loader))
    in_channels = int(sample_x.shape[1])
    image_size = int(sample_x.shape[-1])

    model = build_classification_model(
        model_name=args.model,
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size,
    ).to(device)
    peer_model = None
    peer_optimizer = None
    if args.method in {"dml", "studygroup"}:
        peer_model = build_classification_model(
            model_name=args.model,
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if peer_model is not None:
        peer_optimizer = torch.optim.AdamW(peer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = make_run_dir(
        args.output_dir,
        "classification",
        args.dataset,
        f"{args.model}_{args.method}_{args.classification_imitation_loss}_seed{args.seed}",
    )
    print(f"[classification] run_dir={run_dir}")
    print(f"[classification] params={count_parameters(model)}")

    train_loss_curve = []
    train_acc_curve = []
    val_loss_curve = []
    val_acc_curve = []
    best_val_acc = 0.0
    supervised_loss_fn = lambda logits, targets: F.cross_entropy(logits, targets, reduction="none")
    imitation_loss_fn = build_imitation_loss_fn(args.classification_imitation_loss)

    for epoch in range(1, args.epochs + 1):
        # Keep studygroup warmup semantics.
        effective_lambda = args.lambda_imitation
        if args.method == "naive":
            effective_lambda = 0.0
        elif args.method == "studygroup" and epoch <= args.warmup_epochs:
            effective_lambda = 0.0

        tr_loss, tr_acc = train_one_epoch(
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
        va_loss, va_acc = evaluate(model, val_loader, device)

        train_loss_curve.append(tr_loss)
        train_acc_curve.append(tr_acc)
        val_loss_curve.append(va_loss)
        val_acc_curve.append(va_acc)
        best_val_acc = max(best_val_acc, va_acc)

        print(
            f"[classification][epoch {epoch:03d}] "
            f"train_loss={tr_loss:.6f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.6f} val_acc={va_acc:.4f}"
        )

        if epoch % args.live_plot_interval == 0 or epoch == args.epochs:
            save_curves(
                run_dir / "curves.npz",
                train_loss=train_loss_curve,
                train_acc=train_acc_curve,
                val_loss=val_loss_curve,
                val_acc=val_acc_curve,
            )
            saved = save_live_loss_plot(
                run_dir=run_dir,
                task="classification",
                seed=args.seed,
            )
            if saved:
                print(f"[classification][epoch {epoch:03d}] updated live plot")
            else:
                print(f"[classification][epoch {epoch:03d}] live plot skipped")

    save_curves(
        run_dir / "curves.npz",
        train_loss=train_loss_curve,
        train_acc=train_acc_curve,
        val_loss=val_loss_curve,
        val_acc=val_acc_curve,
    )
    save_json(
        run_dir / "summary.json",
        {
            "task": "classification",
            "dataset": args.dataset,
            "method": args.method,
            "model": args.model,
            "peer_model": args.model if peer_model is not None else None,
            "curve_mode": "single",
            "model_idx": 1,
            "classification_imitation_loss": args.classification_imitation_loss,
            "lambda_imitation": args.lambda_imitation,
            "margin": args.margin,
            "warmup_epochs": args.warmup_epochs,
            "epochs": args.epochs,
            "seed": args.seed,
            "best_val_acc": best_val_acc,
            "final_val_acc": val_acc_curve[-1],
            "final_val_loss": val_loss_curve[-1],
            "best_metric": best_val_acc,
            "best_metric_key": "acc",
            "final_metric": val_acc_curve[-1],
            "num_parameters": count_parameters(model),
            "meta": data["meta"],
        },
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    print("[classification] done")


if __name__ == "__main__":
    main()
