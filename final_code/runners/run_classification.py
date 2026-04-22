from __future__ import annotations

import argparse
import copy
import math
import os
import socket
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.methods import get_directional_weight_builder, weighted_mean
from src.models import build_classification_model
from src.tasks import ClassificationDataConfig, build_classification_dataloaders
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

CLASSIFICATION_MODEL_CHOICES = [
    "simple_cnn",
    "simple_mlp",
    "naive_cnn",
    "naive_mlp",
    "ode_cnn",
    "resnet18",
    "resnet18_gelu",
    "resnet34",
    "resnet34_gelu",
    "resnet34_cifar",
    "resnet34_cifar_gelu",
    "vit_b16",
]
CLASSIFICATION_METHOD_CHOICES = ["independent", "dml", "ssml"]


def apply_batchnorm_eval(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            submodule.eval()


def log_runtime_environment(args, device: torch.device) -> None:
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    fallback_reason = "requested_device_available"
    if args.device.startswith("cuda") and not cuda_available:
        fallback_reason = "cuda_unavailable_fallback_to_cpu"
    elif device.type == "cpu":
        fallback_reason = "cpu_requested"

    print(
        "[classification][runtime] "
        f"pid={os.getpid()} "
        f"ppid={os.getppid()} "
        f"host={socket.gethostname()} "
        f"cwd={Path.cwd()} "
        f"requested_device={args.device} "
        f"resolved_device={device} "
        f"cuda_available={int(cuda_available)} "
        f"cuda_device_count={cuda_device_count} "
        f"cuda_visible_devices={cuda_visible_devices} "
        f"fallback_reason={fallback_reason}"
    )
    if cuda_available and device.type == "cuda":
        current_index = device.index if device.index is not None else torch.cuda.current_device()
        print(
            "[classification][runtime] "
            f"cuda_current_device={current_index} "
            f"cuda_device_name={torch.cuda.get_device_name(current_index)}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Run classification experiment")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, default="resnet18", choices=CLASSIFICATION_MODEL_CHOICES)
    p.add_argument("--peer-model", type=str, default=None, choices=CLASSIFICATION_MODEL_CHOICES)
    p.add_argument(
        "--method",
        type=str,
        default="dml",
        choices=CLASSIFICATION_METHOD_CHOICES,
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd_nesterov"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "cosine"])
    p.add_argument("--scheduler-warmup-epochs", type=int, default=0)
    p.add_argument("--scheduler-min-scale", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--model-ema-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="results/experiments")
    p.add_argument("--protocol-id", type=str, default="default")
    p.add_argument("--hardware-profile", type=str, default="")
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-aug-mode", type=str, default="basic", choices=["basic", "strong"])
    p.add_argument("--train-subset-size", type=int, default=None)
    p.add_argument("--val-subset-size", type=int, default=None)
    p.add_argument("--label-noise-type", type=str, default=None, choices=[None, "symmetric", "asymmetric"])
    p.add_argument("--label-noise-rate", type=float, default=0.0)
    p.add_argument("--classification-imitation-loss", type=str, default="kl", choices=["kl", "js", "mse_logits"])
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--lambda-imitation", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--imitation-decay-start-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-end-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-min-scale", type=float, default=1.0)
    p.add_argument("--freeze-bn-stats", action="store_true")
    p.add_argument("--freeze-bn-stats-until-epoch", type=int, default=-1)
    p.add_argument("--hetero-ssml-one-way", action="store_true")
    p.add_argument("--ssml-student-only", action="store_true")
    p.add_argument("--ssml-freeze-peer", action="store_true")
    p.add_argument("--ssml-worse-only-update", action="store_true")
    p.add_argument("--ssml-anchor-weight", type=float, default=0.0)
    p.add_argument("--ssml-topk-ratio", type=float, default=0.3)
    p.add_argument("--ssml-topk-ratio-start", type=float, default=None)
    p.add_argument("--ssml-topk-ratio-end", type=float, default=None)
    p.add_argument("--ssml-topk-ramp-start-epoch", type=int, default=-1)
    p.add_argument("--ssml-topk-ramp-end-epoch", type=int, default=-1)
    p.add_argument("--ssml-topk-scope", type=str, default="total", choices=["total", "positive"])
    p.add_argument("--ssml-supervised-hotspot-alpha", type=float, default=0.0)
    p.add_argument(
        "--ssml-supervised-weight-mode",
        type=str,
        default="score",
        choices=["score", "binary"],
    )
    p.add_argument(
        "--ssml-gate-score-mode",
        type=str,
        default="peer_better_student_error",
        choices=[
            "relative_gap",
            "peer_better_student_error",
            "peer_better_true_prob_gap",
            "peer_better_true_prob_gap_weighted",
            "peer_confident_student_uncertain",
            "useful_hard_sample",
            "useful_hard_sample_confident",
        ],
    )
    p.add_argument(
        "--ssml-score-transform",
        type=str,
        default="log1p",
        choices=["none", "sqrt", "log1p"],
    )
    p.add_argument(
        "--ssml-guidance-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "reweight_only"],
    )
    p.add_argument("--ssml-peer-correct-only", action="store_true")
    p.add_argument("--ssml-student-incorrect-only", action="store_true")
    p.add_argument("--ssml-student-true-prob-max", type=float, default=1.0)
    p.add_argument("--ssml-disagreement-only", action="store_true")
    p.add_argument("--ssml-class-balanced-topk", action="store_true")
    p.add_argument("--ssml-per-class-budget", type=int, default=0)
    p.add_argument("--ssml-disagreement-floor-ratio", type=float, default=0.0)
    p.add_argument("--ssml-deficit-ema-momentum", type=float, default=0.0)
    p.add_argument("--ssml-extra-class-budget-scale", type=float, default=0.0)
    p.add_argument("--ssml-complement-ramp-start-epoch", type=int, default=-1)
    p.add_argument("--ssml-complement-ramp-end-epoch", type=int, default=-1)
    p.add_argument("--ssml-secondary-peer-init-checkpoint", type=str, default=None)
    p.add_argument("--ssml-secondary-peer-require-same-label", action="store_true")
    p.add_argument("--ssml-secondary-peer-agreement-min", type=float, default=0.0)
    p.add_argument("--ssml-peer-true-prob-threshold", type=float, default=0.0)
    p.add_argument("--ssml-peer-true-prob-threshold-start", type=float, default=None)
    p.add_argument("--ssml-peer-true-prob-threshold-end", type=float, default=None)
    p.add_argument("--ssml-peer-student-prob-gap-min", type=float, default=0.0)
    p.add_argument("--ssml-peer-student-prob-gap-min-start", type=float, default=None)
    p.add_argument("--ssml-peer-student-prob-gap-min-end", type=float, default=None)
    p.add_argument("--ssml-aug-consistency-weight", type=float, default=0.0)
    p.add_argument("--ssml-aug-consistency-shift", type=int, default=0)
    p.add_argument("--ssml-aug-consistency-flip-prob", type=float, default=0.0)
    p.add_argument("--ssml-aug-consistency-noise-std", type=float, default=0.0)
    p.add_argument("--ssml-peer-aug-consistency-min", type=float, default=0.0)
    p.add_argument("--ssml-student-aug-consistency-max", type=float, default=1.0)
    p.add_argument("--ssml-peer-student-aug-consistency-gap-min", type=float, default=0.0)
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--peer-init-checkpoint", type=str, default=None)
    p.add_argument("--live-plot-interval", type=int, default=20)
    return p.parse_args()


def clone_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: Optional[torch.nn.Module], online_model: torch.nn.Module, decay: float) -> None:
    if ema_model is None or decay <= 0.0:
        return
    for ema_param, online_param in zip(ema_model.parameters(), online_model.parameters()):
        ema_param.mul_(decay).add_(online_param.detach(), alpha=1.0 - decay)
    for ema_buffer, online_buffer in zip(ema_model.buffers(), online_model.buffers()):
        ema_buffer.copy_(online_buffer)


def compute_epoch_lr(
    base_lr: float,
    *,
    epoch: int,
    total_epochs: int,
    scheduler_name: str,
    warmup_epochs: int,
    min_scale: float,
) -> float:
    if scheduler_name == "none":
        return float(base_lr)
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")

    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(0, min(int(warmup_epochs), total_epochs))
    min_scale = float(max(0.0, min(float(min_scale), 1.0)))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * (float(epoch) / float(max(warmup_epochs, 1)))

    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    if cosine_epochs == 1:
        progress = 0.0
    else:
        progress = (epoch - warmup_epochs - 1) / max(cosine_epochs - 1, 1)
    progress = float(max(0.0, min(1.0, progress)))
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    scale = min_scale + (1.0 - min_scale) * cosine_scale
    return float(base_lr) * scale


def set_optimizer_lr(optimizer: Optional[torch.optim.Optimizer], lr: float) -> None:
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_optimizer_lr(optimizer: Optional[torch.optim.Optimizer]) -> Optional[float]:
    if optimizer is None or not optimizer.param_groups:
        return None
    return float(optimizer.param_groups[0]["lr"])


def clip_model_gradients(model: Optional[torch.nn.Module], max_norm: float) -> None:
    if model is None or max_norm <= 0.0:
        return
    params = [param for param in model.parameters() if param.requires_grad and param.grad is not None]
    if not params:
        return
    torch.nn.utils.clip_grad_norm_(params, max_norm)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def compute_probability_margin(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected class probabilities with shape [B, C], got: {tuple(probabilities.shape)}")
    if probabilities.shape[1] <= 1:
        return torch.ones(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype)
    top2 = torch.topk(probabilities, k=2, dim=1).values
    return torch.clamp(top2[:, 0] - top2[:, 1], min=0.0, max=1.0)


def compute_normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected class probabilities with shape [B, C], got: {tuple(probabilities.shape)}")
    num_classes = probabilities.shape[1]
    if num_classes <= 1:
        return torch.zeros(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype)
    entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1e-8, max=1.0))).sum(dim=1)
    return torch.clamp(entropy / math.log(num_classes), min=0.0, max=1.0)


def apply_batch_consistency_augmentation(
    images: torch.Tensor,
    *,
    max_shift: int,
    flip_prob: float,
    noise_std: float,
) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B, C, H, W], got: {tuple(images.shape)}")
    augmented = images.clone()
    batch_size = int(images.shape[0])
    if max_shift > 0:
        shifts_y = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=images.device)
        shifts_x = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=images.device)
        for idx in range(batch_size):
            shift_y = int(shifts_y[idx].item())
            shift_x = int(shifts_x[idx].item())
            if shift_y != 0 or shift_x != 0:
                augmented[idx] = torch.roll(
                    augmented[idx],
                    shifts=(shift_y, shift_x),
                    dims=(-2, -1),
                )
    if flip_prob > 0.0 and augmented.shape[1] > 1:
        flip_mask = torch.rand(batch_size, device=images.device) < flip_prob
        if bool(flip_mask.any().item()):
            augmented[flip_mask] = torch.flip(augmented[flip_mask], dims=(-1,))
    if noise_std > 0.0:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented


def compute_probability_consistency(
    reference_probabilities: torch.Tensor,
    augmented_probabilities: torch.Tensor,
) -> torch.Tensor:
    if reference_probabilities.shape != augmented_probabilities.shape:
        raise ValueError(
            "Probability tensors for consistency must share the same shape, "
            f"got {tuple(reference_probabilities.shape)} and {tuple(augmented_probabilities.shape)}"
        )
    total_variation = 0.5 * torch.abs(reference_probabilities - augmented_probabilities).sum(dim=1)
    return torch.clamp(1.0 - total_variation, min=0.0, max=1.0)


def compute_augmented_consistency_scores(
    model: torch.nn.Module,
    peer_model: torch.nn.Module,
    x: torch.Tensor,
    student_prob_dist: torch.Tensor,
    peer_prob_dist: torch.Tensor,
    *,
    max_shift: int,
    flip_prob: float,
    noise_std: float,
    secondary_model: Optional[torch.nn.Module] = None,
    secondary_prob_dist: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    augmented_x = apply_batch_consistency_augmentation(
        x,
        max_shift=max_shift,
        flip_prob=flip_prob,
        noise_std=noise_std,
    )
    student_was_training = model.training
    peer_was_training = peer_model.training
    secondary_was_training = secondary_model.training if secondary_model is not None else False
    model.eval()
    peer_model.eval()
    if secondary_model is not None:
        secondary_model.eval()
    with torch.no_grad():
        student_aug_prob = F.softmax(model(augmented_x), dim=1)
        peer_aug_prob = F.softmax(peer_model(augmented_x), dim=1)
        secondary_aug_prob = (
            F.softmax(secondary_model(augmented_x), dim=1)
            if secondary_model is not None
            else None
        )
    if student_was_training:
        model.train()
    else:
        model.eval()
    if peer_was_training:
        peer_model.train()
    else:
        peer_model.eval()
    if secondary_model is not None:
        if secondary_was_training:
            secondary_model.train()
        else:
            secondary_model.eval()
    secondary_consistency = None
    if secondary_aug_prob is not None and secondary_prob_dist is not None:
        secondary_consistency = compute_probability_consistency(secondary_prob_dist, secondary_aug_prob)
    return (
        compute_probability_consistency(student_prob_dist, student_aug_prob),
        compute_probability_consistency(peer_prob_dist, peer_aug_prob),
        secondary_consistency,
    )


def build_aug_consistency_reweight(
    student_consistency: torch.Tensor,
    peer_consistency: torch.Tensor,
    *,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight <= 0.0:
        ones = torch.ones_like(student_consistency)
        return ones, ones
    student_signal = torch.clamp(peer_consistency - 0.5 * student_consistency, min=0.0, max=1.0)
    peer_signal = torch.clamp(student_consistency - 0.5 * peer_consistency, min=0.0, max=1.0)
    return 1.0 + weight * student_signal, 1.0 + weight * peer_signal


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


def build_elementwise_kd_loss_fn(temperature: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"distill temperature must be positive, got: {temperature}")

    def _loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return (
            F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits.detach() / temperature, dim=1),
                reduction="none",
            )
            * (temperature * temperature)
        )

    return _loss


def build_topk_element_mask(scores: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    if scores.ndim < 2:
        raise ValueError(f"Expected elementwise scores with shape [B, ...], got: {tuple(scores.shape)}")
    if topk_ratio <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)

    positive = scores > 0
    flat_scores = scores.reshape(scores.shape[0], -1)
    flat_positive = positive.reshape(positive.shape[0], -1)
    element_count = flat_scores.shape[1]
    if element_count == 0:
        return torch.zeros_like(scores, dtype=torch.bool)

    k = max(1, min(element_count, math.ceil(element_count * topk_ratio)))
    if k >= element_count:
        return positive

    masked_scores = flat_scores.masked_fill(~flat_positive, float("-inf"))
    topk_values, topk_indices = torch.topk(masked_scores, k=k, dim=1)
    keep_topk = topk_values > 0
    mask = torch.zeros_like(flat_positive)
    mask.scatter_(1, topk_indices, keep_topk)
    return (mask & flat_positive).reshape_as(scores)


def build_sample_hotspot_weights(
    reference: torch.Tensor,
    hotspot_scores: torch.Tensor,
    hotspot_mask: torch.Tensor,
    alpha: float,
    mode: str = "score",
) -> torch.Tensor:
    weights = torch.ones_like(reference, dtype=reference.dtype)
    if alpha <= 0.0 or hotspot_mask.numel() == 0:
        return weights
    if mode == "binary":
        return weights + alpha * hotspot_mask.to(dtype=reference.dtype)
    if mode != "score":
        raise ValueError(f"Unsupported SSML supervised weight mode: {mode}")
    positive_scores = torch.where(
        hotspot_mask,
        torch.clamp(hotspot_scores, min=0.0),
        torch.zeros_like(hotspot_scores),
    )
    positive_count = hotspot_mask.sum().clamp(min=1).to(dtype=reference.dtype)
    mean_positive = positive_scores.sum() / positive_count
    normalized = torch.where(
        hotspot_mask,
        positive_scores / torch.clamp(mean_positive, min=1e-6),
        torch.zeros_like(positive_scores),
    )
    normalized = torch.clamp(normalized, min=0.0, max=4.0)
    return weights + alpha * normalized.to(dtype=reference.dtype)


def build_sample_score_weights(
    reference: torch.Tensor,
    hotspot_scores: torch.Tensor,
    hotspot_mask: torch.Tensor,
) -> torch.Tensor:
    weights = torch.zeros_like(reference, dtype=reference.dtype)
    if hotspot_mask.numel() == 0:
        return weights
    positive_scores = torch.where(
        hotspot_mask,
        torch.clamp(hotspot_scores, min=0.0),
        torch.zeros_like(hotspot_scores),
    )
    denom = positive_scores.sum()
    return positive_scores.to(dtype=reference.dtype) / torch.clamp(denom.to(dtype=reference.dtype), min=1e-6)


def build_topk_sample_mask(scores: torch.Tensor, topk_ratio: float, scope: str = "total") -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected sample scores with shape [B], got: {tuple(scores.shape)}")
    if topk_ratio <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)
    positive = scores > 0
    sample_count = int(scores.shape[0])
    if sample_count == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    if scope == "positive":
        positive_count = int(positive.sum().item())
        if positive_count == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        k = max(1, min(positive_count, math.ceil(positive_count * topk_ratio)))
        if k >= positive_count:
            return positive
    elif scope == "total":
        k = max(1, min(sample_count, math.ceil(sample_count * topk_ratio)))
        if k >= sample_count:
            return positive
    else:
        raise ValueError(f"Unsupported SSML top-k scope: {scope}")
    if k >= sample_count:
        return positive
    masked_scores = scores.masked_fill(~positive, float("-inf"))
    topk_values, topk_indices = torch.topk(masked_scores, k=k, dim=0)
    keep = topk_values > 0
    mask = torch.zeros_like(positive)
    mask[topk_indices[keep]] = True
    return mask & positive


def build_class_balanced_topk_sample_mask(
    scores: torch.Tensor,
    targets: torch.Tensor,
    topk_ratio: float,
    *,
    scope: str = "total",
    per_class_budget: int = 0,
    per_class_budgets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected sample scores with shape [B], got: {tuple(scores.shape)}")
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if scores.numel() == 0:
        return mask
    for cls in torch.unique(targets):
        class_mask = targets == cls
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
        if class_indices.numel() == 0:
            continue
        class_scores = scores[class_indices]
        positive = class_scores > 0
        if scope == "positive":
            candidate_mask = positive
            candidate_count = int(candidate_mask.sum().item())
        elif scope == "total":
            candidate_mask = torch.ones_like(class_scores, dtype=torch.bool)
            candidate_count = int(class_scores.numel())
        else:
            raise ValueError(f"Unsupported SSML top-k scope: {scope}")
        if candidate_count == 0:
            continue
        class_budget = per_class_budget
        if per_class_budgets is not None:
            cls_idx = int(cls.item())
            if 0 <= cls_idx < int(per_class_budgets.numel()):
                class_budget = max(int(per_class_budgets[cls_idx].item()), 0)
        if class_budget > 0:
            k = min(candidate_count, class_budget)
        else:
            k = max(1, min(candidate_count, math.ceil(candidate_count * topk_ratio)))
        if k <= 0:
            continue
        if scope == "positive" and k >= int(positive.sum().item()):
            local_mask = positive
        elif scope == "total" and k >= int(class_scores.numel()):
            local_mask = positive
        else:
            masked_scores = class_scores.masked_fill(~candidate_mask, float("-inf"))
            topk_values, topk_indices = torch.topk(masked_scores, k=k, dim=0)
            keep = topk_values > 0
            local_mask = torch.zeros_like(class_scores, dtype=torch.bool)
            local_mask[topk_indices[keep]] = True
            local_mask &= positive
        mask[class_indices] = local_mask
    return mask


def mask_ratio(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.to(dtype=torch.float32).mean().item())


def masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or mask.numel() == 0:
        return 0.0
    active = bool(mask.any().item())
    if not active:
        return 0.0
    return float(values[mask].mean().item())


def selected_per_class(mask: torch.Tensor, targets: torch.Tensor) -> float:
    if mask.numel() == 0 or not bool(mask.any().item()):
        return 0.0
    active_targets = targets[mask]
    if active_targets.numel() == 0:
        return 0.0
    class_count = max(int(torch.unique(active_targets).numel()), 1)
    return float(active_targets.numel() / class_count)


def compute_js_divergence_from_probs(
    student_prob: torch.Tensor,
    peer_prob: torch.Tensor,
) -> torch.Tensor:
    student_prob = torch.clamp(student_prob, min=1e-8, max=1.0)
    peer_prob = torch.clamp(peer_prob, min=1e-8, max=1.0)
    mix = torch.clamp((student_prob + peer_prob) * 0.5, min=1e-8, max=1.0)
    return 0.5 * (
        F.kl_div(torch.log(student_prob), mix, reduction="none").sum(dim=1)
        + F.kl_div(torch.log(peer_prob), mix, reduction="none").sum(dim=1)
    )


def compute_recall_by_class(
    correct_by_class: torch.Tensor,
    total_by_class: torch.Tensor,
) -> torch.Tensor:
    recall = torch.zeros_like(correct_by_class, dtype=torch.float32)
    valid = total_by_class > 0
    recall[valid] = correct_by_class[valid].to(dtype=torch.float32) / total_by_class[valid].to(dtype=torch.float32)
    return recall


def build_deficit_adjusted_class_budgets(
    base_budget: int,
    deficit_ema: torch.Tensor,
    extra_budget_scale: float,
) -> torch.Tensor:
    class_count = int(deficit_ema.numel())
    if class_count <= 0:
        return torch.zeros(0, dtype=torch.long)
    if base_budget <= 0:
        return torch.zeros(class_count, dtype=torch.long, device=deficit_ema.device)
    budgets = torch.full((class_count,), int(base_budget), dtype=torch.float32, device=deficit_ema.device)
    if extra_budget_scale <= 0.0:
        return budgets.round().to(dtype=torch.long)
    positive = deficit_ema > 0
    if bool(positive.any().item()):
        mean_positive = deficit_ema[positive].mean()
        normalized = deficit_ema / torch.clamp(mean_positive, min=1e-6)
        scale = 1.0 + extra_budget_scale * (normalized - 1.0)
        scale = torch.clamp(scale, min=0.25)
        budgets = budgets * scale
    return torch.clamp(budgets.round(), min=1.0).to(dtype=torch.long)


def compute_epoch_ramp_scale(
    *,
    epoch: int,
    start_epoch: int,
    end_epoch: int,
) -> float:
    if start_epoch < 0 or end_epoch < 0:
        return 1.0
    start_epoch = max(int(start_epoch), 0)
    end_epoch = max(int(end_epoch), 0)
    if end_epoch <= start_epoch:
        return 0.0 if epoch < start_epoch else 1.0
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
    return float(max(0.0, min(1.0, progress)))


def compute_scheduled_scalar(
    *,
    epoch: int,
    start_value: Optional[float],
    end_value: Optional[float],
    ramp_start_epoch: int,
    ramp_end_epoch: int,
    fallback_value: float,
) -> float:
    if start_value is None and end_value is None:
        return float(fallback_value)
    if start_value is None:
        start_value = float(fallback_value)
    if end_value is None:
        end_value = float(fallback_value)
    ramp_scale = compute_epoch_ramp_scale(
        epoch=epoch,
        start_epoch=ramp_start_epoch,
        end_epoch=ramp_end_epoch,
    )
    return float(start_value) + (float(end_value) - float(start_value)) * float(ramp_scale)


@torch.no_grad()
def evaluate_pair_classification_details(
    model: torch.nn.Module,
    peer_model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float | list[float]]:
    model.eval()
    peer_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_peer_loss = 0.0
    total_peer_acc = 0.0
    total_disagreement = 0.0
    total_count = 0
    correct_by_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    total_by_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    peer_correct_by_class = torch.zeros(num_classes, dtype=torch.long, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        peer_logits = peer_model(x)
        loss = F.cross_entropy(logits, y)
        peer_loss = F.cross_entropy(peer_logits, y)
        batch_size = x.size(0)
        student_prob = F.softmax(logits, dim=1)
        peer_prob = F.softmax(peer_logits, dim=1)
        preds = student_prob.argmax(dim=1)
        peer_preds = peer_prob.argmax(dim=1)
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy(logits, y) * batch_size
        total_peer_loss += float(peer_loss.item()) * batch_size
        total_peer_acc += accuracy(peer_logits, y) * batch_size
        total_disagreement += float(compute_js_divergence_from_probs(student_prob, peer_prob).mean().item()) * batch_size
        total_count += batch_size
        total_by_class += torch.bincount(y, minlength=num_classes)
        correct_by_class += torch.bincount(y[preds.eq(y)], minlength=num_classes)
        peer_correct_by_class += torch.bincount(y[peer_preds.eq(y)], minlength=num_classes)

    student_recall = compute_recall_by_class(correct_by_class, total_by_class).cpu()
    peer_recall = compute_recall_by_class(peer_correct_by_class, total_by_class).cpu()
    denom = max(total_count, 1)
    return {
        "val_loss": total_loss / denom,
        "val_acc": total_acc / denom,
        "peer_val_loss": total_peer_loss / denom,
        "peer_val_acc": total_peer_acc / denom,
        "mean_pair_disagreement": total_disagreement / denom,
        "student_recall_by_class": student_recall.tolist(),
        "peer_recall_by_class": peer_recall.tolist(),
    }


def transform_ssml_score_signal(
    error_signal: torch.Tensor,
    transform_mode: str,
) -> torch.Tensor:
    error_signal = torch.clamp(error_signal, min=0.0)
    if transform_mode == "none":
        return error_signal
    if transform_mode == "sqrt":
        return torch.sqrt(error_signal)
    if transform_mode == "log1p":
        return torch.log1p(error_signal)
    raise ValueError(f"Unsupported SSML score transform: {transform_mode}")


def safe_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return float(torch.quantile(flat, q).item())


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str], label: str) -> Optional[str]:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    print(f"[classification] loaded {label}_checkpoint={path}")
    return str(path)


def snapshot_trainable_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


def compute_anchor_penalty(
    model: torch.nn.Module,
    anchor_params: Optional[list[torch.Tensor]],
) -> torch.Tensor:
    if not anchor_params:
        return next(model.parameters()).new_tensor(0.0)
    current_params = [param for param in model.parameters() if param.requires_grad]
    if not current_params:
        return next(model.parameters()).new_tensor(0.0)
    penalties = [F.mse_loss(param, anchor, reduction="mean") for param, anchor in zip(current_params, anchor_params)]
    return torch.stack(penalties).mean() if penalties else next(model.parameters()).new_tensor(0.0)


def compute_ssml_sample_scores(
    student_error: torch.Tensor,
    peer_error: torch.Tensor,
    *,
    margin: float,
    score_mode: str,
    score_transform: str,
    student_true_prob: Optional[torch.Tensor] = None,
    peer_true_prob: Optional[torch.Tensor] = None,
    student_prob_dist: Optional[torch.Tensor] = None,
    peer_prob_dist: Optional[torch.Tensor] = None,
    student_correct: Optional[torch.Tensor] = None,
    peer_correct: Optional[torch.Tensor] = None,
    prediction_disagreement: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_mode == "relative_gap":
        student_scores = torch.clamp(student_error - peer_error - margin, min=0.0)
        peer_scores = torch.clamp(peer_error - student_error - margin, min=0.0)
        return student_scores, peer_scores

    if score_mode == "peer_better_student_error":
        student_signal = transform_ssml_score_signal(student_error, score_transform)
        peer_signal = transform_ssml_score_signal(peer_error, score_transform)
        student_scores = torch.where(
            (student_error - peer_error) > margin,
            student_signal,
            torch.zeros_like(student_error),
        )
        peer_scores = torch.where(
            (peer_error - student_error) > margin,
            peer_signal,
            torch.zeros_like(peer_error),
        )
        return student_scores, peer_scores

    if score_mode == "peer_better_true_prob_gap":
        if student_true_prob is None or peer_true_prob is None:
            raise ValueError("student_true_prob and peer_true_prob are required for peer_better_true_prob_gap")
        student_signal = transform_ssml_score_signal(1.0 - student_true_prob, score_transform)
        peer_signal = transform_ssml_score_signal(1.0 - peer_true_prob, score_transform)
        student_scores = torch.where(
            (peer_true_prob - student_true_prob) > margin,
            student_signal,
            torch.zeros_like(student_true_prob),
        )
        peer_scores = torch.where(
            (student_true_prob - peer_true_prob) > margin,
            peer_signal,
            torch.zeros_like(peer_true_prob),
        )
        return student_scores, peer_scores

    if score_mode == "peer_better_true_prob_gap_weighted":
        if student_true_prob is None or peer_true_prob is None:
            raise ValueError("student_true_prob and peer_true_prob are required for peer_better_true_prob_gap_weighted")
        student_hardness = transform_ssml_score_signal(1.0 - student_true_prob, score_transform)
        peer_hardness = transform_ssml_score_signal(1.0 - peer_true_prob, score_transform)
        student_advantage = torch.clamp(peer_true_prob - student_true_prob - margin, min=0.0)
        peer_advantage = torch.clamp(student_true_prob - peer_true_prob - margin, min=0.0)
        student_scores = student_hardness * student_advantage
        peer_scores = peer_hardness * peer_advantage
        return student_scores, peer_scores

    if score_mode == "peer_confident_student_uncertain":
        if (
            student_true_prob is None
            or peer_true_prob is None
            or student_prob_dist is None
            or peer_prob_dist is None
        ):
            raise ValueError(
                "student_true_prob, peer_true_prob, student_prob_dist, and peer_prob_dist "
                "are required for peer_confident_student_uncertain"
            )
        student_hardness = transform_ssml_score_signal(1.0 - student_true_prob, score_transform)
        peer_hardness = transform_ssml_score_signal(1.0 - peer_true_prob, score_transform)
        student_uncertainty = compute_normalized_entropy(student_prob_dist)
        peer_uncertainty = compute_normalized_entropy(peer_prob_dist)
        student_clarity = compute_probability_margin(student_prob_dist)
        peer_clarity = compute_probability_margin(peer_prob_dist)
        student_advantage = torch.clamp(peer_true_prob - student_true_prob - margin, min=0.0)
        peer_advantage = torch.clamp(student_true_prob - peer_true_prob - margin, min=0.0)
        student_scores = (
            student_hardness
            * student_advantage
            * (1.0 + student_uncertainty)
            * peer_clarity
            * (1.0 - peer_uncertainty)
        )
        peer_scores = (
            peer_hardness
            * peer_advantage
            * (1.0 + peer_uncertainty)
            * student_clarity
            * (1.0 - student_uncertainty)
        )
        return student_scores, peer_scores

    if score_mode == "useful_hard_sample":
        if (
            student_true_prob is None
            or peer_true_prob is None
            or student_correct is None
            or peer_correct is None
            or prediction_disagreement is None
        ):
            raise ValueError(
                "student_true_prob, peer_true_prob, student_correct, peer_correct, and prediction_disagreement "
                "are required for useful_hard_sample"
            )
        student_hardness = transform_ssml_score_signal(1.0 - student_true_prob, score_transform)
        peer_hardness = transform_ssml_score_signal(1.0 - peer_true_prob, score_transform)
        student_safe = (~student_correct) & peer_correct & prediction_disagreement
        peer_safe = (~peer_correct) & student_correct & prediction_disagreement
        student_advantage = torch.clamp(peer_true_prob - student_true_prob - margin, min=0.0)
        peer_advantage = torch.clamp(student_true_prob - peer_true_prob - margin, min=0.0)
        student_scores = torch.where(
            student_safe,
            student_hardness * student_advantage,
            torch.zeros_like(student_true_prob),
        )
        peer_scores = torch.where(
            peer_safe,
            peer_hardness * peer_advantage,
            torch.zeros_like(peer_true_prob),
        )
        return student_scores, peer_scores

    if score_mode == "useful_hard_sample_confident":
        if (
            student_true_prob is None
            or peer_true_prob is None
            or student_correct is None
            or peer_correct is None
            or prediction_disagreement is None
        ):
            raise ValueError(
                "student_true_prob, peer_true_prob, student_correct, peer_correct, and prediction_disagreement "
                "are required for useful_hard_sample_confident"
            )
        student_hardness = transform_ssml_score_signal(1.0 - student_true_prob, score_transform)
        peer_hardness = transform_ssml_score_signal(1.0 - peer_true_prob, score_transform)
        student_safe = (~student_correct) & peer_correct & prediction_disagreement
        peer_safe = (~peer_correct) & student_correct & prediction_disagreement
        student_advantage = torch.clamp(peer_true_prob - student_true_prob - margin, min=0.0)
        peer_advantage = torch.clamp(student_true_prob - peer_true_prob - margin, min=0.0)
        student_scores = torch.where(
            student_safe,
            student_hardness * student_advantage * peer_true_prob,
            torch.zeros_like(student_true_prob),
        )
        peer_scores = torch.where(
            peer_safe,
            peer_hardness * peer_advantage * student_true_prob,
            torch.zeros_like(peer_true_prob),
        )
        return student_scores, peer_scores

    raise ValueError(f"Unsupported SSML gate score mode: {score_mode}")


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


def compute_ssml_guidance_scale(
    *,
    epoch: int,
    method: str,
    warmup_epochs: int,
    decay_start_epoch: int,
    decay_end_epoch: int,
    decay_min_scale: float,
) -> float:
    if method != "ssml":
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if decay_start_epoch < 0 or decay_end_epoch <= decay_start_epoch:
        return 1.0
    if epoch <= decay_start_epoch:
        return 1.0
    if epoch >= decay_end_epoch:
        return decay_min_scale

    progress = (epoch - decay_start_epoch) / max(decay_end_epoch - decay_start_epoch, 1)
    return 1.0 + (decay_min_scale - 1.0) * progress


def choose_one_way_imitation_from_scores(
    student_scores: torch.Tensor,
    peer_scores: torch.Tensor,
) -> tuple[bool, bool]:
    student_total = torch.clamp(student_scores, min=0.0).sum()
    peer_total = torch.clamp(peer_scores, min=0.0).sum()
    student_has_hotspot = float(student_total.item()) > 0.0
    peer_has_hotspot = float(peer_total.item()) > 0.0
    if student_has_hotspot and peer_has_hotspot:
        return True, True
    if student_has_hotspot:
        return True, False
    if peer_has_hotspot:
        return False, True
    return False, False


def train_one_epoch(
    model,
    peer_model: Optional[torch.nn.Module],
    secondary_peer_model: Optional[torch.nn.Module],
    loader,
    optimizer,
    peer_optimizer: Optional[torch.optim.Optimizer],
    device,
    supervised_loss_fn: Callable[..., torch.Tensor],
    imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ssml_elementwise_imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    label_smoothing: float,
    grad_clip: float,
    ema_model: Optional[torch.nn.Module],
    ema_peer_model: Optional[torch.nn.Module],
    model_ema_decay: float,
    lambda_imitation: float,
    margin: float,
    ssml_topk_ratio: float,
    ssml_supervised_hotspot_alpha: float,
    ssml_topk_scope: str,
    ssml_supervised_weight_mode: str,
    ssml_gate_score_mode: str,
    ssml_score_transform: str,
    ssml_guidance_mode: str,
    ssml_peer_correct_only: bool,
    ssml_student_incorrect_only: bool,
    ssml_disagreement_only: bool,
    ssml_class_balanced_topk: bool,
    ssml_per_class_budget: int,
    num_classes: int,
    ssml_dynamic_per_class_budget: Optional[torch.Tensor],
    ssml_disagreement_floor: float,
    ssml_complement_scale: float,
    ssml_secondary_peer_require_same_label: bool,
    ssml_secondary_peer_agreement_min: float,
    ssml_peer_true_prob_threshold: float,
    ssml_peer_student_prob_gap_min: float,
    ssml_student_true_prob_max: float,
    ssml_aug_consistency_weight: float,
    ssml_aug_consistency_shift: int,
    ssml_aug_consistency_flip_prob: float,
    ssml_aug_consistency_noise_std: float,
    ssml_peer_aug_consistency_min: float,
    ssml_student_aug_consistency_max: float,
    ssml_peer_student_aug_consistency_gap_min: float,
    guidance_scale: float,
    method: str,
    freeze_bn_stats: bool = False,
    hetero_ssml_one_way: bool = False,
    ssml_student_only: bool = False,
    ssml_freeze_peer: bool = False,
    ssml_worse_only_update: bool = False,
    ssml_anchor_weight: float = 0.0,
    anchor_params: Optional[list[torch.Tensor]] = None,
) -> dict[str, float]:
    method = canonicalize_method_name(method)
    peer_update_disabled = ssml_student_only or ssml_freeze_peer
    model.train()
    if freeze_bn_stats:
        apply_batchnorm_eval(model)
    if peer_model is not None:
        if method == "ssml" and peer_update_disabled:
            peer_model.eval()
        else:
            peer_model.train()
            if freeze_bn_stats:
                apply_batchnorm_eval(peer_model)
    dml_weight_builder = get_directional_weight_builder("dml")
    total_loss = 0.0
    total_acc = 0.0
    total_student_positive_ratio = 0.0
    total_peer_positive_ratio = 0.0
    total_student_selected_ratio = 0.0
    total_peer_selected_ratio = 0.0
    total_student_selected_of_positive_ratio = 0.0
    total_peer_selected_of_positive_ratio = 0.0
    total_student_selected_score_mean = 0.0
    total_peer_selected_score_mean = 0.0
    total_student_hotspot_error_mean = 0.0
    total_student_background_error_mean = 0.0
    total_peer_hotspot_error_mean = 0.0
    total_peer_background_error_mean = 0.0
    total_student_hotspot_gap_mean = 0.0
    total_peer_hotspot_gap_mean = 0.0
    total_student_hotspot_error_share = 0.0
    total_peer_hotspot_error_share = 0.0
    total_student_incorrect_ratio = 0.0
    total_peer_incorrect_ratio = 0.0
    total_prediction_disagreement_ratio = 0.0
    total_student_teacher_usable_ratio = 0.0
    total_peer_teacher_usable_ratio = 0.0
    total_student_teacher_safe_ratio = 0.0
    total_peer_teacher_safe_ratio = 0.0
    total_student_useful_hard_ratio = 0.0
    total_peer_useful_hard_ratio = 0.0
    total_student_score_p90 = 0.0
    total_peer_score_p90 = 0.0
    total_student_worse_ratio = 0.0
    total_peer_worse_ratio = 0.0
    total_student_worse_update_ratio = 0.0
    total_peer_worse_update_ratio = 0.0
    total_student_update_ratio = 0.0
    total_peer_update_ratio = 0.0
    total_student_selected_per_class = 0.0
    total_peer_selected_per_class = 0.0
    total_student_aug_consistency_mean = 0.0
    total_peer_aug_consistency_mean = 0.0
    total_secondary_peer_agreement_ratio = 0.0
    total_secondary_peer_consensus_ratio = 0.0
    total_secondary_peer_aug_consistency_mean = 0.0
    total_anchor_loss = 0.0
    total_preserved_disagreement = 0.0
    total_disagreement_floor_gap = 0.0
    total_safe_teacher_miss_rate = torch.zeros(num_classes, dtype=torch.float32, device=device)
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if peer_optimizer is not None:
            peer_optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        hard_supervised_loss = supervised_loss_fn(logits, y)
        train_supervised_loss = (
            hard_supervised_loss
            if label_smoothing <= 0.0
            else F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
        )
        loss = train_supervised_loss.mean()
        student_positive_ratio = 0.0
        peer_positive_ratio = 0.0
        student_selected_ratio = 0.0
        peer_selected_ratio = 0.0
        student_selected_of_positive_ratio = 0.0
        peer_selected_of_positive_ratio = 0.0
        student_selected_score_mean = 0.0
        peer_selected_score_mean = 0.0
        student_hotspot_error_mean = 0.0
        student_background_error_mean = 0.0
        peer_hotspot_error_mean = 0.0
        peer_background_error_mean = 0.0
        student_hotspot_gap_mean = 0.0
        peer_hotspot_gap_mean = 0.0
        student_hotspot_error_share = 0.0
        peer_hotspot_error_share = 0.0
        student_incorrect_ratio = 0.0
        peer_incorrect_ratio = 0.0
        prediction_disagreement_ratio = 0.0
        student_teacher_usable_ratio = 0.0
        peer_teacher_usable_ratio = 0.0
        student_teacher_safe_ratio = 0.0
        peer_teacher_safe_ratio = 0.0
        student_useful_hard_ratio = 0.0
        peer_useful_hard_ratio = 0.0
        student_score_p90 = 0.0
        peer_score_p90 = 0.0
        student_worse_ratio = 0.0
        peer_worse_ratio = 0.0
        student_worse_update_ratio = 0.0
        peer_worse_update_ratio = 0.0
        student_update_ratio = 0.0
        peer_update_ratio = 0.0
        student_selected_per_class = 0.0
        peer_selected_per_class = 0.0
        student_aug_consistency_mean = 0.0
        peer_aug_consistency_mean = 0.0
        secondary_peer_agreement_ratio = 0.0
        secondary_peer_consensus_ratio = 0.0
        secondary_peer_aug_consistency_mean = 0.0
        anchor_loss_metric = 0.0
        preserved_disagreement_mean = 0.0
        disagreement_floor_gap_mean = 0.0
        safe_teacher_miss_rate_by_class = torch.zeros(num_classes, dtype=torch.float32, device=device)

        if method == "independent":
            loss.backward()
            clip_model_gradients(model, grad_clip)
            optimizer.step()
            update_ema_model(ema_model, model, model_ema_decay)

        elif method == "dml":
            if peer_model is None or peer_optimizer is None:
                raise ValueError("peer_model and peer_optimizer are required when method='dml'")
            peer_logits = peer_model(x)
            peer_hard_supervised_loss = supervised_loss_fn(peer_logits, y)
            peer_train_supervised_loss = (
                peer_hard_supervised_loss
                if label_smoothing <= 0.0
                else F.cross_entropy(peer_logits, y, reduction="none", label_smoothing=label_smoothing)
            )
            w_student, w_peer = dml_weight_builder(
                hard_supervised_loss.detach(),
                peer_hard_supervised_loss.detach(),
                margin=margin,
            )
            if lambda_imitation <= 0.0:
                w_student = torch.zeros_like(w_student)
                w_peer = torch.zeros_like(w_peer)

            imitation_term_student = weighted_mean(imitation_loss_fn(logits, peer_logits.detach()), w_student)
            loss = train_supervised_loss.mean() + lambda_imitation * imitation_term_student

            imitation_term_peer = weighted_mean(imitation_loss_fn(peer_logits, logits.detach()), w_peer)
            peer_loss = peer_train_supervised_loss.mean() + lambda_imitation * imitation_term_peer

            (loss + peer_loss).backward()
            clip_model_gradients(model, grad_clip)
            clip_model_gradients(peer_model, grad_clip)
            optimizer.step()
            peer_optimizer.step()
            update_ema_model(ema_model, model, model_ema_decay)
            update_ema_model(ema_peer_model, peer_model, model_ema_decay)

        elif method == "ssml":
            if peer_model is None:
                raise ValueError("peer_model is required when method='ssml'")
            if not peer_update_disabled and peer_optimizer is None:
                raise ValueError("peer_optimizer is required when method='ssml' unless peer updates are disabled")
            if peer_update_disabled:
                with torch.no_grad():
                    peer_logits = peer_model(x)
            else:
                peer_logits = peer_model(x)
            secondary_peer_logits = None
            if secondary_peer_model is not None:
                with torch.no_grad():
                    secondary_peer_logits = secondary_peer_model(x)
            peer_hard_supervised_loss = supervised_loss_fn(peer_logits, y)
            peer_train_supervised_loss = (
                peer_hard_supervised_loss
                if label_smoothing <= 0.0
                else F.cross_entropy(peer_logits, y, reduction="none", label_smoothing=label_smoothing)
            )
            zero = logits.new_tensor(0.0)
            student_sample_loss = hard_supervised_loss.detach()
            peer_sample_loss = peer_hard_supervised_loss.detach()
            student_gap = student_sample_loss - peer_sample_loss
            peer_gap = peer_sample_loss - student_sample_loss
            student_probs = F.softmax(logits.detach(), dim=1)
            peer_probs = F.softmax(peer_logits.detach(), dim=1)
            secondary_peer_probs = (
                F.softmax(secondary_peer_logits.detach(), dim=1)
                if secondary_peer_logits is not None
                else None
            )
            student_pred = student_probs.argmax(dim=1)
            peer_pred = peer_probs.argmax(dim=1)
            secondary_peer_pred = (
                secondary_peer_probs.argmax(dim=1)
                if secondary_peer_probs is not None
                else None
            )
            student_correct = student_pred.eq(y)
            peer_correct = peer_pred.eq(y)
            secondary_peer_correct = (
                secondary_peer_pred.eq(y)
                if secondary_peer_pred is not None
                else None
            )
            prediction_disagreement = student_pred.ne(peer_pred)
            student_true_prob = student_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            peer_true_prob = peer_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            secondary_peer_true_prob = (
                secondary_peer_probs.gather(1, y.unsqueeze(1)).squeeze(1)
                if secondary_peer_probs is not None
                else None
            )
            aug_consistency_enabled = (
                ssml_aug_consistency_weight > 0.0
                and (
                    ssml_aug_consistency_shift > 0
                    or ssml_aug_consistency_flip_prob > 0.0
                    or ssml_aug_consistency_noise_std > 0.0
                )
            )
            secondary_consensus_mask = torch.ones_like(student_correct, dtype=torch.bool)
            if secondary_peer_probs is not None:
                peer_top1_prob = peer_probs.max(dim=1).values
                secondary_peer_top1_prob = secondary_peer_probs.max(dim=1).values
                peer_secondary_agreement = peer_pred.eq(secondary_peer_pred)
                secondary_peer_agreement_ratio = float(peer_secondary_agreement.float().mean().item())
                if ssml_secondary_peer_require_same_label:
                    secondary_consensus_mask &= peer_secondary_agreement
                if ssml_secondary_peer_agreement_min > 0.0:
                    secondary_consensus_mask &= (
                        torch.minimum(peer_top1_prob, secondary_peer_top1_prob)
                        >= ssml_secondary_peer_agreement_min
                    )
                if ssml_peer_true_prob_threshold > 0.0 and secondary_peer_true_prob is not None:
                    secondary_consensus_mask &= secondary_peer_true_prob >= ssml_peer_true_prob_threshold
                if (
                    ssml_peer_student_prob_gap_min > 0.0
                    and secondary_peer_true_prob is not None
                ):
                    secondary_consensus_mask &= (
                        secondary_peer_true_prob - student_true_prob
                    ) >= ssml_peer_student_prob_gap_min
            student_scores, peer_scores = compute_ssml_sample_scores(
                student_sample_loss,
                peer_sample_loss,
                margin=margin,
                score_mode=ssml_gate_score_mode,
                score_transform=ssml_score_transform,
                student_true_prob=student_true_prob,
                peer_true_prob=peer_true_prob,
                student_prob_dist=student_probs,
                peer_prob_dist=peer_probs,
                student_correct=student_correct,
                peer_correct=peer_correct,
                prediction_disagreement=prediction_disagreement,
            )
            if aug_consistency_enabled:
                (
                    student_aug_consistency,
                    peer_aug_consistency,
                    secondary_peer_aug_consistency,
                ) = compute_augmented_consistency_scores(
                    model,
                    peer_model,
                    x,
                    student_probs,
                    peer_probs,
                    max_shift=ssml_aug_consistency_shift,
                    flip_prob=ssml_aug_consistency_flip_prob,
                    noise_std=ssml_aug_consistency_noise_std,
                    secondary_model=secondary_peer_model,
                    secondary_prob_dist=secondary_peer_probs,
                )
                if freeze_bn_stats:
                    apply_batchnorm_eval(model)
                    apply_batchnorm_eval(peer_model)
                    if secondary_peer_model is not None:
                        apply_batchnorm_eval(secondary_peer_model)
                student_consistency_factor, peer_consistency_factor = build_aug_consistency_reweight(
                    student_aug_consistency,
                    peer_aug_consistency,
                    weight=ssml_aug_consistency_weight,
                )
                student_scores = student_scores * student_consistency_factor
                peer_scores = peer_scores * peer_consistency_factor
                student_aug_consistency_mean = float(student_aug_consistency.mean().item())
                peer_aug_consistency_mean = float(peer_aug_consistency.mean().item())
                if secondary_peer_aug_consistency is not None:
                    secondary_peer_aug_consistency_mean = float(secondary_peer_aug_consistency.mean().item())
                if ssml_peer_aug_consistency_min > 0.0:
                    student_scores = student_scores * (
                        peer_aug_consistency >= ssml_peer_aug_consistency_min
                    ).to(dtype=student_scores.dtype)
                    peer_scores = peer_scores * (
                        student_aug_consistency >= ssml_peer_aug_consistency_min
                    ).to(dtype=peer_scores.dtype)
                    if secondary_peer_aug_consistency is not None:
                        secondary_consensus_mask &= (
                            secondary_peer_aug_consistency >= ssml_peer_aug_consistency_min
                        )
                if ssml_student_aug_consistency_max < 1.0:
                    student_scores = student_scores * (
                        student_aug_consistency <= ssml_student_aug_consistency_max
                    ).to(dtype=student_scores.dtype)
                    peer_scores = peer_scores * (
                        peer_aug_consistency <= ssml_student_aug_consistency_max
                    ).to(dtype=peer_scores.dtype)
                if ssml_peer_student_aug_consistency_gap_min > 0.0:
                    student_scores = student_scores * (
                        (peer_aug_consistency - student_aug_consistency)
                        >= ssml_peer_student_aug_consistency_gap_min
                    ).to(dtype=student_scores.dtype)
                    peer_scores = peer_scores * (
                        (student_aug_consistency - peer_aug_consistency)
                        >= ssml_peer_student_aug_consistency_gap_min
                    ).to(dtype=peer_scores.dtype)
                    if secondary_peer_aug_consistency is not None:
                        secondary_consensus_mask &= (
                            (secondary_peer_aug_consistency - student_aug_consistency)
                            >= ssml_peer_student_aug_consistency_gap_min
                        )
            if secondary_peer_probs is not None:
                student_scores = student_scores * secondary_consensus_mask.to(dtype=student_scores.dtype)
                peer_scores = peer_scores * secondary_consensus_mask.to(dtype=peer_scores.dtype)
                secondary_peer_consensus_ratio = float(secondary_consensus_mask.float().mean().item())
            student_useful_hard = (~student_correct) & peer_correct & prediction_disagreement
            peer_useful_hard = (~peer_correct) & student_correct & prediction_disagreement
            student_teacher_usable = peer_correct.clone()
            peer_teacher_usable = student_correct.clone()
            if ssml_peer_true_prob_threshold > 0.0:
                student_teacher_usable &= peer_true_prob >= ssml_peer_true_prob_threshold
                peer_teacher_usable &= student_true_prob >= ssml_peer_true_prob_threshold
            student_teacher_safe = student_teacher_usable.clone()
            peer_teacher_safe = peer_teacher_usable.clone()
            if ssml_peer_student_prob_gap_min > 0.0:
                student_teacher_safe &= (peer_true_prob - student_true_prob) >= ssml_peer_student_prob_gap_min
                peer_teacher_safe &= (student_true_prob - peer_true_prob) >= ssml_peer_student_prob_gap_min
                student_scores = student_scores * student_teacher_safe.to(dtype=student_scores.dtype)
                peer_scores = peer_scores * peer_teacher_safe.to(dtype=peer_scores.dtype)
            if ssml_peer_correct_only:
                student_scores = student_scores * student_teacher_usable.to(dtype=student_scores.dtype)
                peer_scores = peer_scores * peer_teacher_usable.to(dtype=peer_scores.dtype)
            if ssml_student_incorrect_only:
                student_scores = student_scores * (~student_correct).to(dtype=student_scores.dtype)
                peer_scores = peer_scores * (~peer_correct).to(dtype=peer_scores.dtype)
            if ssml_student_true_prob_max < 1.0:
                student_scores = student_scores * (student_true_prob <= ssml_student_true_prob_max).to(dtype=student_scores.dtype)
                peer_scores = peer_scores * (peer_true_prob <= ssml_student_true_prob_max).to(dtype=peer_scores.dtype)
            if ssml_disagreement_only:
                disagreement_weight = prediction_disagreement.to(dtype=student_scores.dtype)
                student_scores = student_scores * disagreement_weight
                peer_scores = peer_scores * disagreement_weight
            worse_student_mask = student_sample_loss > peer_sample_loss
            worse_peer_mask = peer_sample_loss > student_sample_loss
            if ssml_worse_only_update:
                student_scores = student_scores * worse_student_mask.to(dtype=student_scores.dtype)
                peer_scores = peer_scores * worse_peer_mask.to(dtype=peer_scores.dtype)

            if ssml_class_balanced_topk:
                mask_student = build_class_balanced_topk_sample_mask(
                    student_scores,
                    y,
                    ssml_topk_ratio,
                    scope=ssml_topk_scope,
                    per_class_budget=ssml_per_class_budget,
                    per_class_budgets=ssml_dynamic_per_class_budget,
                )
                mask_peer = build_class_balanced_topk_sample_mask(
                    peer_scores,
                    y,
                    ssml_topk_ratio,
                    scope=ssml_topk_scope,
                    per_class_budget=ssml_per_class_budget,
                    per_class_budgets=ssml_dynamic_per_class_budget,
                )
            else:
                mask_student = build_topk_sample_mask(student_scores, ssml_topk_ratio, scope=ssml_topk_scope)
                mask_peer = build_topk_sample_mask(peer_scores, ssml_topk_ratio, scope=ssml_topk_scope)
            if guidance_scale <= 0.0:
                mask_student = torch.zeros_like(mask_student, dtype=torch.bool)
                mask_peer = torch.zeros_like(mask_peer, dtype=torch.bool)
            elif lambda_imitation <= 0.0 and ssml_guidance_mode != "reweight_only":
                mask_student = torch.zeros_like(mask_student, dtype=torch.bool)
                mask_peer = torch.zeros_like(mask_peer, dtype=torch.bool)
            elif hetero_ssml_one_way and ssml_guidance_mode != "reweight_only":
                student_imitates, peer_imitates = choose_one_way_imitation_from_scores(
                    student_scores,
                    peer_scores,
                )
                if not student_imitates:
                    mask_student = torch.zeros_like(mask_student, dtype=torch.bool)
                if not peer_imitates:
                    mask_peer = torch.zeros_like(mask_peer, dtype=torch.bool)
            if peer_update_disabled:
                mask_peer = torch.zeros_like(mask_peer, dtype=torch.bool)

            imit_student = imitation_loss_fn(logits, peer_logits.detach())
            hotspot_weight_student = build_sample_hotspot_weights(
                train_supervised_loss,
                student_scores,
                mask_student,
                ssml_supervised_hotspot_alpha * guidance_scale,
                mode=ssml_supervised_weight_mode,
            )
            supervised_term_student = weighted_mean(train_supervised_loss, hotspot_weight_student)
            if ssml_guidance_mode == "reweight_only":
                imitation_term_student = zero
                loss = supervised_term_student
            else:
                imitation_weight_student = build_sample_score_weights(
                    imit_student,
                    student_scores,
                    mask_student,
                )
                imitation_term_student = weighted_mean(imit_student, imitation_weight_student)
                loss = supervised_term_student + lambda_imitation * imitation_term_student

            non_transfer_mask = ~(mask_student | mask_peer)
            if ssml_disagreement_floor > 0.0 and bool(non_transfer_mask.any().item()):
                live_student_prob = F.softmax(logits, dim=1)
                live_peer_prob = (
                    F.softmax(peer_logits.detach(), dim=1)
                    if peer_update_disabled
                    else F.softmax(peer_logits, dim=1)
                )
                disagreement_values = compute_js_divergence_from_probs(live_student_prob, live_peer_prob)
                disagreement_gap = torch.clamp(ssml_disagreement_floor - disagreement_values, min=0.0)
                preserve_loss = disagreement_gap[non_transfer_mask].mean()
                loss = loss + lambda_imitation * preserve_loss
                preserved_disagreement_mean = masked_tensor_mean(disagreement_values, non_transfer_mask)
                disagreement_floor_gap_mean = masked_tensor_mean(disagreement_gap, non_transfer_mask)

            hotspot_weight_peer = build_sample_hotspot_weights(
                peer_train_supervised_loss,
                peer_scores,
                mask_peer,
                ssml_supervised_hotspot_alpha * guidance_scale,
                mode=ssml_supervised_weight_mode,
            )
            supervised_term_peer = weighted_mean(peer_train_supervised_loss, hotspot_weight_peer)
            if peer_update_disabled:
                imitation_term_peer = zero
                peer_loss = zero
            elif ssml_guidance_mode == "reweight_only":
                imitation_term_peer = zero
                peer_loss = supervised_term_peer
            else:
                imit_peer = imitation_loss_fn(peer_logits, logits.detach())
                imitation_weight_peer = build_sample_score_weights(
                    imit_peer,
                    peer_scores,
                    mask_peer,
                )
                imitation_term_peer = weighted_mean(imit_peer, imitation_weight_peer)
                peer_loss = supervised_term_peer + lambda_imitation * imitation_term_peer
            if ssml_anchor_weight > 0.0 and anchor_params:
                anchor_penalty = compute_anchor_penalty(model, anchor_params)
                anchor_loss_metric = float(anchor_penalty.item())
                loss = loss + ssml_anchor_weight * anchor_penalty

            if peer_update_disabled:
                loss.backward()
                clip_model_gradients(model, grad_clip)
                optimizer.step()
                update_ema_model(ema_model, model, model_ema_decay)
            else:
                (loss + peer_loss).backward()
                clip_model_gradients(model, grad_clip)
                clip_model_gradients(peer_model, grad_clip)
                optimizer.step()
                peer_optimizer.step()
                update_ema_model(ema_model, model, model_ema_decay)
                update_ema_model(ema_peer_model, peer_model, model_ema_decay)

            student_positive_ratio = mask_ratio(student_scores > 0)
            peer_positive_ratio = mask_ratio(peer_scores > 0)
            student_selected_ratio = mask_ratio(mask_student)
            peer_selected_ratio = mask_ratio(mask_peer)
            if student_positive_ratio > 0.0:
                student_selected_of_positive_ratio = student_selected_ratio / student_positive_ratio
            if peer_positive_ratio > 0.0:
                peer_selected_of_positive_ratio = peer_selected_ratio / peer_positive_ratio
            student_selected_score_mean = masked_tensor_mean(student_scores, mask_student)
            peer_selected_score_mean = masked_tensor_mean(peer_scores, mask_peer)
            student_hotspot_error_mean = masked_tensor_mean(student_sample_loss, mask_student)
            student_background_error_mean = masked_tensor_mean(student_sample_loss, ~mask_student)
            peer_hotspot_error_mean = masked_tensor_mean(peer_sample_loss, mask_peer)
            peer_background_error_mean = masked_tensor_mean(peer_sample_loss, ~mask_peer)
            student_hotspot_gap_mean = masked_tensor_mean(student_gap, mask_student)
            peer_hotspot_gap_mean = masked_tensor_mean(peer_gap, mask_peer)
            student_incorrect_ratio = float((~student_correct).float().mean().item())
            peer_incorrect_ratio = float((~peer_correct).float().mean().item())
            prediction_disagreement_ratio = float(prediction_disagreement.float().mean().item())
            student_teacher_usable_ratio = float(student_teacher_usable.float().mean().item())
            peer_teacher_usable_ratio = float(peer_teacher_usable.float().mean().item())
            student_teacher_safe_ratio = float(student_teacher_safe.float().mean().item())
            peer_teacher_safe_ratio = float(peer_teacher_safe.float().mean().item())
            student_useful_hard_ratio = float(student_useful_hard.float().mean().item())
            peer_useful_hard_ratio = float(peer_useful_hard.float().mean().item())
            class_totals = torch.bincount(y, minlength=num_classes).to(dtype=torch.float32)
            safe_teacher_miss_counts = torch.bincount(y[student_useful_hard], minlength=num_classes).to(dtype=torch.float32)
            valid_class_mask = class_totals > 0
            safe_teacher_miss_rate_by_class[valid_class_mask] = (
                safe_teacher_miss_counts[valid_class_mask] / class_totals[valid_class_mask]
            )
            student_score_p90 = safe_quantile(student_scores, 0.9)
            peer_score_p90 = safe_quantile(peer_scores, 0.9)
            student_worse_ratio = float(worse_student_mask.float().mean().item())
            peer_worse_ratio = float(worse_peer_mask.float().mean().item())
            student_worse_update_ratio = masked_tensor_mean(
                worse_student_mask.to(dtype=student_sample_loss.dtype),
                mask_student,
            )
            peer_worse_update_ratio = masked_tensor_mean(
                worse_peer_mask.to(dtype=peer_sample_loss.dtype),
                mask_peer,
            )
            student_update_ratio = mask_ratio(mask_student)
            peer_update_ratio = mask_ratio(mask_peer)
            student_selected_per_class = selected_per_class(mask_student, y)
            peer_selected_per_class = selected_per_class(mask_peer, y)
            student_total_error = float(student_sample_loss.sum().item())
            peer_total_error = float(peer_sample_loss.sum().item())
            if student_total_error > 0.0 and bool(mask_student.any().item()):
                student_hotspot_error_share = float((student_sample_loss[mask_student].sum() / student_sample_loss.sum()).item())
            if peer_total_error > 0.0 and bool(mask_peer.any().item()):
                peer_hotspot_error_share = float((peer_sample_loss[mask_peer].sum() / peer_sample_loss.sum()).item())

        else:
            raise ValueError(f"Unsupported method '{method}'")

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy(logits, y) * batch_size
        total_student_positive_ratio += student_positive_ratio * batch_size
        total_peer_positive_ratio += peer_positive_ratio * batch_size
        total_student_selected_ratio += student_selected_ratio * batch_size
        total_peer_selected_ratio += peer_selected_ratio * batch_size
        total_student_selected_of_positive_ratio += student_selected_of_positive_ratio * batch_size
        total_peer_selected_of_positive_ratio += peer_selected_of_positive_ratio * batch_size
        total_student_selected_score_mean += student_selected_score_mean * batch_size
        total_peer_selected_score_mean += peer_selected_score_mean * batch_size
        total_student_hotspot_error_mean += student_hotspot_error_mean * batch_size
        total_student_background_error_mean += student_background_error_mean * batch_size
        total_peer_hotspot_error_mean += peer_hotspot_error_mean * batch_size
        total_peer_background_error_mean += peer_background_error_mean * batch_size
        total_student_hotspot_gap_mean += student_hotspot_gap_mean * batch_size
        total_peer_hotspot_gap_mean += peer_hotspot_gap_mean * batch_size
        total_student_hotspot_error_share += student_hotspot_error_share * batch_size
        total_peer_hotspot_error_share += peer_hotspot_error_share * batch_size
        total_student_incorrect_ratio += student_incorrect_ratio * batch_size
        total_peer_incorrect_ratio += peer_incorrect_ratio * batch_size
        total_prediction_disagreement_ratio += prediction_disagreement_ratio * batch_size
        total_student_teacher_usable_ratio += student_teacher_usable_ratio * batch_size
        total_peer_teacher_usable_ratio += peer_teacher_usable_ratio * batch_size
        total_student_teacher_safe_ratio += student_teacher_safe_ratio * batch_size
        total_peer_teacher_safe_ratio += peer_teacher_safe_ratio * batch_size
        total_student_useful_hard_ratio += student_useful_hard_ratio * batch_size
        total_peer_useful_hard_ratio += peer_useful_hard_ratio * batch_size
        total_student_score_p90 += student_score_p90 * batch_size
        total_peer_score_p90 += peer_score_p90 * batch_size
        total_student_worse_ratio += student_worse_ratio * batch_size
        total_peer_worse_ratio += peer_worse_ratio * batch_size
        total_student_worse_update_ratio += student_worse_update_ratio * batch_size
        total_peer_worse_update_ratio += peer_worse_update_ratio * batch_size
        total_student_update_ratio += student_update_ratio * batch_size
        total_peer_update_ratio += peer_update_ratio * batch_size
        total_student_selected_per_class += student_selected_per_class * batch_size
        total_peer_selected_per_class += peer_selected_per_class * batch_size
        total_student_aug_consistency_mean += student_aug_consistency_mean * batch_size
        total_peer_aug_consistency_mean += peer_aug_consistency_mean * batch_size
        total_secondary_peer_agreement_ratio += secondary_peer_agreement_ratio * batch_size
        total_secondary_peer_consensus_ratio += secondary_peer_consensus_ratio * batch_size
        total_secondary_peer_aug_consistency_mean += secondary_peer_aug_consistency_mean * batch_size
        total_anchor_loss += anchor_loss_metric * batch_size
        total_preserved_disagreement += preserved_disagreement_mean * batch_size
        total_disagreement_floor_gap += disagreement_floor_gap_mean * batch_size
        total_safe_teacher_miss_rate += safe_teacher_miss_rate_by_class * batch_size
        total_count += batch_size
    return {
        "train_loss": total_loss / total_count,
        "train_acc": total_acc / total_count,
        "student_positive_score_ratio": total_student_positive_ratio / total_count,
        "peer_positive_score_ratio": total_peer_positive_ratio / total_count,
        "student_selected_ratio": total_student_selected_ratio / total_count,
        "peer_selected_ratio": total_peer_selected_ratio / total_count,
        "student_selected_of_positive_ratio": total_student_selected_of_positive_ratio / total_count,
        "peer_selected_of_positive_ratio": total_peer_selected_of_positive_ratio / total_count,
        "student_selected_score_mean": total_student_selected_score_mean / total_count,
        "peer_selected_score_mean": total_peer_selected_score_mean / total_count,
        "student_hotspot_error_mean": total_student_hotspot_error_mean / total_count,
        "student_background_error_mean": total_student_background_error_mean / total_count,
        "peer_hotspot_error_mean": total_peer_hotspot_error_mean / total_count,
        "peer_background_error_mean": total_peer_background_error_mean / total_count,
        "student_hotspot_gap_mean": total_student_hotspot_gap_mean / total_count,
        "peer_hotspot_gap_mean": total_peer_hotspot_gap_mean / total_count,
        "student_hotspot_error_share": total_student_hotspot_error_share / total_count,
        "peer_hotspot_error_share": total_peer_hotspot_error_share / total_count,
        "student_incorrect_ratio": total_student_incorrect_ratio / total_count,
        "peer_incorrect_ratio": total_peer_incorrect_ratio / total_count,
        "prediction_disagreement_ratio": total_prediction_disagreement_ratio / total_count,
        "student_teacher_usable_ratio": total_student_teacher_usable_ratio / total_count,
        "peer_teacher_usable_ratio": total_peer_teacher_usable_ratio / total_count,
        "student_teacher_safe_ratio": total_student_teacher_safe_ratio / total_count,
        "peer_teacher_safe_ratio": total_peer_teacher_safe_ratio / total_count,
        "student_useful_hard_ratio": total_student_useful_hard_ratio / total_count,
        "peer_useful_hard_ratio": total_peer_useful_hard_ratio / total_count,
        "student_score_p90": total_student_score_p90 / total_count,
        "peer_score_p90": total_peer_score_p90 / total_count,
        "student_worse_ratio": total_student_worse_ratio / total_count,
        "peer_worse_ratio": total_peer_worse_ratio / total_count,
        "student_worse_update_ratio": total_student_worse_update_ratio / total_count,
        "peer_worse_update_ratio": total_peer_worse_update_ratio / total_count,
        "student_update_ratio": total_student_update_ratio / total_count,
        "peer_update_ratio": total_peer_update_ratio / total_count,
        "student_selected_per_class": total_student_selected_per_class / total_count,
        "peer_selected_per_class": total_peer_selected_per_class / total_count,
        "student_aug_consistency_mean": total_student_aug_consistency_mean / total_count,
        "peer_aug_consistency_mean": total_peer_aug_consistency_mean / total_count,
        "secondary_peer_agreement_ratio": total_secondary_peer_agreement_ratio / total_count,
        "secondary_peer_consensus_ratio": total_secondary_peer_consensus_ratio / total_count,
        "secondary_peer_aug_consistency_mean": total_secondary_peer_aug_consistency_mean / total_count,
        "anchor_loss_mean": total_anchor_loss / total_count,
        "preserved_disagreement_mean": total_preserved_disagreement / total_count,
        "disagreement_floor_gap_mean": total_disagreement_floor_gap / total_count,
        "student_safe_teacher_miss_rate_by_class": (total_safe_teacher_miss_rate / total_count).cpu().tolist(),
    }


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
    args.method = canonicalize_method_name(args.method)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_runtime_environment(args, device)

    data = build_classification_dataloaders(
        ClassificationDataConfig(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
            train_aug_mode=args.train_aug_mode,
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

    peer_model_name = (args.peer_model or args.model) if uses_peer_model(args.method) else None
    pair_meta = build_pair_metadata(args.model, peer_model_name)
    model = build_classification_model(
        model_name=args.model,
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size,
    ).to(device)
    peer_model = None
    secondary_peer_model = None
    peer_optimizer = None
    if uses_peer_model(args.method):
        peer_model = build_classification_model(
            model_name=pair_meta["peer_model"],
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
        ).to(device)
        if args.method == "ssml" and args.ssml_secondary_peer_init_checkpoint:
            secondary_peer_model = build_classification_model(
                model_name=pair_meta["peer_model"],
                num_classes=num_classes,
                in_channels=in_channels,
                image_size=image_size,
            ).to(device)
    loaded_init_checkpoint = load_model_checkpoint(model, args.init_checkpoint, "init")
    loaded_peer_init_checkpoint = None
    loaded_secondary_peer_init_checkpoint = None
    if peer_model is not None:
        loaded_peer_init_checkpoint = load_model_checkpoint(peer_model, args.peer_init_checkpoint, "peer_init")
    if secondary_peer_model is not None:
        loaded_secondary_peer_init_checkpoint = load_model_checkpoint(
            secondary_peer_model,
            args.ssml_secondary_peer_init_checkpoint,
            "secondary_peer_init",
        )
    ssml_student_only = args.method == "ssml" and args.ssml_student_only and uses_peer_model(args.method)
    ssml_freeze_peer = args.method == "ssml" and args.ssml_freeze_peer and uses_peer_model(args.method)
    peer_update_disabled = ssml_student_only or ssml_freeze_peer
    if peer_model is not None and peer_update_disabled:
        for param in peer_model.parameters():
            param.requires_grad_(False)
        peer_model.eval()
    if secondary_peer_model is not None:
        for param in secondary_peer_model.parameters():
            param.requires_grad_(False)
        secondary_peer_model.eval()
    anchor_params = None
    if args.method == "ssml" and args.ssml_anchor_weight > 0.0:
        anchor_params = snapshot_trainable_parameters(model)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd_nesterov":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    if peer_model is not None and not peer_update_disabled:
        if args.optimizer == "adamw":
            peer_optimizer = torch.optim.AdamW(peer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            peer_optimizer = torch.optim.SGD(
                peer_model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True,
            )
    ema_model = clone_ema_model(model) if args.model_ema_decay > 0.0 else None
    ema_peer_model = (
        clone_ema_model(peer_model)
        if args.model_ema_decay > 0.0 and peer_model is not None and not peer_update_disabled
        else None
    )

    run_dir = make_run_dir(
        args.output_dir,
        "classification",
        args.dataset,
        f"{pair_meta['pair_tag']}_{args.method}_{args.classification_imitation_loss}_seed{args.seed}",
    )
    print(f"[classification] run_dir={run_dir}")
    print(f"[classification] params={count_parameters(model)}")

    epoch_log_path = Path(run_dir) / "epoch_metrics.jsonl"
    if epoch_log_path.exists():
        epoch_log_path.unlink()

    train_loss_curve = []
    train_acc_curve = []
    val_loss_curve = []
    val_acc_curve = []
    peer_val_loss_curve = []
    peer_val_acc_curve = []
    best_val_acc = 0.0
    best_peer_val_acc = 0.0
    best_epoch = None
    best_peer_epoch = None
    first_active_epoch = None
    best_val_acc_before_activation = 0.0
    best_val_acc_before_activation_epoch = None
    best_val_acc_after_activation = 0.0
    best_val_acc_after_activation_epoch = None
    best_model_path = run_dir / "best_model.pt"
    best_peer_model_path = run_dir / "best_peer_model.pt"
    last_current_lr = get_optimizer_lr(optimizer)
    last_current_peer_lr = get_optimizer_lr(peer_optimizer)
    last_effective_ssml_topk_ratio = args.ssml_topk_ratio
    last_effective_peer_true_prob_threshold = args.ssml_peer_true_prob_threshold
    last_effective_peer_student_prob_gap_min = args.ssml_peer_student_prob_gap_min
    supervised_loss_fn = lambda logits, targets: F.cross_entropy(logits, targets, reduction="none")
    imitation_loss_fn = build_imitation_loss_fn(args.classification_imitation_loss)
    ssml_elementwise_imitation_loss_fn = build_elementwise_kd_loss_fn(args.distill_temperature)
    hetero_ssml_one_way = args.hetero_ssml_one_way and pair_meta["is_heterogeneous_pair"]
    last_train_stats = {
        "train_loss": 0.0,
        "train_acc": 0.0,
        "student_positive_score_ratio": 0.0,
        "peer_positive_score_ratio": 0.0,
        "student_selected_ratio": 0.0,
        "peer_selected_ratio": 0.0,
        "student_selected_of_positive_ratio": 0.0,
        "peer_selected_of_positive_ratio": 0.0,
        "student_selected_score_mean": 0.0,
        "peer_selected_score_mean": 0.0,
        "student_hotspot_error_mean": 0.0,
        "student_background_error_mean": 0.0,
        "peer_hotspot_error_mean": 0.0,
        "peer_background_error_mean": 0.0,
        "student_hotspot_gap_mean": 0.0,
        "peer_hotspot_gap_mean": 0.0,
        "student_hotspot_error_share": 0.0,
        "peer_hotspot_error_share": 0.0,
        "student_incorrect_ratio": 0.0,
        "peer_incorrect_ratio": 0.0,
        "prediction_disagreement_ratio": 0.0,
        "student_teacher_usable_ratio": 0.0,
        "peer_teacher_usable_ratio": 0.0,
        "student_teacher_safe_ratio": 0.0,
        "peer_teacher_safe_ratio": 0.0,
        "student_useful_hard_ratio": 0.0,
        "peer_useful_hard_ratio": 0.0,
        "student_score_p90": 0.0,
        "peer_score_p90": 0.0,
        "student_worse_ratio": 0.0,
        "peer_worse_ratio": 0.0,
        "student_worse_update_ratio": 0.0,
        "peer_worse_update_ratio": 0.0,
        "student_update_ratio": 0.0,
        "peer_update_ratio": 0.0,
        "student_selected_per_class": 0.0,
        "peer_selected_per_class": 0.0,
        "student_aug_consistency_mean": 0.0,
        "peer_aug_consistency_mean": 0.0,
        "secondary_peer_agreement_ratio": 0.0,
        "secondary_peer_consensus_ratio": 0.0,
        "secondary_peer_aug_consistency_mean": 0.0,
        "anchor_loss_mean": 0.0,
        "preserved_disagreement_mean": 0.0,
        "disagreement_floor_gap_mean": 0.0,
        "student_safe_teacher_miss_rate_by_class": [0.0 for _ in range(num_classes)],
    }
    warmstart_pair_disagreement = 0.0
    disagreement_floor = 0.0
    last_effective_disagreement_floor = 0.0
    last_complement_scale = 1.0
    last_effective_extra_class_budget_scale = 0.0
    class_deficit_ema = torch.zeros(num_classes, dtype=torch.float32)
    current_class_budget = None
    last_student_recall_by_class = [0.0 for _ in range(num_classes)]
    last_peer_recall_by_class = [0.0 for _ in range(num_classes)]
    last_class_budget_by_class = (
        current_class_budget.tolist() if current_class_budget is not None else [0 for _ in range(num_classes)]
    )
    last_class_deficit_ema = class_deficit_ema.tolist()

    if args.method == "ssml" and peer_model is not None and args.ssml_disagreement_floor_ratio > 0.0:
        warmstart_stats = evaluate_pair_classification_details(
            model,
            peer_model,
            val_loader,
            device,
            num_classes,
        )
        warmstart_pair_disagreement = float(warmstart_stats["mean_pair_disagreement"])
        disagreement_floor = warmstart_pair_disagreement * max(args.ssml_disagreement_floor_ratio, 0.0)
        last_student_recall_by_class = list(warmstart_stats["student_recall_by_class"])
        last_peer_recall_by_class = list(warmstart_stats["peer_recall_by_class"])

    for epoch in range(1, args.epochs + 1):
        current_lr = compute_epoch_lr(
            args.lr,
            epoch=epoch,
            total_epochs=args.epochs,
            scheduler_name=args.lr_scheduler,
            warmup_epochs=args.scheduler_warmup_epochs,
            min_scale=args.scheduler_min_scale,
        )
        set_optimizer_lr(optimizer, current_lr)
        current_peer_lr = None
        if peer_optimizer is not None:
            current_peer_lr = compute_epoch_lr(
                args.lr,
                epoch=epoch,
                total_epochs=args.epochs,
                scheduler_name=args.lr_scheduler,
                warmup_epochs=args.scheduler_warmup_epochs,
                min_scale=args.scheduler_min_scale,
            )
            set_optimizer_lr(peer_optimizer, current_peer_lr)
        last_current_lr = current_lr
        last_current_peer_lr = current_peer_lr
        complement_scale = compute_epoch_ramp_scale(
            epoch=epoch,
            start_epoch=args.ssml_complement_ramp_start_epoch,
            end_epoch=args.ssml_complement_ramp_end_epoch,
        )
        effective_disagreement_floor = disagreement_floor * complement_scale
        effective_extra_class_budget_scale = args.ssml_extra_class_budget_scale * complement_scale
        last_complement_scale = complement_scale
        last_effective_disagreement_floor = effective_disagreement_floor
        last_effective_extra_class_budget_scale = effective_extra_class_budget_scale
        current_class_budget = (
            build_deficit_adjusted_class_budgets(
                args.ssml_per_class_budget,
                class_deficit_ema,
                effective_extra_class_budget_scale,
            )
            if args.ssml_class_balanced_topk and args.ssml_per_class_budget > 0
            else None
        )
        last_class_budget_by_class = (
            current_class_budget.tolist() if current_class_budget is not None else [0 for _ in range(num_classes)]
        )
        effective_lambda = compute_effective_lambda(
            args.lambda_imitation,
            epoch=epoch,
            method=args.method,
            warmup_epochs=args.warmup_epochs,
            decay_start_epoch=args.imitation_decay_start_epoch,
            decay_end_epoch=args.imitation_decay_end_epoch,
            decay_min_scale=args.imitation_decay_min_scale,
        )
        guidance_scale = compute_ssml_guidance_scale(
            epoch=epoch,
            method=args.method,
            warmup_epochs=args.warmup_epochs,
            decay_start_epoch=args.imitation_decay_start_epoch,
            decay_end_epoch=args.imitation_decay_end_epoch,
            decay_min_scale=args.imitation_decay_min_scale,
        )
        effective_ssml_topk_ratio = compute_scheduled_scalar(
            epoch=epoch,
            start_value=args.ssml_topk_ratio_start,
            end_value=args.ssml_topk_ratio_end,
            ramp_start_epoch=args.ssml_topk_ramp_start_epoch,
            ramp_end_epoch=args.ssml_topk_ramp_end_epoch,
            fallback_value=args.ssml_topk_ratio,
        )
        effective_peer_true_prob_threshold = compute_scheduled_scalar(
            epoch=epoch,
            start_value=args.ssml_peer_true_prob_threshold_start,
            end_value=args.ssml_peer_true_prob_threshold_end,
            ramp_start_epoch=args.ssml_topk_ramp_start_epoch,
            ramp_end_epoch=args.ssml_topk_ramp_end_epoch,
            fallback_value=args.ssml_peer_true_prob_threshold,
        )
        effective_peer_student_prob_gap_min = compute_scheduled_scalar(
            epoch=epoch,
            start_value=args.ssml_peer_student_prob_gap_min_start,
            end_value=args.ssml_peer_student_prob_gap_min_end,
            ramp_start_epoch=args.ssml_topk_ramp_start_epoch,
            ramp_end_epoch=args.ssml_topk_ramp_end_epoch,
            fallback_value=args.ssml_peer_student_prob_gap_min,
        )
        last_effective_ssml_topk_ratio = effective_ssml_topk_ratio
        last_effective_peer_true_prob_threshold = effective_peer_true_prob_threshold
        last_effective_peer_student_prob_gap_min = effective_peer_student_prob_gap_min
        effective_freeze_bn_stats = args.freeze_bn_stats and (
            args.freeze_bn_stats_until_epoch < 0 or epoch <= args.freeze_bn_stats_until_epoch
        )

        train_stats = train_one_epoch(
            model,
            peer_model,
            secondary_peer_model,
            train_loader,
            optimizer,
            peer_optimizer,
            device,
            supervised_loss_fn=supervised_loss_fn,
            imitation_loss_fn=imitation_loss_fn,
            ssml_elementwise_imitation_loss_fn=ssml_elementwise_imitation_loss_fn,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            ema_model=ema_model,
            ema_peer_model=ema_peer_model,
            model_ema_decay=args.model_ema_decay,
            lambda_imitation=effective_lambda,
            margin=args.margin,
            ssml_topk_ratio=effective_ssml_topk_ratio,
            ssml_supervised_hotspot_alpha=args.ssml_supervised_hotspot_alpha,
            ssml_topk_scope=args.ssml_topk_scope,
            ssml_supervised_weight_mode=args.ssml_supervised_weight_mode,
            ssml_gate_score_mode=args.ssml_gate_score_mode,
            ssml_score_transform=args.ssml_score_transform,
            ssml_guidance_mode=args.ssml_guidance_mode,
            ssml_peer_correct_only=args.ssml_peer_correct_only,
            ssml_student_incorrect_only=args.ssml_student_incorrect_only,
            ssml_disagreement_only=args.ssml_disagreement_only,
            ssml_class_balanced_topk=args.ssml_class_balanced_topk,
            ssml_per_class_budget=args.ssml_per_class_budget,
            num_classes=num_classes,
            ssml_dynamic_per_class_budget=current_class_budget,
            ssml_disagreement_floor=effective_disagreement_floor,
            ssml_complement_scale=complement_scale,
            ssml_secondary_peer_require_same_label=args.ssml_secondary_peer_require_same_label,
            ssml_secondary_peer_agreement_min=args.ssml_secondary_peer_agreement_min,
            ssml_peer_true_prob_threshold=effective_peer_true_prob_threshold,
            ssml_peer_student_prob_gap_min=effective_peer_student_prob_gap_min,
            ssml_student_true_prob_max=args.ssml_student_true_prob_max,
            ssml_aug_consistency_weight=args.ssml_aug_consistency_weight,
            ssml_aug_consistency_shift=args.ssml_aug_consistency_shift,
            ssml_aug_consistency_flip_prob=args.ssml_aug_consistency_flip_prob,
            ssml_aug_consistency_noise_std=args.ssml_aug_consistency_noise_std,
            ssml_peer_aug_consistency_min=args.ssml_peer_aug_consistency_min,
            ssml_student_aug_consistency_max=args.ssml_student_aug_consistency_max,
            ssml_peer_student_aug_consistency_gap_min=args.ssml_peer_student_aug_consistency_gap_min,
            guidance_scale=guidance_scale,
            method=args.method,
            freeze_bn_stats=effective_freeze_bn_stats,
            hetero_ssml_one_way=hetero_ssml_one_way,
            ssml_student_only=ssml_student_only,
            ssml_freeze_peer=ssml_freeze_peer,
            ssml_worse_only_update=args.ssml_worse_only_update,
            ssml_anchor_weight=args.ssml_anchor_weight,
            anchor_params=anchor_params,
        )
        last_train_stats = train_stats
        pair_val_details = None
        eval_model = ema_model if ema_model is not None else model
        eval_peer_model = (
            ema_peer_model if ema_peer_model is not None else peer_model
        )
        if peer_model is not None:
            pair_val_details = evaluate_pair_classification_details(
                eval_model,
                eval_peer_model,
                val_loader,
                device,
                num_classes,
            )
            va_loss = float(pair_val_details["val_loss"])
            va_acc = float(pair_val_details["val_acc"])
            peer_va_loss = float(pair_val_details["peer_val_loss"])
            peer_va_acc = float(pair_val_details["peer_val_acc"])
            last_student_recall_by_class = list(pair_val_details["student_recall_by_class"])
            last_peer_recall_by_class = list(pair_val_details["peer_recall_by_class"])
        else:
            va_loss, va_acc = evaluate(eval_model, val_loader, device)
            peer_va_loss = None
            peer_va_acc = None

        train_loss_curve.append(train_stats["train_loss"])
        train_acc_curve.append(train_stats["train_acc"])
        val_loss_curve.append(va_loss)
        val_acc_curve.append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(eval_model.state_dict(), best_model_path)
        if peer_va_loss is not None and peer_va_acc is not None:
            peer_val_loss_curve.append(peer_va_loss)
            peer_val_acc_curve.append(peer_va_acc)
            if peer_va_acc > best_peer_val_acc:
                best_peer_val_acc = peer_va_acc
                best_peer_epoch = epoch
                torch.save(eval_peer_model.state_dict(), best_peer_model_path)

        if args.method == "ssml" and args.ssml_class_balanced_topk and args.ssml_per_class_budget > 0:
            if pair_val_details is not None:
                student_recall_tensor = torch.tensor(pair_val_details["student_recall_by_class"], dtype=torch.float32)
                peer_recall_tensor = torch.tensor(pair_val_details["peer_recall_by_class"], dtype=torch.float32)
                deficit_signal = torch.clamp(peer_recall_tensor - student_recall_tensor, min=0.0)
            else:
                deficit_signal = torch.tensor(train_stats["student_safe_teacher_miss_rate_by_class"], dtype=torch.float32)
            momentum = float(max(0.0, min(args.ssml_deficit_ema_momentum, 0.999)))
            if momentum > 0.0:
                class_deficit_ema = momentum * class_deficit_ema + (1.0 - momentum) * deficit_signal
            else:
                class_deficit_ema = deficit_signal
            last_class_deficit_ema = class_deficit_ema.tolist()

        if first_active_epoch is None and train_stats["student_selected_ratio"] > 0.0:
            first_active_epoch = epoch
        if first_active_epoch is None:
            if va_acc > best_val_acc_before_activation:
                best_val_acc_before_activation = va_acc
                best_val_acc_before_activation_epoch = epoch
        else:
            if va_acc > best_val_acc_after_activation:
                best_val_acc_after_activation = va_acc
                best_val_acc_after_activation_epoch = epoch

        status = (
            f"[classification][epoch {epoch:03d}] "
            f"lr={current_lr:.5f} "
            f"lambda={effective_lambda:.4f} "
            f"g_scale={guidance_scale:.3f} "
            f"c_scale={complement_scale:.3f} "
            f"topk={effective_ssml_topk_ratio:.4f} "
            f"pthr={effective_peer_true_prob_threshold:.4f} "
            f"pgap={effective_peer_student_prob_gap_min:.4f} "
            f"train_loss={train_stats['train_loss']:.6f} train_acc={train_stats['train_acc']:.4f} "
            f"s_pos={train_stats['student_positive_score_ratio']:.4f} "
            f"s_sel={train_stats['student_selected_ratio']:.4f} "
            f"s_sel_pos={train_stats['student_selected_of_positive_ratio']:.4f} "
            f"s_bad={train_stats['student_incorrect_ratio']:.4f} "
            f"s_use={train_stats['student_teacher_usable_ratio']:.4f} "
            f"s_safe={train_stats['student_teacher_safe_ratio']:.4f} "
            f"s_uh={train_stats['student_useful_hard_ratio']:.4f} "
            f"dis={train_stats['prediction_disagreement_ratio']:.4f} "
            f"dis_keep={train_stats['preserved_disagreement_mean']:.4f} "
            f"dis_gap={train_stats['disagreement_floor_gap_mean']:.4f} "
            f"sec_ag={train_stats['secondary_peer_agreement_ratio']:.4f} "
            f"sec_cons={train_stats['secondary_peer_consensus_ratio']:.4f} "
            f"s_hot_ce={train_stats['student_hotspot_error_mean']:.4f} "
            f"s_bg_ce={train_stats['student_background_error_mean']:.4f} "
            f"s_gap={train_stats['student_hotspot_gap_mean']:.4f} "
            f"s_aug={train_stats['student_aug_consistency_mean']:.4f} "
            f"val_loss={va_loss:.6f} val_acc={va_acc:.4f}"
        )
        if peer_va_loss is not None and peer_va_acc is not None:
            status += (
                f" | p_lr={current_peer_lr if current_peer_lr is not None else 0.0:.5f} "
                f"p_pos={train_stats['peer_positive_score_ratio']:.4f} "
                f"p_sel={train_stats['peer_selected_ratio']:.4f} "
                f"p_sel_pos={train_stats['peer_selected_of_positive_ratio']:.4f} "
                f"p_bad={train_stats['peer_incorrect_ratio']:.4f} "
                f"p_use={train_stats['peer_teacher_usable_ratio']:.4f} "
                f"p_safe={train_stats['peer_teacher_safe_ratio']:.4f} "
                f"p_uh={train_stats['peer_useful_hard_ratio']:.4f} "
                f"p_hot_ce={train_stats['peer_hotspot_error_mean']:.4f} "
                f"p_bg_ce={train_stats['peer_background_error_mean']:.4f} "
                f"p_gap={train_stats['peer_hotspot_gap_mean']:.4f} "
                f"p_aug={train_stats['peer_aug_consistency_mean']:.4f} "
                f"peer_val_loss={peer_va_loss:.6f} peer_val_acc={peer_va_acc:.4f}"
            )
        print(status)
        append_jsonl(
            epoch_log_path,
            {
                "epoch": epoch,
                "method": args.method,
                "protocol_id": args.protocol_id,
                "hardware_profile": args.hardware_profile,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "optimizer": args.optimizer,
                "momentum": args.momentum,
                "lr": args.lr,
                "lr_scheduler": args.lr_scheduler,
                "scheduler_warmup_epochs": args.scheduler_warmup_epochs,
                "scheduler_min_scale": args.scheduler_min_scale,
                "current_lr": current_lr,
                "current_peer_lr": current_peer_lr,
                "label_smoothing": args.label_smoothing,
                "grad_clip": args.grad_clip,
                "model_ema_decay": args.model_ema_decay,
                "train_aug_mode": args.train_aug_mode,
                "model": args.model,
                "peer_model": pair_meta["peer_model"],
                "lambda_imitation": effective_lambda,
                "margin": args.margin,
                "ssml_topk_ratio": args.ssml_topk_ratio,
                "ssml_topk_ratio_start": args.ssml_topk_ratio_start,
                "ssml_topk_ratio_end": args.ssml_topk_ratio_end,
                "ssml_topk_ramp_start_epoch": args.ssml_topk_ramp_start_epoch,
                "ssml_topk_ramp_end_epoch": args.ssml_topk_ramp_end_epoch,
                "effective_ssml_topk_ratio": effective_ssml_topk_ratio,
                "ssml_topk_scope": args.ssml_topk_scope,
                "ssml_supervised_hotspot_alpha": args.ssml_supervised_hotspot_alpha,
                "ssml_supervised_weight_mode": args.ssml_supervised_weight_mode,
                "ssml_gate_score_mode": args.ssml_gate_score_mode,
                "ssml_score_transform": args.ssml_score_transform,
                "ssml_guidance_mode": args.ssml_guidance_mode,
                "ssml_peer_correct_only": args.ssml_peer_correct_only,
                "ssml_student_incorrect_only": args.ssml_student_incorrect_only,
                "ssml_student_true_prob_max": args.ssml_student_true_prob_max,
                "ssml_disagreement_only": args.ssml_disagreement_only,
                "ssml_class_balanced_topk": args.ssml_class_balanced_topk,
                "ssml_per_class_budget": args.ssml_per_class_budget,
                "ssml_disagreement_floor_ratio": args.ssml_disagreement_floor_ratio,
                "ssml_deficit_ema_momentum": args.ssml_deficit_ema_momentum,
                "ssml_extra_class_budget_scale": args.ssml_extra_class_budget_scale,
                "ssml_complement_ramp_start_epoch": args.ssml_complement_ramp_start_epoch,
                "ssml_complement_ramp_end_epoch": args.ssml_complement_ramp_end_epoch,
                "ssml_secondary_peer_init_checkpoint": loaded_secondary_peer_init_checkpoint,
                "ssml_secondary_peer_require_same_label": args.ssml_secondary_peer_require_same_label,
                "ssml_secondary_peer_agreement_min": args.ssml_secondary_peer_agreement_min,
                "ssml_peer_true_prob_threshold": args.ssml_peer_true_prob_threshold,
                "ssml_peer_student_prob_gap_min": args.ssml_peer_student_prob_gap_min,
                "ssml_peer_true_prob_threshold_start": args.ssml_peer_true_prob_threshold_start,
                "ssml_peer_true_prob_threshold_end": args.ssml_peer_true_prob_threshold_end,
                "ssml_peer_student_prob_gap_min_start": args.ssml_peer_student_prob_gap_min_start,
                "ssml_peer_student_prob_gap_min_end": args.ssml_peer_student_prob_gap_min_end,
                "effective_peer_true_prob_threshold": effective_peer_true_prob_threshold,
                "effective_peer_student_prob_gap_min": effective_peer_student_prob_gap_min,
                "ssml_aug_consistency_weight": args.ssml_aug_consistency_weight,
                "ssml_aug_consistency_shift": args.ssml_aug_consistency_shift,
                "ssml_aug_consistency_flip_prob": args.ssml_aug_consistency_flip_prob,
                "ssml_aug_consistency_noise_std": args.ssml_aug_consistency_noise_std,
                "ssml_peer_aug_consistency_min": args.ssml_peer_aug_consistency_min,
                "ssml_student_aug_consistency_max": args.ssml_student_aug_consistency_max,
                "ssml_peer_student_aug_consistency_gap_min": args.ssml_peer_student_aug_consistency_gap_min,
                "ssml_freeze_peer": ssml_freeze_peer,
                "ssml_worse_only_update": args.ssml_worse_only_update,
                "ssml_anchor_weight": args.ssml_anchor_weight,
                "init_checkpoint": loaded_init_checkpoint,
                "peer_init_checkpoint": loaded_peer_init_checkpoint,
                "secondary_peer_init_checkpoint": loaded_secondary_peer_init_checkpoint,
                "guidance_scale": guidance_scale,
                "complement_scale": complement_scale,
                "effective_disagreement_floor": effective_disagreement_floor,
                "effective_extra_class_budget_scale": effective_extra_class_budget_scale,
                "freeze_bn_stats": args.freeze_bn_stats,
                "freeze_bn_stats_until_epoch": args.freeze_bn_stats_until_epoch,
                "effective_freeze_bn_stats": effective_freeze_bn_stats,
                "train_loss": train_stats["train_loss"],
                "train_acc": train_stats["train_acc"],
                "student_positive_score_ratio": train_stats["student_positive_score_ratio"],
                "peer_positive_score_ratio": train_stats["peer_positive_score_ratio"],
                "student_selected_ratio": train_stats["student_selected_ratio"],
                "peer_selected_ratio": train_stats["peer_selected_ratio"],
                "student_selected_of_positive_ratio": train_stats["student_selected_of_positive_ratio"],
                "peer_selected_of_positive_ratio": train_stats["peer_selected_of_positive_ratio"],
                "student_selected_score_mean": train_stats["student_selected_score_mean"],
                "peer_selected_score_mean": train_stats["peer_selected_score_mean"],
                "student_hotspot_error_mean": train_stats["student_hotspot_error_mean"],
                "student_background_error_mean": train_stats["student_background_error_mean"],
                "peer_hotspot_error_mean": train_stats["peer_hotspot_error_mean"],
                "peer_background_error_mean": train_stats["peer_background_error_mean"],
                "student_hotspot_gap_mean": train_stats["student_hotspot_gap_mean"],
                "peer_hotspot_gap_mean": train_stats["peer_hotspot_gap_mean"],
                "student_hotspot_error_share": train_stats["student_hotspot_error_share"],
                "peer_hotspot_error_share": train_stats["peer_hotspot_error_share"],
                "student_incorrect_ratio": train_stats["student_incorrect_ratio"],
                "peer_incorrect_ratio": train_stats["peer_incorrect_ratio"],
                "prediction_disagreement_ratio": train_stats["prediction_disagreement_ratio"],
                "student_teacher_usable_ratio": train_stats["student_teacher_usable_ratio"],
                "peer_teacher_usable_ratio": train_stats["peer_teacher_usable_ratio"],
                "student_teacher_safe_ratio": train_stats["student_teacher_safe_ratio"],
                "peer_teacher_safe_ratio": train_stats["peer_teacher_safe_ratio"],
                "student_useful_hard_ratio": train_stats["student_useful_hard_ratio"],
                "peer_useful_hard_ratio": train_stats["peer_useful_hard_ratio"],
                "student_score_p90": train_stats["student_score_p90"],
                "peer_score_p90": train_stats["peer_score_p90"],
                "student_worse_ratio": train_stats["student_worse_ratio"],
                "peer_worse_ratio": train_stats["peer_worse_ratio"],
                "student_worse_update_ratio": train_stats["student_worse_update_ratio"],
                "peer_worse_update_ratio": train_stats["peer_worse_update_ratio"],
                "student_update_ratio": train_stats["student_update_ratio"],
                "peer_update_ratio": train_stats["peer_update_ratio"],
                "student_selected_per_class": train_stats["student_selected_per_class"],
                "peer_selected_per_class": train_stats["peer_selected_per_class"],
                "student_aug_consistency_mean": train_stats["student_aug_consistency_mean"],
                "peer_aug_consistency_mean": train_stats["peer_aug_consistency_mean"],
                "secondary_peer_agreement_ratio": train_stats["secondary_peer_agreement_ratio"],
                "secondary_peer_consensus_ratio": train_stats["secondary_peer_consensus_ratio"],
                "secondary_peer_aug_consistency_mean": train_stats["secondary_peer_aug_consistency_mean"],
                "anchor_loss_mean": train_stats["anchor_loss_mean"],
                "preserved_disagreement_mean": train_stats["preserved_disagreement_mean"],
                "disagreement_floor_gap_mean": train_stats["disagreement_floor_gap_mean"],
                "warmstart_pair_disagreement": warmstart_pair_disagreement,
                "disagreement_floor": disagreement_floor,
                "student_safe_teacher_miss_rate_by_class": train_stats["student_safe_teacher_miss_rate_by_class"],
                "student_val_recall_by_class": last_student_recall_by_class,
                "peer_val_recall_by_class": last_peer_recall_by_class,
                "class_deficit_ema": last_class_deficit_ema,
                "dynamic_class_budget_by_class": last_class_budget_by_class,
                "first_active_epoch_so_far": first_active_epoch,
                "best_val_acc_so_far": best_val_acc,
                "best_epoch_so_far": best_epoch,
                "best_val_acc_before_activation_so_far": best_val_acc_before_activation if best_val_acc_before_activation_epoch is not None else None,
                "best_val_acc_before_activation_epoch_so_far": best_val_acc_before_activation_epoch,
                "best_val_acc_after_activation_so_far": best_val_acc_after_activation if best_val_acc_after_activation_epoch is not None else None,
                "best_val_acc_after_activation_epoch_so_far": best_val_acc_after_activation_epoch,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "peer_val_loss": peer_va_loss,
                "peer_val_acc": peer_va_acc,
            },
        )

        if epoch % args.live_plot_interval == 0 or epoch == args.epochs:
            save_curves(
                run_dir / "curves.npz",
                train_loss=train_loss_curve,
                train_acc=train_acc_curve,
                val_loss=val_loss_curve,
                val_acc=val_acc_curve,
                train_loss1=train_loss_curve,
                train_acc1=train_acc_curve,
                val_loss1=val_loss_curve,
                val_acc1=val_acc_curve,
                val_loss2=peer_val_loss_curve,
                val_acc2=peer_val_acc_curve,
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

    if args.method == "ssml":
        before_text = "none" if best_val_acc_before_activation_epoch is None else f"{best_val_acc_before_activation:.4f}@{best_val_acc_before_activation_epoch}"
        after_text = "none" if best_val_acc_after_activation_epoch is None else f"{best_val_acc_after_activation:.4f}@{best_val_acc_after_activation_epoch}"
        print(
            "[classification][ssml_diag] "
            f"gate_mode={args.ssml_gate_score_mode} "
            f"score_transform={args.ssml_score_transform} "
            f"weight_mode={args.ssml_supervised_weight_mode} "
            f"freeze_peer={ssml_freeze_peer} "
            f"worse_only={args.ssml_worse_only_update} "
            f"anchor_w={args.ssml_anchor_weight:.5f} "
            f"student_incorrect_only={args.ssml_student_incorrect_only} "
            f"student_true_prob_max={args.ssml_student_true_prob_max:.3f} "
            f"useful_hard_ratio={last_train_stats['student_useful_hard_ratio']:.4f} "
            f"warmstart_dis={warmstart_pair_disagreement:.4f} "
            f"dis_floor={disagreement_floor:.4f} "
            f"comp_ramp={args.ssml_complement_ramp_start_epoch}->{args.ssml_complement_ramp_end_epoch} "
            f"comp_scale={last_complement_scale:.3f} "
            f"sec_ag={last_train_stats['secondary_peer_agreement_ratio']:.4f} "
            f"sec_cons={last_train_stats['secondary_peer_consensus_ratio']:.4f} "
            f"sec_ag_min={args.ssml_secondary_peer_agreement_min:.3f} "
            f"preserved_dis={last_train_stats['preserved_disagreement_mean']:.4f} "
            f"dis_gap={last_train_stats['disagreement_floor_gap_mean']:.4f} "
            f"peer_true_prob_threshold={last_effective_peer_true_prob_threshold:.3f} "
            f"peer_student_gap_min={last_effective_peer_student_prob_gap_min:.3f} "
            f"topk={last_effective_ssml_topk_ratio:.3f} "
            f"aug_consistency_w={args.ssml_aug_consistency_weight:.3f} "
            f"aug_shift={args.ssml_aug_consistency_shift} "
            f"aug_flip={args.ssml_aug_consistency_flip_prob:.2f} "
            f"aug_noise={args.ssml_aug_consistency_noise_std:.3f} "
            f"peer_aug_min={args.ssml_peer_aug_consistency_min:.3f} "
            f"student_aug_max={args.ssml_student_aug_consistency_max:.3f} "
            f"aug_gap_min={args.ssml_peer_student_aug_consistency_gap_min:.3f} "
            f"freeze_bn={int(args.freeze_bn_stats)} "
            f"freeze_bn_until={args.freeze_bn_stats_until_epoch} "
            f"first_active_epoch={first_active_epoch} "
            f"best_before_active={before_text} "
            f"best_after_active={after_text}"
        )

    save_curves(
        run_dir / "curves.npz",
        train_loss=train_loss_curve,
        train_acc=train_acc_curve,
        val_loss=val_loss_curve,
        val_acc=val_acc_curve,
        train_loss1=train_loss_curve,
        train_acc1=train_acc_curve,
        val_loss1=val_loss_curve,
        val_acc1=val_acc_curve,
        val_loss2=peer_val_loss_curve,
        val_acc2=peer_val_acc_curve,
    )
    summary = {
        "task": "classification",
        "dataset": args.dataset,
        "method": args.method,
        "protocol_id": args.protocol_id,
        "hardware_profile": args.hardware_profile,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "scheduler_warmup_epochs": args.scheduler_warmup_epochs,
        "scheduler_min_scale": args.scheduler_min_scale,
        "last_current_lr": last_current_lr,
        "last_current_peer_lr": last_current_peer_lr,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "model_ema_decay": args.model_ema_decay,
        "ema_evaluation": args.model_ema_decay > 0.0,
        "train_aug_mode": args.train_aug_mode,
        "device": str(device),
        "requested_device": args.device,
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
        "classification_imitation_loss": args.classification_imitation_loss,
        "distill_temperature": args.distill_temperature,
        "lambda_imitation": args.lambda_imitation,
        "margin": args.margin,
        "ssml_topk_ratio": args.ssml_topk_ratio,
        "ssml_topk_ratio_start": args.ssml_topk_ratio_start,
        "ssml_topk_ratio_end": args.ssml_topk_ratio_end,
        "ssml_topk_ramp_start_epoch": args.ssml_topk_ramp_start_epoch,
        "ssml_topk_ramp_end_epoch": args.ssml_topk_ramp_end_epoch,
        "ssml_topk_scope": args.ssml_topk_scope,
        "ssml_supervised_hotspot_alpha": args.ssml_supervised_hotspot_alpha,
        "ssml_supervised_weight_mode": args.ssml_supervised_weight_mode,
        "ssml_gate_score_mode": args.ssml_gate_score_mode,
        "ssml_score_transform": args.ssml_score_transform,
        "ssml_guidance_mode": args.ssml_guidance_mode,
        "ssml_peer_correct_only": args.ssml_peer_correct_only,
        "ssml_student_incorrect_only": args.ssml_student_incorrect_only,
        "ssml_student_true_prob_max": args.ssml_student_true_prob_max,
        "ssml_disagreement_only": args.ssml_disagreement_only,
        "ssml_class_balanced_topk": args.ssml_class_balanced_topk,
        "ssml_per_class_budget": args.ssml_per_class_budget,
        "ssml_disagreement_floor_ratio": args.ssml_disagreement_floor_ratio,
        "ssml_deficit_ema_momentum": args.ssml_deficit_ema_momentum,
        "ssml_extra_class_budget_scale": args.ssml_extra_class_budget_scale,
        "ssml_complement_ramp_start_epoch": args.ssml_complement_ramp_start_epoch,
        "ssml_complement_ramp_end_epoch": args.ssml_complement_ramp_end_epoch,
        "ssml_secondary_peer_init_checkpoint": loaded_secondary_peer_init_checkpoint,
        "ssml_secondary_peer_require_same_label": args.ssml_secondary_peer_require_same_label,
        "ssml_secondary_peer_agreement_min": args.ssml_secondary_peer_agreement_min,
        "ssml_peer_true_prob_threshold": args.ssml_peer_true_prob_threshold,
        "ssml_peer_student_prob_gap_min": args.ssml_peer_student_prob_gap_min,
        "ssml_peer_true_prob_threshold_start": args.ssml_peer_true_prob_threshold_start,
        "ssml_peer_true_prob_threshold_end": args.ssml_peer_true_prob_threshold_end,
        "ssml_peer_student_prob_gap_min_start": args.ssml_peer_student_prob_gap_min_start,
        "ssml_peer_student_prob_gap_min_end": args.ssml_peer_student_prob_gap_min_end,
        "ssml_aug_consistency_weight": args.ssml_aug_consistency_weight,
        "ssml_aug_consistency_shift": args.ssml_aug_consistency_shift,
        "ssml_aug_consistency_flip_prob": args.ssml_aug_consistency_flip_prob,
        "ssml_aug_consistency_noise_std": args.ssml_aug_consistency_noise_std,
        "ssml_peer_aug_consistency_min": args.ssml_peer_aug_consistency_min,
        "ssml_student_aug_consistency_max": args.ssml_student_aug_consistency_max,
        "ssml_peer_student_aug_consistency_gap_min": args.ssml_peer_student_aug_consistency_gap_min,
        "ssml_freeze_peer": ssml_freeze_peer,
        "ssml_worse_only_update": args.ssml_worse_only_update,
        "ssml_anchor_weight": args.ssml_anchor_weight,
        "init_checkpoint": loaded_init_checkpoint,
        "peer_init_checkpoint": loaded_peer_init_checkpoint,
        "secondary_peer_init_checkpoint": loaded_secondary_peer_init_checkpoint,
        "freeze_bn_stats": args.freeze_bn_stats,
        "freeze_bn_stats_until_epoch": args.freeze_bn_stats_until_epoch,
        "warmup_epochs": args.warmup_epochs,
        "imitation_decay_start_epoch": args.imitation_decay_start_epoch,
        "imitation_decay_end_epoch": args.imitation_decay_end_epoch,
        "imitation_decay_min_scale": args.imitation_decay_min_scale,
        "hetero_ssml_one_way": hetero_ssml_one_way,
        "ssml_one_way_rule": (
            "disabled_in_reweight_only"
            if args.ssml_guidance_mode == "reweight_only"
            else "hotspot_presence_bidirectional_fallback"
        ),
        "ssml_student_only": ssml_student_only,
        "ssml_rule": (
            "peer_better_sample_reweight_only"
            if args.ssml_guidance_mode == "reweight_only"
            else "peer_better_sample_weighted_kd"
        ),
        "ssml_gate_rule": f"{args.ssml_gate_score_mode}_topk_sample",
        "ssml_supervised_rule": (
            "sample_binary_reweighting"
            if args.ssml_supervised_weight_mode == "binary"
            else "sample_score_reweighting"
        ),
        "ssml_directionality": (
            "primary_model_only_frozen_peer"
            if peer_update_disabled
            else "bidirectional_hotspot_focus"
            if args.ssml_guidance_mode == "reweight_only"
            else "hetero_weaker_to_stronger_only"
            if hetero_ssml_one_way
            else "bidirectional"
        ),
        "epochs": args.epochs,
        "seed": args.seed,
        "epoch_log_path": str(epoch_log_path),
        "best_epoch": best_epoch,
        "best_val_acc_epoch": best_epoch,
        "first_active_epoch": first_active_epoch,
        "best_val_acc_before_activation": best_val_acc_before_activation if best_val_acc_before_activation_epoch is not None else None,
        "best_val_acc_before_activation_epoch": best_val_acc_before_activation_epoch,
        "best_val_acc_after_activation": best_val_acc_after_activation if best_val_acc_after_activation_epoch is not None else None,
        "best_val_acc_after_activation_epoch": best_val_acc_after_activation_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc_curve[-1],
        "final_val_loss": val_loss_curve[-1],
        "best_metric": best_val_acc,
        "best_metric_key": "acc",
        "final_metric": val_acc_curve[-1],
        "best_metric1": best_val_acc,
        "final_metric1": val_acc_curve[-1],
        "best_val_acc1": best_val_acc,
        "final_val_acc1": val_acc_curve[-1],
        "final_val_loss1": val_loss_curve[-1],
        "final_val1": val_acc_curve[-1],
        "student_positive_score_ratio": last_train_stats["student_positive_score_ratio"],
        "peer_positive_score_ratio": last_train_stats["peer_positive_score_ratio"],
        "student_selected_ratio": last_train_stats["student_selected_ratio"],
        "peer_selected_ratio": last_train_stats["peer_selected_ratio"],
        "student_selected_of_positive_ratio": last_train_stats["student_selected_of_positive_ratio"],
        "peer_selected_of_positive_ratio": last_train_stats["peer_selected_of_positive_ratio"],
        "student_selected_score_mean": last_train_stats["student_selected_score_mean"],
        "peer_selected_score_mean": last_train_stats["peer_selected_score_mean"],
        "student_hotspot_error_mean": last_train_stats["student_hotspot_error_mean"],
        "student_background_error_mean": last_train_stats["student_background_error_mean"],
        "peer_hotspot_error_mean": last_train_stats["peer_hotspot_error_mean"],
        "peer_background_error_mean": last_train_stats["peer_background_error_mean"],
        "student_hotspot_gap_mean": last_train_stats["student_hotspot_gap_mean"],
        "peer_hotspot_gap_mean": last_train_stats["peer_hotspot_gap_mean"],
        "student_hotspot_error_share": last_train_stats["student_hotspot_error_share"],
        "peer_hotspot_error_share": last_train_stats["peer_hotspot_error_share"],
        "num_parameters": count_parameters(model),
        "num_parameters1": count_parameters(model),
        "meta": data["meta"],
        "student_incorrect_ratio": last_train_stats["student_incorrect_ratio"],
        "peer_incorrect_ratio": last_train_stats["peer_incorrect_ratio"],
        "prediction_disagreement_ratio": last_train_stats["prediction_disagreement_ratio"],
        "student_teacher_usable_ratio": last_train_stats["student_teacher_usable_ratio"],
        "peer_teacher_usable_ratio": last_train_stats["peer_teacher_usable_ratio"],
        "student_teacher_safe_ratio": last_train_stats["student_teacher_safe_ratio"],
        "peer_teacher_safe_ratio": last_train_stats["peer_teacher_safe_ratio"],
        "student_useful_hard_ratio": last_train_stats["student_useful_hard_ratio"],
        "peer_useful_hard_ratio": last_train_stats["peer_useful_hard_ratio"],
        "student_score_p90": last_train_stats["student_score_p90"],
        "peer_score_p90": last_train_stats["peer_score_p90"],
        "student_worse_ratio": last_train_stats["student_worse_ratio"],
        "peer_worse_ratio": last_train_stats["peer_worse_ratio"],
        "student_worse_update_ratio": last_train_stats["student_worse_update_ratio"],
        "peer_worse_update_ratio": last_train_stats["peer_worse_update_ratio"],
        "student_update_ratio": last_train_stats["student_update_ratio"],
        "peer_update_ratio": last_train_stats["peer_update_ratio"],
        "student_selected_per_class": last_train_stats["student_selected_per_class"],
        "peer_selected_per_class": last_train_stats["peer_selected_per_class"],
        "student_aug_consistency_mean": last_train_stats["student_aug_consistency_mean"],
        "peer_aug_consistency_mean": last_train_stats["peer_aug_consistency_mean"],
        "secondary_peer_agreement_ratio": last_train_stats["secondary_peer_agreement_ratio"],
        "secondary_peer_consensus_ratio": last_train_stats["secondary_peer_consensus_ratio"],
        "secondary_peer_aug_consistency_mean": last_train_stats["secondary_peer_aug_consistency_mean"],
        "anchor_loss_mean": last_train_stats["anchor_loss_mean"],
        "preserved_disagreement_mean": last_train_stats["preserved_disagreement_mean"],
        "disagreement_floor_gap_mean": last_train_stats["disagreement_floor_gap_mean"],
        "warmstart_pair_disagreement": warmstart_pair_disagreement,
        "disagreement_floor": disagreement_floor,
        "last_effective_disagreement_floor": last_effective_disagreement_floor,
        "last_complement_scale": last_complement_scale,
        "last_effective_extra_class_budget_scale": last_effective_extra_class_budget_scale,
        "last_effective_ssml_topk_ratio": last_effective_ssml_topk_ratio,
        "last_effective_peer_true_prob_threshold": last_effective_peer_true_prob_threshold,
        "last_effective_peer_student_prob_gap_min": last_effective_peer_student_prob_gap_min,
        "student_safe_teacher_miss_rate_by_class": last_train_stats["student_safe_teacher_miss_rate_by_class"],
        "student_val_recall_by_class": last_student_recall_by_class,
        "peer_val_recall_by_class": last_peer_recall_by_class,
        "class_deficit_ema": last_class_deficit_ema,
        "dynamic_class_budget_by_class": last_class_budget_by_class,
        "best_model_path": str(best_model_path),
        "best_peer_model_path": str(best_peer_model_path) if peer_model is not None else None,
    }
    if peer_model is not None:
        summary.update(
            {
                "best_metric2": best_peer_val_acc,
                "best_epoch2": best_peer_epoch,
                "final_metric2": peer_val_acc_curve[-1],
                "best_val_acc2": best_peer_val_acc,
                "final_val_acc2": peer_val_acc_curve[-1],
                "final_val_loss2": peer_val_loss_curve[-1],
                "final_val2": peer_val_acc_curve[-1],
                "num_parameters2": count_parameters(peer_model),
            }
        )
    save_json(
        run_dir / "summary.json",
        summary,
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    if ema_model is not None:
        torch.save(ema_model.state_dict(), run_dir / "ema_model.pt")
    if peer_model is not None:
        torch.save(peer_model.state_dict(), run_dir / "peer_model.pt")
    if ema_peer_model is not None:
        torch.save(ema_peer_model.state_dict(), run_dir / "ema_peer_model.pt")
    print("[classification] done")


if __name__ == "__main__":
    main()
