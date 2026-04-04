from __future__ import annotations

import argparse
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
    "ode_cnn",
    "resnet18",
    "resnet18_gelu",
    "resnet34",
    "resnet34_gelu",
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
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--lambda-imitation", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--imitation-decay-start-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-end-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-min-scale", type=float, default=1.0)
    p.add_argument("--freeze-bn-stats", action="store_true")
    p.add_argument("--hetero-ssml-one-way", action="store_true")
    p.add_argument("--ssml-student-only", action="store_true")
    p.add_argument("--ssml-freeze-peer", action="store_true")
    p.add_argument("--ssml-worse-only-update", action="store_true")
    p.add_argument("--ssml-anchor-weight", type=float, default=0.0)
    p.add_argument("--ssml-topk-ratio", type=float, default=0.3)
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
    p.add_argument("--ssml-peer-true-prob-threshold", type=float, default=0.0)
    p.add_argument("--ssml-peer-student-prob-gap-min", type=float, default=0.0)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented_x = apply_batch_consistency_augmentation(
        x,
        max_shift=max_shift,
        flip_prob=flip_prob,
        noise_std=noise_std,
    )
    student_was_training = model.training
    peer_was_training = peer_model.training
    model.eval()
    peer_model.eval()
    with torch.no_grad():
        student_aug_prob = F.softmax(model(augmented_x), dim=1)
        peer_aug_prob = F.softmax(peer_model(augmented_x), dim=1)
    if student_was_training:
        model.train()
    else:
        model.eval()
    if peer_was_training:
        peer_model.train()
    else:
        peer_model.eval()
    return (
        compute_probability_consistency(student_prob_dist, student_aug_prob),
        compute_probability_consistency(peer_prob_dist, peer_aug_prob),
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
        if per_class_budget > 0:
            k = min(candidate_count, per_class_budget)
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
    loader,
    optimizer,
    peer_optimizer: Optional[torch.optim.Optimizer],
    device,
    supervised_loss_fn: Callable[..., torch.Tensor],
    imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ssml_elementwise_imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
    total_anchor_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if peer_optimizer is not None:
            peer_optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        supervised_loss = supervised_loss_fn(logits, y)
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
        anchor_loss_metric = 0.0

        if method == "independent":
            loss = supervised_loss.mean()
            loss.backward()
            optimizer.step()

        elif method == "dml":
            if peer_model is None or peer_optimizer is None:
                raise ValueError("peer_model and peer_optimizer are required when method='dml'")
            peer_logits = peer_model(x)
            peer_supervised_loss = supervised_loss_fn(peer_logits, y)
            w_student, w_peer = dml_weight_builder(
                supervised_loss.detach(),
                peer_supervised_loss.detach(),
                margin=margin,
            )
            if lambda_imitation <= 0.0:
                w_student = torch.zeros_like(w_student)
                w_peer = torch.zeros_like(w_peer)

            imitation_term_student = weighted_mean(imitation_loss_fn(logits, peer_logits.detach()), w_student)
            loss = supervised_loss.mean() + lambda_imitation * imitation_term_student

            imitation_term_peer = weighted_mean(imitation_loss_fn(peer_logits, logits.detach()), w_peer)
            peer_loss = peer_supervised_loss.mean() + lambda_imitation * imitation_term_peer

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
                    peer_logits = peer_model(x)
            else:
                peer_logits = peer_model(x)
            peer_supervised_loss = supervised_loss_fn(peer_logits, y)
            zero = logits.new_tensor(0.0)
            student_sample_loss = supervised_loss.detach()
            peer_sample_loss = peer_supervised_loss.detach()
            student_gap = student_sample_loss - peer_sample_loss
            peer_gap = peer_sample_loss - student_sample_loss
            student_probs = F.softmax(logits.detach(), dim=1)
            peer_probs = F.softmax(peer_logits.detach(), dim=1)
            student_pred = student_probs.argmax(dim=1)
            peer_pred = peer_probs.argmax(dim=1)
            student_correct = student_pred.eq(y)
            peer_correct = peer_pred.eq(y)
            prediction_disagreement = student_pred.ne(peer_pred)
            student_true_prob = student_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            peer_true_prob = peer_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            aug_consistency_enabled = (
                ssml_aug_consistency_weight > 0.0
                and (
                    ssml_aug_consistency_shift > 0
                    or ssml_aug_consistency_flip_prob > 0.0
                    or ssml_aug_consistency_noise_std > 0.0
                )
            )
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
                student_aug_consistency, peer_aug_consistency = compute_augmented_consistency_scores(
                    model,
                    peer_model,
                    x,
                    student_probs,
                    peer_probs,
                    max_shift=ssml_aug_consistency_shift,
                    flip_prob=ssml_aug_consistency_flip_prob,
                    noise_std=ssml_aug_consistency_noise_std,
                )
                if freeze_bn_stats:
                    apply_batchnorm_eval(model)
                    apply_batchnorm_eval(peer_model)
                student_consistency_factor, peer_consistency_factor = build_aug_consistency_reweight(
                    student_aug_consistency,
                    peer_aug_consistency,
                    weight=ssml_aug_consistency_weight,
                )
                student_scores = student_scores * student_consistency_factor
                peer_scores = peer_scores * peer_consistency_factor
                student_aug_consistency_mean = float(student_aug_consistency.mean().item())
                peer_aug_consistency_mean = float(peer_aug_consistency.mean().item())
                if ssml_peer_aug_consistency_min > 0.0:
                    student_scores = student_scores * (
                        peer_aug_consistency >= ssml_peer_aug_consistency_min
                    ).to(dtype=student_scores.dtype)
                    peer_scores = peer_scores * (
                        student_aug_consistency >= ssml_peer_aug_consistency_min
                    ).to(dtype=peer_scores.dtype)
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
                )
                mask_peer = build_class_balanced_topk_sample_mask(
                    peer_scores,
                    y,
                    ssml_topk_ratio,
                    scope=ssml_topk_scope,
                    per_class_budget=ssml_per_class_budget,
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
                supervised_loss,
                student_scores,
                mask_student,
                ssml_supervised_hotspot_alpha * guidance_scale,
                mode=ssml_supervised_weight_mode,
            )
            supervised_term_student = weighted_mean(supervised_loss, hotspot_weight_student)
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

            hotspot_weight_peer = build_sample_hotspot_weights(
                peer_supervised_loss,
                peer_scores,
                mask_peer,
                ssml_supervised_hotspot_alpha * guidance_scale,
                mode=ssml_supervised_weight_mode,
            )
            supervised_term_peer = weighted_mean(peer_supervised_loss, hotspot_weight_peer)
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
                optimizer.step()
            else:
                (loss + peer_loss).backward()
                optimizer.step()
                peer_optimizer.step()

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
        total_anchor_loss += anchor_loss_metric * batch_size
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
        "anchor_loss_mean": total_anchor_loss / total_count,
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
    peer_optimizer = None
    if uses_peer_model(args.method):
        peer_model = build_classification_model(
            model_name=pair_meta["peer_model"],
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
        ).to(device)
    loaded_init_checkpoint = load_model_checkpoint(model, args.init_checkpoint, "init")
    loaded_peer_init_checkpoint = None
    if peer_model is not None:
        loaded_peer_init_checkpoint = load_model_checkpoint(peer_model, args.peer_init_checkpoint, "peer_init")
    ssml_student_only = args.method == "ssml" and args.ssml_student_only and uses_peer_model(args.method)
    ssml_freeze_peer = args.method == "ssml" and args.ssml_freeze_peer and uses_peer_model(args.method)
    peer_update_disabled = ssml_student_only or ssml_freeze_peer
    if peer_model is not None and peer_update_disabled:
        for param in peer_model.parameters():
            param.requires_grad_(False)
        peer_model.eval()
    anchor_params = None
    if args.method == "ssml" and args.ssml_anchor_weight > 0.0:
        anchor_params = snapshot_trainable_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if peer_model is not None and not peer_update_disabled:
        peer_optimizer = torch.optim.AdamW(peer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        "anchor_loss_mean": 0.0,
    }

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
        guidance_scale = compute_ssml_guidance_scale(
            epoch=epoch,
            method=args.method,
            warmup_epochs=args.warmup_epochs,
            decay_start_epoch=args.imitation_decay_start_epoch,
            decay_end_epoch=args.imitation_decay_end_epoch,
            decay_min_scale=args.imitation_decay_min_scale,
        )

        train_stats = train_one_epoch(
            model,
            peer_model,
            train_loader,
            optimizer,
            peer_optimizer,
            device,
            supervised_loss_fn=supervised_loss_fn,
            imitation_loss_fn=imitation_loss_fn,
            ssml_elementwise_imitation_loss_fn=ssml_elementwise_imitation_loss_fn,
            lambda_imitation=effective_lambda,
            margin=args.margin,
            ssml_topk_ratio=args.ssml_topk_ratio,
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
            ssml_peer_true_prob_threshold=args.ssml_peer_true_prob_threshold,
            ssml_peer_student_prob_gap_min=args.ssml_peer_student_prob_gap_min,
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
            freeze_bn_stats=args.freeze_bn_stats,
            hetero_ssml_one_way=hetero_ssml_one_way,
            ssml_student_only=ssml_student_only,
            ssml_freeze_peer=ssml_freeze_peer,
            ssml_worse_only_update=args.ssml_worse_only_update,
            ssml_anchor_weight=args.ssml_anchor_weight,
            anchor_params=anchor_params,
        )
        last_train_stats = train_stats
        va_loss, va_acc = evaluate(model, val_loader, device)
        peer_va_loss = None
        peer_va_acc = None
        if peer_model is not None:
            peer_va_loss, peer_va_acc = evaluate(peer_model, val_loader, device)

        train_loss_curve.append(train_stats["train_loss"])
        train_acc_curve.append(train_stats["train_acc"])
        val_loss_curve.append(va_loss)
        val_acc_curve.append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        if peer_va_loss is not None and peer_va_acc is not None:
            peer_val_loss_curve.append(peer_va_loss)
            peer_val_acc_curve.append(peer_va_acc)
            if peer_va_acc > best_peer_val_acc:
                best_peer_val_acc = peer_va_acc
                best_peer_epoch = epoch
                torch.save(peer_model.state_dict(), run_dir / "best_peer_model.pt")

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
            f"lambda={effective_lambda:.4f} "
            f"g_scale={guidance_scale:.3f} "
            f"train_loss={train_stats['train_loss']:.6f} train_acc={train_stats['train_acc']:.4f} "
            f"s_pos={train_stats['student_positive_score_ratio']:.4f} "
            f"s_sel={train_stats['student_selected_ratio']:.4f} "
            f"s_sel_pos={train_stats['student_selected_of_positive_ratio']:.4f} "
            f"s_bad={train_stats['student_incorrect_ratio']:.4f} "
            f"s_use={train_stats['student_teacher_usable_ratio']:.4f} "
            f"s_safe={train_stats['student_teacher_safe_ratio']:.4f} "
            f"s_uh={train_stats['student_useful_hard_ratio']:.4f} "
            f"dis={train_stats['prediction_disagreement_ratio']:.4f} "
            f"s_hot_ce={train_stats['student_hotspot_error_mean']:.4f} "
            f"s_bg_ce={train_stats['student_background_error_mean']:.4f} "
            f"s_gap={train_stats['student_hotspot_gap_mean']:.4f} "
            f"s_aug={train_stats['student_aug_consistency_mean']:.4f} "
            f"val_loss={va_loss:.6f} val_acc={va_acc:.4f}"
        )
        if peer_va_loss is not None and peer_va_acc is not None:
            status += (
                f" | p_pos={train_stats['peer_positive_score_ratio']:.4f} "
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
                "model": args.model,
                "peer_model": pair_meta["peer_model"],
                "lambda_imitation": effective_lambda,
                "margin": args.margin,
                "ssml_topk_ratio": args.ssml_topk_ratio,
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
                "ssml_peer_true_prob_threshold": args.ssml_peer_true_prob_threshold,
                "ssml_peer_student_prob_gap_min": args.ssml_peer_student_prob_gap_min,
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
                "guidance_scale": guidance_scale,
                "freeze_bn_stats": args.freeze_bn_stats,
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
                "anchor_loss_mean": train_stats["anchor_loss_mean"],
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
            f"peer_true_prob_threshold={args.ssml_peer_true_prob_threshold:.3f} "
            f"peer_student_gap_min={args.ssml_peer_student_prob_gap_min:.3f} "
            f"aug_consistency_w={args.ssml_aug_consistency_weight:.3f} "
            f"aug_shift={args.ssml_aug_consistency_shift} "
            f"aug_flip={args.ssml_aug_consistency_flip_prob:.2f} "
            f"aug_noise={args.ssml_aug_consistency_noise_std:.3f} "
            f"peer_aug_min={args.ssml_peer_aug_consistency_min:.3f} "
            f"student_aug_max={args.ssml_student_aug_consistency_max:.3f} "
            f"aug_gap_min={args.ssml_peer_student_aug_consistency_gap_min:.3f} "
            f"freeze_bn={int(args.freeze_bn_stats)} "
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
        "ssml_peer_true_prob_threshold": args.ssml_peer_true_prob_threshold,
        "ssml_peer_student_prob_gap_min": args.ssml_peer_student_prob_gap_min,
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
        "freeze_bn_stats": args.freeze_bn_stats,
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
        "anchor_loss_mean": last_train_stats["anchor_loss_mean"],
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
    if peer_model is not None:
        torch.save(peer_model.state_dict(), run_dir / "peer_model.pt")
    print("[classification] done")


if __name__ == "__main__":
    main()
