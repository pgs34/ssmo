from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.methods import get_directional_weight_builder, mask_activation_ratio, weighted_mean
from src.models import build_time_series_model
from src.tasks import TimeSeriesDataConfig, build_time_series_dataloaders
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

TIME_SERIES_MODEL_CHOICES = [
    "dlinear",
    "transformer",
    "transformer_gelu",
    "transformer_wide",
    "patchtst",
    "gru",
    "neural_ode",
]
TIME_SERIES_METHOD_CHOICES = ["independent", "dml", "ssml"]
TIME_SERIES_SSML_EVAL_OUTPUT_CHOICES = ["guided", "student", "peer", "best_branch"]


class TimeSeriesCorrectionGate(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 1,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        output = torch.nn.Linear(hidden_dim, 1)
        if output_dim != 1:
            output = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.constant_(output.bias, init_bias)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            output,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        if self.output_dim == 1:
            return logits.squeeze(-1)
        return logits


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
    p.add_argument("--imitation-decay-start-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-end-epoch", type=int, default=-1)
    p.add_argument("--imitation-decay-min-scale", type=float, default=1.0)
    p.add_argument("--ssml-handoff-end-epoch", type=int, default=-1)
    p.add_argument("--hetero-ssml-one-way", action="store_true")
    p.add_argument("--ssml-student-only", action="store_true")
    p.add_argument("--ssml-freeze-peer", action="store_true")
    p.add_argument("--ssml-worse-only-update", action="store_true")
    p.add_argument("--ssml-anchor-weight", type=float, default=0.0)
    p.add_argument("--ssml-snapshot-anchor-start-epoch", type=int, default=-1)
    p.add_argument("--ssml-snapshot-anchor-weight", type=float, default=0.0)
    p.add_argument(
        "--ssml-snapshot-anchor-mask-mode",
        type=str,
        default="selected",
        choices=["selected", "all"],
    )
    p.add_argument("--ssml-peer-taper-end-epoch", type=int, default=-1)
    p.add_argument("--ssml-target-active-ratio-start", type=float, default=-1.0)
    p.add_argument("--ssml-target-active-ratio-end", type=float, default=-1.0)
    p.add_argument("--ssml-active-ratio-adapt-rate", type=float, default=0.0)
    p.add_argument("--ssml-topk-ratio", type=float, default=0.3)
    p.add_argument("--ssml-topk-scope", type=str, default="total", choices=["total", "positive"])
    p.add_argument("--ssml-max-selected-ratio", type=float, default=1.0)
    p.add_argument("--ssml-adaptive-dense-threshold", type=float, default=1.1)
    p.add_argument("--ssml-adaptive-dense-topk-ratio", type=float, default=1.0)
    p.add_argument(
        "--ssml-adaptive-dense-topk-scope",
        type=str,
        default="positive",
        choices=["total", "positive"],
    )
    p.add_argument("--ssml-adaptive-dense-max-selected-ratio", type=float, default=1.0)
    p.add_argument("--ssml-adaptive-dense-score-smoothing-kernel", type=int, default=-1)
    p.add_argument("--ssml-adaptive-dense-window-expand-kernel", type=int, default=-1)
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
            "relative_advantage",
            "absolute_gap",
            "peer_better_student_error",
            "peer_better_student_error_relgain",
        ],
    )
    p.add_argument(
        "--ssml-score-transform",
        type=str,
        default="none",
        choices=["none", "sqrt", "log1p"],
    )
    p.add_argument("--ssml-positive-upper-quantile", type=float, default=1.0)
    p.add_argument("--ssml-score-smoothing-kernel", type=int, default=1)
    p.add_argument("--ssml-window-score-kernel", type=int, default=1)
    p.add_argument("--ssml-window-expand-kernel", type=int, default=1)
    p.add_argument("--ssml-tail-start-ratio", type=float, default=0.0)
    p.add_argument("--ssml-residual-beta", type=float, default=1.0)
    p.add_argument("--ssml-ema-decay", type=float, default=0.0)
    p.add_argument(
        "--ssml-imitation-space",
        type=str,
        default="raw",
        choices=["raw", "delta", "residual"],
    )
    p.add_argument("--ssml-residual-space-kernel", type=int, default=9)
    p.add_argument("--ssml-conflict-aware-projection", action="store_true")
    p.add_argument(
        "--ssml-guidance-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "reweight_only", "corrective", "delta_fusion"],
    )
    p.add_argument(
        "--ssml-eval-output-mode",
        type=str,
        default="guided",
        choices=TIME_SERIES_SSML_EVAL_OUTPUT_CHOICES,
    )
    p.add_argument("--ssml-correction-gate-hidden-dim", type=int, default=32)
    p.add_argument("--ssml-correction-gate-dropout", type=float, default=0.0)
    p.add_argument("--ssml-correction-init-bias", type=float, default=0.0)
    p.add_argument("--ssml-correction-sparsity-weight", type=float, default=0.0)
    p.add_argument("--ssml-correction-threshold", type=float, default=0.5)
    p.add_argument("--ssml-correction-ramp-start-epoch", type=int, default=1)
    p.add_argument("--ssml-correction-ramp-end-epoch", type=int, default=1)
    p.add_argument("--ssml-correction-freeze-student-epochs", type=int, default=0)
    p.add_argument("--ssml-correction-student-train-end-epoch", type=int, default=-1)
    p.add_argument("--ssml-correction-only", action="store_true")
    p.add_argument("--ssml-correction-tail-start-ratio", type=float, default=0.0)
    p.add_argument("--ssml-correction-regime-focus-quantile", type=float, default=0.0)
    p.add_argument("--ssml-correction-focus-loss-alpha", type=float, default=0.0)
    p.add_argument("--ssml-correction-peer-advantage-quantile", type=float, default=0.0)
    p.add_argument("--ssml-correction-peer-advantage-min", type=float, default=0.0)
    p.add_argument("--ssml-correction-peer-advantage-smoothing-kernel", type=int, default=1)
    p.add_argument("--ssml-correction-budget-ratio", type=float, default=0.0)
    p.add_argument("--ssml-router-bin-endpoints", type=str, default="")
    p.add_argument("--ssml-router-ema-decay", type=float, default=0.0)
    p.add_argument("--ssml-trend-only-teaching", action="store_true")
    p.add_argument("--ssml-fusion-tail-start-ratio", type=float, default=0.0)
    p.add_argument("--ssml-fusion-max-scale", type=float, default=1.0)
    p.add_argument(
        "--ssml-correction-feature-mode",
        type=str,
        default="basic",
        choices=["basic", "trend_residual"],
    )
    p.add_argument("--ssml-correction-use-regime-features", action="store_true")
    p.add_argument("--ssml-correction-decomposition-kernel", type=int, default=9)
    p.add_argument("--ssml-correction-trend-scale", type=float, default=1.0)
    p.add_argument("--ssml-correction-residual-scale", type=float, default=1.0)
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--peer-init-checkpoint", type=str, default=None)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--early-stop-min-epochs", type=int, default=0)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
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


def _positive_activation_ratio(weights: torch.Tensor) -> float:
    if weights.numel() == 0:
        return 0.0
    return float((weights > 0).to(dtype=torch.float32).mean().item())


def build_topk_element_mask(
    scores: torch.Tensor,
    topk_ratio: float,
    *,
    scope: str = "total",
    positive_upper_quantile: float = 1.0,
) -> torch.Tensor:
    if scores.ndim < 2:
        raise ValueError(f"Expected elementwise scores with shape [B, ...], got: {tuple(scores.shape)}")
    if topk_ratio <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)

    positive = scores > 0
    flat_scores = scores.reshape(scores.shape[0], -1)
    flat_positive = positive.reshape(positive.shape[0], -1)
    candidate_mask = flat_positive.clone()
    if 0.0 < positive_upper_quantile < 1.0:
        for i in range(flat_scores.shape[0]):
            pos_scores = flat_scores[i][flat_positive[i]]
            if pos_scores.numel() == 0:
                candidate_mask[i].zero_()
                continue
            cutoff = torch.quantile(pos_scores, positive_upper_quantile)
            candidate_mask[i] = flat_positive[i] & (flat_scores[i] <= cutoff)

    element_count = flat_scores.shape[1]
    if element_count == 0:
        return torch.zeros_like(scores, dtype=torch.bool)

    if scope == "positive":
        positive_counts = candidate_mask.sum(dim=1)
        if int(positive_counts.max().item()) == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
    elif scope != "total":
        raise ValueError(f"Unsupported SSML top-k scope: {scope}")

    mask = torch.zeros_like(flat_positive)
    for i in range(flat_scores.shape[0]):
        candidate_count = int(candidate_mask[i].sum().item()) if scope == "positive" else element_count
        if candidate_count == 0:
            continue
        k = max(1, min(candidate_count, math.ceil(candidate_count * topk_ratio)))
        if scope == "total" and k >= element_count:
            mask[i] = candidate_mask[i]
            continue
        if scope == "positive" and k >= int(candidate_mask[i].sum().item()):
            mask[i] = candidate_mask[i]
            continue
        masked_scores = flat_scores[i].masked_fill(~candidate_mask[i], float("-inf"))
        topk_values, topk_indices = torch.topk(masked_scores, k=k, dim=0)
        keep_topk = topk_values > 0
        row_mask = torch.zeros_like(candidate_mask[i])
        row_mask[topk_indices[keep_topk]] = True
        mask[i] = row_mask & candidate_mask[i]
    return mask.reshape_as(scores)


def build_elementwise_hotspot_weights(
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
    flat_scores = torch.where(
        hotspot_mask,
        torch.clamp(hotspot_scores, min=0.0),
        torch.zeros_like(hotspot_scores),
    ).reshape(reference.shape[0], -1)
    flat_mask = hotspot_mask.reshape(reference.shape[0], -1)
    positive_count = flat_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=reference.dtype)
    mean_positive = flat_scores.sum(dim=1, keepdim=True) / positive_count
    normalized = flat_scores / torch.clamp(mean_positive, min=1e-6)
    normalized = torch.where(flat_mask, normalized, torch.zeros_like(normalized))
    normalized = torch.clamp(normalized, min=0.0, max=4.0)
    return weights + alpha * normalized.reshape_as(reference)


def limit_element_mask_by_ratio(
    scores: torch.Tensor,
    mask: torch.Tensor,
    max_ratio: float,
) -> torch.Tensor:
    if max_ratio >= 1.0 or mask.numel() == 0:
        return mask
    flat_scores = scores.reshape(scores.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    element_count = flat_scores.shape[1]
    if element_count == 0:
        return mask
    max_k = max(1, min(element_count, math.ceil(element_count * max_ratio)))
    limited = torch.zeros_like(flat_mask)
    for i in range(flat_scores.shape[0]):
        candidate_count = int(flat_mask[i].sum().item())
        if candidate_count == 0:
            continue
        k = min(candidate_count, max_k)
        if k >= candidate_count:
            limited[i] = flat_mask[i]
            continue
        masked_scores = flat_scores[i].masked_fill(~flat_mask[i], float("-inf"))
        topk_values, topk_indices = torch.topk(masked_scores, k=k, dim=0)
        keep = topk_values > 0
        row_mask = torch.zeros_like(flat_mask[i])
        row_mask[topk_indices[keep]] = True
        limited[i] = row_mask & flat_mask[i]
    return limited.reshape_as(mask)


def build_elementwise_score_weights(
    reference: torch.Tensor,
    hotspot_scores: torch.Tensor,
    hotspot_mask: torch.Tensor,
) -> torch.Tensor:
    weights = torch.zeros_like(reference, dtype=reference.dtype)
    if hotspot_mask.numel() == 0:
        return weights
    flat_scores = torch.where(
        hotspot_mask,
        torch.clamp(hotspot_scores, min=0.0),
        torch.zeros_like(hotspot_scores),
    ).reshape(reference.shape[0], -1)
    denom = flat_scores.sum(dim=1, keepdim=True)
    normalized = flat_scores / torch.clamp(denom, min=1e-6)
    return normalized.reshape_as(reference)


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


def smooth_time_series_scores(
    scores: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    if kernel_size <= 1 or scores.ndim < 3:
        return scores
    effective_kernel = max(1, int(kernel_size))
    if effective_kernel % 2 == 0:
        effective_kernel += 1
    batch, horizon = scores.shape[0], scores.shape[1]
    trailing_shape = scores.shape[2:]
    flat = scores.permute(0, *range(2, scores.ndim), 1).reshape(-1, 1, horizon)
    pad = effective_kernel // 2
    padded = F.pad(flat, (pad, pad), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel_size=effective_kernel, stride=1)
    return smoothed.reshape(batch, *trailing_shape, horizon).permute(0, scores.ndim - 1, *range(1, scores.ndim - 1))


def expand_time_series_mask(
    mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    if kernel_size <= 1 or mask.ndim < 3:
        return mask
    effective_kernel = max(1, int(kernel_size))
    if effective_kernel % 2 == 0:
        effective_kernel += 1
    batch, horizon = mask.shape[0], mask.shape[1]
    trailing_shape = mask.shape[2:]
    flat = mask.to(dtype=torch.float32).permute(0, *range(2, mask.ndim), 1).reshape(-1, 1, horizon)
    pad = effective_kernel // 2
    padded = F.pad(flat, (pad, pad), mode="constant", value=0.0)
    expanded = F.max_pool1d(padded, kernel_size=effective_kernel, stride=1)
    return expanded.reshape(batch, *trailing_shape, horizon).permute(0, mask.ndim - 1, *range(1, mask.ndim - 1)) > 0


def build_residual_teacher_target(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    beta: float,
    *,
    trend_only: bool = False,
    trend_kernel: int = 9,
) -> torch.Tensor:
    base = student_pred.detach()
    teacher = teacher_pred.detach()
    beta = float(max(0.0, min(1.0, beta)))
    if trend_only:
        student_trend, student_residual = decompose_forecast_trend_residual(base, trend_kernel)
        teacher_trend, _ = decompose_forecast_trend_residual(teacher, trend_kernel)
        if beta >= 1.0:
            target_trend = teacher_trend
        elif beta <= 0.0:
            target_trend = student_trend
        else:
            target_trend = student_trend + beta * (teacher_trend - student_trend)
        return target_trend + student_residual
    if beta >= 1.0:
        return teacher
    if beta <= 0.0:
        return base
    return base + beta * (teacher - base)


def masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or mask.numel() == 0:
        return 0.0
    active = bool(mask.any().item())
    if not active:
        return 0.0
    return float(values[mask].mean().item())


def mask_ratio(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.to(dtype=torch.float32).mean().item())


def safe_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return float(torch.quantile(flat, q).item())


def compute_ssml_element_scores(
    student_error: torch.Tensor,
    peer_error: torch.Tensor,
    *,
    margin: float,
    score_mode: str,
    score_transform: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_mode == "relative_advantage":
        peer_advantage = (student_error - peer_error) / torch.clamp(student_error, min=1e-6)
        student_advantage = (peer_error - student_error) / torch.clamp(peer_error, min=1e-6)
        student_scores = torch.clamp(peer_advantage - margin, min=0.0)
        peer_scores = torch.clamp(student_advantage - margin, min=0.0)
        return student_scores, peer_scores

    if score_mode == "absolute_gap":
        student_scores = torch.clamp((student_error - peer_error) - margin, min=0.0)
        peer_scores = torch.clamp((peer_error - student_error) - margin, min=0.0)
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

    if score_mode == "peer_better_student_error_relgain":
        student_signal = transform_ssml_score_signal(student_error, score_transform)
        peer_signal = transform_ssml_score_signal(peer_error, score_transform)
        student_rel_gain = torch.clamp(
            (student_error - peer_error) / torch.clamp(student_error, min=1e-6) - margin,
            min=0.0,
        )
        peer_rel_gain = torch.clamp(
            (peer_error - student_error) / torch.clamp(peer_error, min=1e-6) - margin,
            min=0.0,
        )
        student_scores = student_signal * student_rel_gain
        peer_scores = peer_signal * peer_rel_gain
        return student_scores, peer_scores

    raise ValueError(f"Unsupported SSML gate score mode: {score_mode}")


def resolve_adaptive_dense_ssml_params(
    *,
    positive_ratio: float,
    topk_ratio: float,
    topk_scope: str,
    max_selected_ratio: float,
    score_smoothing_kernel: int,
    window_expand_kernel: int,
    adaptive_dense_threshold: float,
    adaptive_dense_topk_ratio: float,
    adaptive_dense_topk_scope: str,
    adaptive_dense_max_selected_ratio: float,
    adaptive_dense_score_smoothing_kernel: int,
    adaptive_dense_window_expand_kernel: int,
) -> dict[str, float | int | str | bool]:
    dense_mode = adaptive_dense_threshold <= 1.0 and positive_ratio >= adaptive_dense_threshold
    smoothing_kernel = score_smoothing_kernel
    window_expand = window_expand_kernel
    if dense_mode:
        if adaptive_dense_score_smoothing_kernel > 0:
            smoothing_kernel = adaptive_dense_score_smoothing_kernel
        if adaptive_dense_window_expand_kernel > 0:
            window_expand = adaptive_dense_window_expand_kernel
    return {
        "dense_mode": dense_mode,
        "topk_ratio": adaptive_dense_topk_ratio if dense_mode else topk_ratio,
        "topk_scope": adaptive_dense_topk_scope if dense_mode else topk_scope,
        "max_selected_ratio": adaptive_dense_max_selected_ratio if dense_mode else max_selected_ratio,
        "score_smoothing_kernel": smoothing_kernel,
        "window_expand_kernel": window_expand,
    }


def build_forecast_delta_representation(
    forecast: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    if forecast.ndim != 3:
        raise ValueError(f"Expected forecast with shape [B, H, C], got {tuple(forecast.shape)}")
    if history.ndim != 3:
        raise ValueError(f"Expected history with shape [B, L, C], got {tuple(history.shape)}")
    target_dim = forecast.shape[-1]
    if history.shape[-1] < target_dim:
        raise ValueError(
            f"History feature dimension {history.shape[-1]} is smaller than forecast target dimension {target_dim}"
        )
    last_obs = history[:, -1:, :target_dim]
    first_delta = forecast[:, :1, :] - last_obs
    if forecast.shape[1] == 1:
        return first_delta
    future_deltas = forecast[:, 1:, :] - forecast[:, :-1, :]
    return torch.cat([first_delta, future_deltas], dim=1)


def build_forecast_trend_representation(
    forecast: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    if kernel_size <= 1:
        return forecast
    effective_kernel = max(1, int(kernel_size))
    if effective_kernel % 2 == 0:
        effective_kernel += 1
    batch, horizon = forecast.shape[0], forecast.shape[1]
    trailing_shape = forecast.shape[2:]
    flat = forecast.permute(0, *range(2, forecast.ndim), 1).reshape(-1, 1, horizon)
    pad = effective_kernel // 2
    padded = F.pad(flat, (pad, pad), mode="replicate")
    trend = F.avg_pool1d(padded, kernel_size=effective_kernel, stride=1)
    return trend.reshape(batch, *trailing_shape, horizon).permute(0, forecast.ndim - 1, *range(1, forecast.ndim - 1))


def decompose_forecast_trend_residual(
    forecast: torch.Tensor,
    kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    trend = build_forecast_trend_representation(forecast, kernel_size)
    return trend, forecast - trend


def build_forecast_residual_representation(
    forecast: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    _, residual = decompose_forecast_trend_residual(forecast, kernel_size)
    return residual


def build_imitation_representation(
    forecast: torch.Tensor,
    history: torch.Tensor,
    imitation_space: str,
    residual_space_kernel: int,
) -> torch.Tensor:
    if imitation_space == "raw":
        return forecast
    if imitation_space == "delta":
        return build_forecast_delta_representation(forecast, history)
    if imitation_space == "residual":
        return build_forecast_residual_representation(forecast, residual_space_kernel)
    raise ValueError(f"Unsupported SSML imitation space: {imitation_space}")


def build_tail_horizon_mask(
    reference: torch.Tensor,
    tail_start_ratio: float,
) -> torch.Tensor:
    if reference.ndim < 2:
        raise ValueError(f"Expected time-series tensor with shape [B, H, ...], got {tuple(reference.shape)}")
    if tail_start_ratio <= 0.0:
        return torch.ones_like(reference, dtype=torch.bool)
    horizon = reference.shape[1]
    if horizon <= 0:
        return torch.zeros_like(reference, dtype=torch.bool)
    start_index = int(math.floor(horizon * tail_start_ratio))
    start_index = max(0, min(horizon - 1, start_index))
    shape = [1] * reference.ndim
    shape[1] = horizon
    mask = torch.zeros(shape, dtype=torch.bool, device=reference.device)
    mask[:, start_index:, ...] = True
    return mask.expand_as(reference)


def parse_horizon_router_bin_endpoints(
    spec: str,
    pred_len: int,
) -> list[int]:
    pred_len = max(int(pred_len), 1)
    if spec.strip():
        endpoints = []
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            endpoints.append(int(token))
    else:
        endpoints = [min(8, pred_len), min(16, pred_len), pred_len]
    cleaned = sorted({max(1, min(pred_len, value)) for value in endpoints})
    if not cleaned or cleaned[-1] != pred_len:
        cleaned.append(pred_len)
    return cleaned


def build_horizon_router_tensor(
    reference: torch.Tensor,
    router_weights: Optional[torch.Tensor],
    bin_endpoints: Optional[list[int]],
) -> torch.Tensor:
    if router_weights is None or router_weights.numel() == 0 or not bin_endpoints:
        return torch.ones_like(reference, dtype=reference.dtype)
    router_weights = router_weights.to(device=reference.device, dtype=reference.dtype)
    scale = torch.ones(
        [1, reference.shape[1], *([1] * max(reference.ndim - 2, 0))],
        device=reference.device,
        dtype=reference.dtype,
    )
    start = 0
    for idx, end in enumerate(bin_endpoints):
        end = max(start + 1, min(int(end), reference.shape[1]))
        weight = router_weights[min(idx, int(router_weights.numel()) - 1)]
        scale[:, start:end, ...] = weight
        start = end
        if start >= reference.shape[1]:
            break
    return scale.expand_as(reference)


def compute_horizon_bin_relative_gains(
    student_error: torch.Tensor,
    teacher_error: torch.Tensor,
    bin_endpoints: Optional[list[int]],
) -> torch.Tensor:
    if not bin_endpoints:
        return student_error.new_zeros((0,), dtype=torch.float32)
    gains = []
    start = 0
    for end in bin_endpoints:
        end = max(start + 1, min(int(end), student_error.shape[1]))
        student_slice = student_error[:, start:end, ...]
        teacher_slice = teacher_error[:, start:end, ...]
        student_mean = student_slice.mean()
        teacher_mean = teacher_slice.mean()
        if float(student_mean.item()) <= 0.0:
            gains.append(student_mean.new_tensor(0.0))
        else:
            gains.append(torch.clamp((student_mean - teacher_mean) / torch.clamp(student_mean, min=1e-6), min=0.0))
        start = end
        if start >= student_error.shape[1]:
            break
    return torch.stack(gains).to(dtype=torch.float32)


def update_horizon_router_state(
    previous_gains: Optional[torch.Tensor],
    current_gains: torch.Tensor,
    ema_decay: float,
) -> torch.Tensor:
    current_gains = torch.clamp(current_gains.detach().to(dtype=torch.float32), min=0.0)
    if current_gains.numel() == 0:
        if previous_gains is None:
            return current_gains
        return previous_gains.to(dtype=torch.float32)
    if previous_gains is None or previous_gains.numel() != current_gains.numel():
        return current_gains
    ema_decay = float(max(0.0, min(ema_decay, 0.9999)))
    if ema_decay <= 0.0:
        return current_gains
    return ema_decay * previous_gains.to(dtype=torch.float32) + (1.0 - ema_decay) * current_gains


def build_horizon_router_weights(router_gains: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if router_gains is None:
        return None
    router_gains = torch.clamp(router_gains.detach().to(dtype=torch.float32), min=0.0)
    if router_gains.numel() == 0:
        return router_gains
    max_gain = float(router_gains.max().item())
    if max_gain <= 0.0:
        return torch.ones_like(router_gains)
    normalized = router_gains / max_gain
    return 0.25 + 0.75 * normalized


def apply_ssml_handoff(
    *,
    epoch: int,
    handoff_end_epoch: int,
    lambda_imitation: float,
    guidance_scale: float,
    correction_apply_scale: float,
) -> tuple[float, float, float, bool]:
    if handoff_end_epoch >= 0 and epoch > handoff_end_epoch:
        return 0.0, 0.0, 0.0, True
    return lambda_imitation, guidance_scale, correction_apply_scale, False


def conflict_project_gradients(
    params: list[torch.nn.Parameter],
    supervised_objective: torch.Tensor,
    imitation_objective: torch.Tensor,
) -> tuple[float, float, bool]:
    sup_grads = torch.autograd.grad(supervised_objective, params, retain_graph=True, allow_unused=True)
    im_grads = (
        torch.autograd.grad(imitation_objective, params, retain_graph=True, allow_unused=True)
        if float(imitation_objective.detach().item()) > 0.0
        else tuple(None for _ in params)
    )

    device = supervised_objective.device
    dot = torch.zeros((), device=device)
    sup_sq = torch.zeros((), device=device)
    im_sq = torch.zeros((), device=device)
    grads_to_apply: list[torch.Tensor | None] = []

    normalized_sup = []
    normalized_im = []
    for sup_grad, im_grad, param in zip(sup_grads, im_grads, params):
        sup_tensor = sup_grad if sup_grad is not None else torch.zeros_like(param)
        im_tensor = im_grad if im_grad is not None else torch.zeros_like(param)
        dot = dot + torch.sum(sup_tensor * im_tensor)
        sup_sq = sup_sq + torch.sum(sup_tensor * sup_tensor)
        im_sq = im_sq + torch.sum(im_tensor * im_tensor)
        normalized_sup.append(sup_tensor)
        normalized_im.append(im_tensor)

    projection_applied = bool(dot.item() < 0.0 and sup_sq.item() > 0.0 and im_sq.item() > 0.0)
    if projection_applied:
        proj_scale = dot / torch.clamp(sup_sq, min=1e-12)
    else:
        proj_scale = torch.zeros((), device=device)

    for sup_tensor, im_tensor, param in zip(normalized_sup, normalized_im, params):
        if projection_applied:
            im_tensor = im_tensor - proj_scale * sup_tensor
        total = sup_tensor + im_tensor
        param.grad = total.detach().clone()
        grads_to_apply.append(param.grad)

    cosine = 0.0
    if sup_sq.item() > 0.0 and im_sq.item() > 0.0:
        cosine = float((dot / torch.sqrt(torch.clamp(sup_sq * im_sq, min=1e-12))).item())
    return float(dot.item()), cosine, projection_applied


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


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str], label: str) -> Optional[str]:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    print(f"[time_series] loaded {label}_checkpoint={path}")
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


@torch.no_grad()
def refresh_frozen_snapshot(
    snapshot_model: Optional[torch.nn.Module],
    online_model: torch.nn.Module,
) -> torch.nn.Module:
    if snapshot_model is None:
        return clone_ema_model(online_model)
    snapshot_model.load_state_dict(copy.deepcopy(online_model.state_dict()))
    snapshot_model.eval()
    for param in snapshot_model.parameters():
        param.requires_grad_(False)
    return snapshot_model


def compute_prediction_anchor_penalty(
    current_pred: torch.Tensor,
    anchor_pred: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    anchor_loss = F.mse_loss(current_pred, anchor_pred.detach(), reduction="none")
    if mask is None:
        return anchor_loss.mean()
    weight = mask.to(dtype=anchor_loss.dtype)
    if float(weight.sum().item()) <= 0.0:
        return anchor_loss.new_tensor(0.0)
    return weighted_mean(anchor_loss, weight)


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


def compute_correction_ramp_scale(
    *,
    epoch: int,
    start_epoch: int,
    end_epoch: int,
) -> float:
    start_epoch = max(start_epoch, 0)
    end_epoch = max(end_epoch, 0)
    if end_epoch <= start_epoch:
        return 0.0 if epoch < start_epoch else 1.0
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
    return float(max(0.0, min(1.0, progress)))


def compute_linear_epoch_schedule(
    epoch: int,
    total_epochs: int,
    start_value: float,
    end_value: float,
) -> float:
    if total_epochs <= 1:
        return float(end_value)
    progress = max(0.0, min(1.0, (epoch - 1) / max(total_epochs - 1, 1)))
    return float(start_value + (end_value - start_value) * progress)


def compute_peer_taper_weight(
    *,
    epoch: int,
    taper_end_epoch: int,
) -> float:
    if taper_end_epoch < 0:
        return 1.0
    if epoch > taper_end_epoch:
        return 0.0
    if taper_end_epoch <= 1:
        return 1.0
    progress = max(0.0, min(1.0, (epoch - 1) / max(taper_end_epoch - 1, 1)))
    return float(max(0.0, 1.0 - progress))


def clamp_budget_ratio(value: float) -> float:
    return float(max(0.10, min(0.80, value)))


def adapt_effective_budget_ratio(
    current_ratio: float,
    *,
    observed_active_ratio: float,
    target_active_ratio: float,
    adapt_rate: float,
) -> float:
    adapt_rate = float(max(0.0, adapt_rate))
    if adapt_rate <= 0.0:
        return current_ratio
    updated = float(current_ratio) + adapt_rate * (float(target_active_ratio) - float(observed_active_ratio))
    return clamp_budget_ratio(updated)


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


def extract_recent_target_context(
    x: torch.Tensor,
    pred_len: int,
    num_targets: int,
) -> torch.Tensor:
    context = x[:, -min(pred_len, x.shape[1]) :, :num_targets]
    if context.shape[1] == pred_len:
        return context
    pad = context[:, :1, :].expand(-1, pred_len - context.shape[1], -1)
    return torch.cat([pad, context], dim=1)


def build_context_regime_feature_maps(
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if context.ndim != 3:
        raise ValueError(f"Expected context with shape [B, H, C], got {tuple(context.shape)}")
    pred_len = context.shape[1]
    base = context[:, :1, :]
    if pred_len <= 1:
        zeros = torch.zeros_like(base).expand(-1, pred_len, -1)
        return zeros, zeros, zeros

    diffs = context[:, 1:, :] - context[:, :-1, :]
    context_scale = torch.clamp(context.abs().mean(dim=1, keepdim=True), min=1e-6)
    diff_scale = torch.clamp(diffs.abs().mean(dim=1, keepdim=True), min=1e-6)
    slope = diffs.mean(dim=1, keepdim=True)
    volatility = diffs.square().mean(dim=1, keepdim=True).sqrt()
    if diffs.shape[1] > 1:
        previous_diff_mean = diffs[:, :-1, :].mean(dim=1, keepdim=True)
        last_diff = diffs[:, -1:, :]
        change_point = (last_diff - previous_diff_mean).abs()
    else:
        change_point = diffs.abs()

    slope_feature = torch.tanh(slope / torch.clamp(volatility, min=1e-6))
    volatility_feature = torch.tanh(volatility / context_scale)
    change_point_feature = torch.tanh(change_point / diff_scale)
    return (
        slope_feature.expand(-1, pred_len, -1),
        volatility_feature.expand(-1, pred_len, -1),
        change_point_feature.expand(-1, pred_len, -1),
    )


def build_correction_gate_features(
    x: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    feature_mode: str,
    use_regime_features: bool,
    decomposition_kernel: int,
) -> torch.Tensor:
    context = extract_recent_target_context(
        x,
        pred_len=student_pred.shape[1],
        num_targets=student_pred.shape[-1],
    )
    regime_features = []
    if use_regime_features:
        slope_feature, volatility_feature, change_point_feature = build_context_regime_feature_maps(context)
        regime_features = [slope_feature, volatility_feature, change_point_feature]
    if feature_mode == "basic":
        delta = teacher_pred - student_pred
        feature_tensors = [
            context,
            student_pred,
            teacher_pred,
            delta,
            delta.abs(),
            *regime_features,
        ]
        return torch.stack(feature_tensors, dim=-1)

    if feature_mode == "trend_residual":
        context_trend, context_residual = decompose_forecast_trend_residual(context, decomposition_kernel)
        student_trend, student_residual = decompose_forecast_trend_residual(student_pred, decomposition_kernel)
        teacher_trend, teacher_residual = decompose_forecast_trend_residual(teacher_pred, decomposition_kernel)
        trend_delta = teacher_trend - student_trend
        residual_delta = teacher_residual - student_residual
        feature_tensors = [
            context_trend,
            context_residual,
            context_residual.abs(),
            student_trend,
            teacher_trend,
            trend_delta,
            trend_delta.abs(),
            student_residual,
            teacher_residual,
            residual_delta,
            residual_delta.abs(),
            *regime_features,
        ]
        return torch.stack(feature_tensors, dim=-1)

    raise ValueError(f"Unsupported correction feature mode: {feature_mode}")


def resolve_correction_gate_shape(
    feature_mode: str,
    use_regime_features: bool,
) -> tuple[int, int]:
    regime_dim = 3 if use_regime_features else 0
    if feature_mode == "basic":
        return 5 + regime_dim, 1
    if feature_mode == "trend_residual":
        return 11 + regime_dim, 2
    raise ValueError(f"Unsupported correction feature mode: {feature_mode}")


def compute_corrective_branch_masks(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
    feature_mode: str,
    decomposition_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_mode == "basic":
        teacher_better_mask = (
            F.mse_loss(teacher_pred, target, reduction="none") + margin
            < F.mse_loss(student_pred, target, reduction="none")
        ).to(dtype=student_pred.dtype)
        return teacher_better_mask, teacher_better_mask

    if feature_mode == "trend_residual":
        student_trend, student_residual = decompose_forecast_trend_residual(student_pred, decomposition_kernel)
        teacher_trend, teacher_residual = decompose_forecast_trend_residual(teacher_pred, decomposition_kernel)
        target_trend, target_residual = decompose_forecast_trend_residual(target, decomposition_kernel)
        trend_teacher_better = (
            F.mse_loss(teacher_trend, target_trend, reduction="none") + margin
            < F.mse_loss(student_trend, target_trend, reduction="none")
        ).to(dtype=student_pred.dtype)
        residual_teacher_better = (
            F.mse_loss(teacher_residual, target_residual, reduction="none") + margin
            < F.mse_loss(student_residual, target_residual, reduction="none")
        ).to(dtype=student_pred.dtype)
        combined = torch.maximum(trend_teacher_better, residual_teacher_better)
        branch_targets = torch.stack([trend_teacher_better, residual_teacher_better], dim=-1)
        return combined, branch_targets

    raise ValueError(f"Unsupported correction feature mode: {feature_mode}")


def build_quantile_focus_mask(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if scores.shape != candidate_mask.shape:
        raise ValueError(
            f"Focus scores {tuple(scores.shape)} must match candidate mask {tuple(candidate_mask.shape)}"
        )
    if quantile <= 0.0:
        return candidate_mask
    flat_scores = scores.reshape(scores.shape[0], -1)
    flat_candidates = candidate_mask.reshape(candidate_mask.shape[0], -1)
    focused = torch.zeros_like(flat_candidates)
    clamped_quantile = min(max(float(quantile), 0.0), 1.0)
    for i in range(flat_scores.shape[0]):
        row_candidates = flat_candidates[i]
        if not bool(row_candidates.any().item()):
            continue
        candidate_scores = flat_scores[i][row_candidates]
        cutoff = torch.quantile(candidate_scores, clamped_quantile)
        row_focus = row_candidates & (flat_scores[i] >= cutoff)
        if not bool(row_focus.any().item()):
            top_idx = torch.argmax(flat_scores[i].masked_fill(~row_candidates, float("-inf")))
            row_focus[top_idx] = True
        focused[i] = row_focus
    return focused.reshape_as(candidate_mask)


def build_budgeted_gate_mask(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    budget_ratio: float,
) -> torch.Tensor:
    if scores.shape != candidate_mask.shape:
        raise ValueError(
            f"Budget scores {tuple(scores.shape)} must match candidate mask {tuple(candidate_mask.shape)}"
        )
    if budget_ratio <= 0.0 or budget_ratio >= 1.0:
        return candidate_mask

    flat_scores = scores.reshape(scores.shape[0], -1)
    flat_candidates = candidate_mask.reshape(candidate_mask.shape[0], -1)
    budgeted = torch.zeros_like(flat_candidates)
    clamped_ratio = min(max(float(budget_ratio), 0.0), 1.0)
    for i in range(flat_scores.shape[0]):
        row_candidates = flat_candidates[i]
        candidate_count = int(row_candidates.sum().item())
        if candidate_count == 0:
            continue
        k = max(1, min(candidate_count, math.ceil(candidate_count * clamped_ratio)))
        if k >= candidate_count:
            budgeted[i] = row_candidates
            continue
        masked_scores = flat_scores[i].masked_fill(~row_candidates, float("-inf"))
        _, topk_indices = torch.topk(masked_scores, k=k, dim=0)
        row_mask = torch.zeros_like(row_candidates)
        row_mask[topk_indices] = True
        budgeted[i] = row_mask & row_candidates
    return budgeted.reshape_as(candidate_mask)


def build_correction_focus_mask(
    x: torch.Tensor,
    reference: torch.Tensor,
    *,
    tail_start_ratio: float,
    regime_focus_quantile: float,
    student_pred: Optional[torch.Tensor] = None,
    teacher_pred: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    peer_advantage_quantile: float = 0.0,
    peer_advantage_min: float = 0.0,
    peer_advantage_smoothing_kernel: int = 1,
) -> torch.Tensor:
    focus_mask = build_tail_horizon_mask(reference, tail_start_ratio)
    if regime_focus_quantile > 0.0:
        context = extract_recent_target_context(
            x,
            pred_len=reference.shape[1],
            num_targets=reference.shape[-1],
        )
        slope_feature, volatility_feature, change_point_feature = build_context_regime_feature_maps(context)
        regime_score = volatility_feature.abs() + change_point_feature.abs() + 0.5 * slope_feature.abs()
        focus_mask = build_quantile_focus_mask(regime_score, focus_mask, regime_focus_quantile)

    if (
        (peer_advantage_quantile > 0.0 or peer_advantage_min > 0.0)
        and student_pred is not None
        and teacher_pred is not None
        and target is not None
    ):
        teacher_advantage = torch.clamp(
            F.mse_loss(student_pred.detach(), target.detach(), reduction="none")
            - F.mse_loss(teacher_pred.detach(), target.detach(), reduction="none")
            - peer_advantage_min,
            min=0.0,
        )
        teacher_advantage = smooth_time_series_scores(
            teacher_advantage,
            peer_advantage_smoothing_kernel,
        )
        advantage_mask = focus_mask & (teacher_advantage > 0)
        if bool(advantage_mask.any().item()):
            if peer_advantage_quantile > 0.0:
                focus_mask = build_quantile_focus_mask(
                    teacher_advantage,
                    advantage_mask,
                    peer_advantage_quantile,
                )
            else:
                focus_mask = advantage_mask

    return focus_mask


def compute_corrective_prediction(
    x: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    correction_gate: TimeSeriesCorrectionGate,
    *,
    guidance_scale: float,
    feature_mode: str,
    use_regime_features: bool,
    decomposition_kernel: int,
    trend_scale: float,
    residual_scale: float,
    budget_ratio: float,
    focus_mask: Optional[torch.Tensor] = None,
    horizon_router_weights: Optional[torch.Tensor] = None,
    horizon_router_bin_endpoints: Optional[list[int]] = None,
    trend_only_teaching: bool = False,
    detach_gate_inputs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_student_pred = student_pred.detach() if detach_gate_inputs else student_pred
    gate_teacher_pred = teacher_pred.detach() if detach_gate_inputs else teacher_pred
    gate_x = x.detach() if detach_gate_inputs else x
    gate_features = build_correction_gate_features(
        gate_x,
        gate_student_pred,
        gate_teacher_pred,
        feature_mode=feature_mode,
        use_regime_features=use_regime_features,
        decomposition_kernel=decomposition_kernel,
    )
    gate_logits = correction_gate(gate_features)
    focus = focus_mask.to(dtype=student_pred.dtype) if focus_mask is not None else None

    if feature_mode == "trend_residual":
        trend_gate = torch.sigmoid(gate_logits[..., 0])
        residual_gate = torch.sigmoid(gate_logits[..., 1])
        if focus is not None:
            trend_gate = trend_gate * focus
            residual_gate = residual_gate * focus
        router_scale = build_horizon_router_tensor(
            trend_gate,
            horizon_router_weights,
            horizon_router_bin_endpoints,
        )
        trend_gate = trend_gate * router_scale
        residual_gate = residual_gate * router_scale
        candidate_mask = (
            focus > 0
            if focus is not None
            else torch.ones_like(trend_gate, dtype=torch.bool)
        )
        if 0.0 < budget_ratio < 1.0:
            budget_mask = build_budgeted_gate_mask(
                torch.maximum(trend_gate, residual_gate),
                candidate_mask,
                budget_ratio,
            ).to(dtype=trend_gate.dtype)
            trend_gate = trend_gate * budget_mask
            residual_gate = residual_gate * budget_mask
        student_trend, student_residual = decompose_forecast_trend_residual(student_pred, decomposition_kernel)
        teacher_trend, teacher_residual = decompose_forecast_trend_residual(teacher_pred.detach(), decomposition_kernel)
        if trend_only_teaching:
            residual_gate = torch.zeros_like(residual_gate)
            teacher_residual = student_residual.detach()
        corrected_pred = (
            student_trend
            + trend_gate * guidance_scale * trend_scale * (teacher_trend - student_trend)
            + student_residual
            + residual_gate * guidance_scale * residual_scale * (teacher_residual - student_residual)
        )
        effective_gate = torch.maximum(trend_gate, residual_gate)
        return corrected_pred, effective_gate, gate_logits

    teacher_target = (
        build_residual_teacher_target(
            student_pred,
            teacher_pred,
            1.0,
            trend_only=True,
            trend_kernel=decomposition_kernel,
        )
        if trend_only_teaching
        else teacher_pred
    )
    delta = teacher_target - student_pred
    gate = torch.sigmoid(gate_logits)
    if focus is not None:
        gate = gate * focus
    router_scale = build_horizon_router_tensor(
        gate,
        horizon_router_weights,
        horizon_router_bin_endpoints,
    )
    gate = gate * router_scale
    candidate_mask = (
        focus > 0
        if focus is not None
        else torch.ones_like(gate, dtype=torch.bool)
    )
    if 0.0 < budget_ratio < 1.0:
        budget_mask = build_budgeted_gate_mask(
            gate,
            candidate_mask,
            budget_ratio,
        ).to(dtype=gate.dtype)
        gate = gate * budget_mask
    scaled_gate = gate * guidance_scale
    corrected_pred = student_pred + scaled_gate * (teacher_pred.detach() - student_pred)
    return corrected_pred, gate, gate_logits


def compute_delta_fusion_prediction(
    x: torch.Tensor,
    student_pred: torch.Tensor,
    peer_pred: torch.Tensor,
    correction_gate: TimeSeriesCorrectionGate,
    *,
    guidance_scale: float,
    feature_mode: str,
    use_regime_features: bool,
    decomposition_kernel: int,
    tail_start_ratio: float,
    fusion_max_scale: float,
    budget_ratio: float = 0.0,
    horizon_router_weights: Optional[torch.Tensor] = None,
    horizon_router_bin_endpoints: Optional[list[int]] = None,
    detach_gate_inputs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_student_pred = student_pred.detach() if detach_gate_inputs else student_pred
    gate_peer_pred = peer_pred.detach() if detach_gate_inputs else peer_pred
    gate_x = x.detach() if detach_gate_inputs else x
    gate_features = build_correction_gate_features(
        gate_x,
        gate_student_pred,
        gate_peer_pred,
        feature_mode=feature_mode,
        use_regime_features=use_regime_features,
        decomposition_kernel=decomposition_kernel,
    )
    gate_logits = correction_gate(gate_features)
    if gate_logits.ndim > student_pred.ndim:
        gate_logits = gate_logits[..., 0]
    gate = torch.sigmoid(gate_logits)
    focus_mask = build_tail_horizon_mask(student_pred, tail_start_ratio)
    gate = gate * focus_mask.to(dtype=gate.dtype)
    router_scale = build_horizon_router_tensor(
        gate,
        horizon_router_weights,
        horizon_router_bin_endpoints,
    )
    gate = gate * router_scale
    candidate_mask = focus_mask > 0
    if 0.0 < budget_ratio < 1.0:
        budget_mask = build_budgeted_gate_mask(
            gate,
            candidate_mask,
            budget_ratio,
        ).to(dtype=gate.dtype)
        gate = gate * budget_mask
    fusion_max_scale = max(float(fusion_max_scale), 0.0)
    gate = gate * fusion_max_scale
    scaled_gate = gate * guidance_scale
    if guidance_scale <= 0.0:
        fused_pred = student_pred
    else:
        fused_pred = peer_pred.detach() + scaled_gate * (student_pred - peer_pred.detach())
    return fused_pred, gate, gate_logits, focus_mask


def train_one_epoch(
    model,
    peer_model: Optional[torch.nn.Module],
    ema_model: Optional[torch.nn.Module],
    ema_peer_model: Optional[torch.nn.Module],
    correction_gate: Optional[TimeSeriesCorrectionGate],
    loader,
    optimizer,
    peer_optimizer: Optional[torch.optim.Optimizer],
    device,
    supervised_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    elementwise_imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_imitation: float,
    margin: float,
    ssml_handoff_end_epoch: int,
    ssml_topk_ratio: float,
    ssml_supervised_hotspot_alpha: float,
    ssml_supervised_weight_mode: str,
    ssml_topk_scope: str,
    ssml_max_selected_ratio: float,
    ssml_adaptive_dense_threshold: float,
    ssml_adaptive_dense_topk_ratio: float,
    ssml_adaptive_dense_topk_scope: str,
    ssml_adaptive_dense_max_selected_ratio: float,
    ssml_adaptive_dense_score_smoothing_kernel: int,
    ssml_adaptive_dense_window_expand_kernel: int,
    ssml_gate_score_mode: str,
    ssml_score_transform: str,
    ssml_positive_upper_quantile: float,
    ssml_score_smoothing_kernel: int,
    ssml_window_score_kernel: int,
    ssml_window_expand_kernel: int,
    ssml_tail_start_ratio: float,
    ssml_residual_beta: float,
    ssml_ema_decay: float,
    ssml_imitation_space: str,
    ssml_residual_space_kernel: int,
    ssml_conflict_aware_projection: bool,
    ssml_guidance_mode: str,
    ssml_correction_sparsity_weight: float,
    ssml_correction_threshold: float,
    ssml_correction_only: bool,
    ssml_correction_tail_start_ratio: float,
    ssml_correction_regime_focus_quantile: float,
    ssml_correction_focus_loss_alpha: float,
    ssml_correction_peer_advantage_quantile: float,
    ssml_correction_peer_advantage_min: float,
    ssml_correction_peer_advantage_smoothing_kernel: int,
    ssml_correction_budget_ratio: float,
    ssml_router_bin_endpoints: Optional[list[int]],
    ssml_student_horizon_router_weights: Optional[torch.Tensor],
    ssml_peer_horizon_router_weights: Optional[torch.Tensor],
    ssml_trend_only_teaching: bool,
    ssml_fusion_tail_start_ratio: float,
    ssml_fusion_max_scale: float,
    ssml_correction_feature_mode: str,
    ssml_correction_use_regime_features: bool,
    ssml_correction_decomposition_kernel: int,
    ssml_correction_trend_scale: float,
    ssml_correction_residual_scale: float,
    guidance_scale: float,
    correction_apply_scale: float,
    correction_freeze_student: bool,
    correction_backbone_frozen: bool,
    method: str,
    hetero_ssml_one_way: bool = False,
    ssml_student_only: bool = False,
    ssml_freeze_peer: bool = False,
    ssml_worse_only_update: bool = False,
    ssml_anchor_weight: float = 0.0,
    anchor_params: Optional[list[torch.Tensor]] = None,
    snapshot_anchor_model: Optional[torch.nn.Module] = None,
    snapshot_anchor_weight: float = 0.0,
    snapshot_anchor_mask_mode: str = "selected",
):
    method = canonicalize_method_name(method)
    peer_update_disabled = ssml_student_only or ssml_freeze_peer
    if correction_backbone_frozen:
        model.eval()
    else:
        model.train()
    if correction_gate is not None:
        correction_gate.train()
    if peer_model is not None:
        if method == "ssml" and peer_update_disabled:
            peer_model.eval()
        else:
            peer_model.train()
    dml_weight_builder = get_directional_weight_builder("dml")

    total_train_objective = 0.0
    total_supervised = 0.0
    total_imitation = 0.0
    total_mean_weight = 0.0
    total_active_ratio = 0.0
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
    total_student_error_mean = 0.0
    total_peer_error_mean = 0.0
    total_student_score_p90 = 0.0
    total_peer_score_p90 = 0.0
    total_student_worse_ratio = 0.0
    total_peer_worse_ratio = 0.0
    total_student_worse_update_ratio = 0.0
    total_peer_worse_update_ratio = 0.0
    total_student_update_ratio = 0.0
    total_peer_update_ratio = 0.0
    total_student_dense_mode_ratio = 0.0
    total_peer_dense_mode_ratio = 0.0
    total_anchor_loss = 0.0
    total_snapshot_anchor_loss = 0.0
    total_conflict_cosine = 0.0
    total_conflict_projection_applied_ratio = 0.0
    total_correction_focus_ratio = 0.0
    student_router_gain_total = None
    peer_router_gain_total = None
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if peer_optimizer is not None:
            peer_optimizer.zero_grad(set_to_none=True)
        if correction_backbone_frozen:
            with torch.no_grad():
                pred = model(x)
        else:
            pred = model(x)
        supervised_loss = supervised_loss_fn(pred, y)
        supervised_term_student = supervised_loss.mean()
        imitation_term_student = supervised_loss.new_tensor(0.0)
        mean_weight_metric = 0.0
        active_ratio_metric = 0.0
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
        student_error_mean = 0.0
        peer_error_mean = 0.0
        student_score_p90 = 0.0
        peer_score_p90 = 0.0
        student_worse_ratio = 0.0
        peer_worse_ratio = 0.0
        student_worse_update_ratio = 0.0
        peer_worse_update_ratio = 0.0
        student_update_ratio = 0.0
        peer_update_ratio = 0.0
        student_dense_mode_ratio = 0.0
        peer_dense_mode_ratio = 0.0
        anchor_loss_metric = 0.0
        snapshot_anchor_loss_metric = 0.0
        conflict_cosine_metric = 0.0
        conflict_projection_applied_ratio = 0.0
        correction_focus_ratio_metric = 1.0
        student_router_gain_metric = None
        peer_router_gain_metric = None

        if method == "independent":
            loss = supervised_term_student
            loss.backward()
            optimizer.step()

        elif method == "dml":
            if peer_model is None or peer_optimizer is None:
                raise ValueError("peer_model and peer_optimizer are required when method='dml'")
            peer_pred = peer_model(x)
            peer_supervised_loss = supervised_loss_fn(peer_pred, y)
            w_student, w_peer = dml_weight_builder(
                supervised_loss.detach(),
                peer_supervised_loss.detach(),
                margin=margin,
            )
            if lambda_imitation <= 0.0:
                w_student = torch.zeros_like(w_student)
                w_peer = torch.zeros_like(w_peer)

            imitation_term_student = weighted_mean(imitation_loss_fn(pred, peer_pred.detach()), w_student)
            loss = supervised_term_student + lambda_imitation * imitation_term_student

            imitation_term_peer = weighted_mean(imitation_loss_fn(peer_pred, pred.detach()), w_peer)
            peer_loss = peer_supervised_loss.mean() + lambda_imitation * imitation_term_peer

            (loss + peer_loss).backward()
            optimizer.step()
            peer_optimizer.step()

            mean_weight_metric = mask_activation_ratio(w_student)
            active_ratio_metric = _positive_activation_ratio(w_student)

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
            with torch.no_grad():
                teacher_pred_student = ema_peer_model(x) if ema_peer_model is not None else peer_pred.detach()
                teacher_pred_peer = ema_model(x) if ema_model is not None else pred.detach()
            snapshot_anchor_pred = None
            if snapshot_anchor_model is not None and snapshot_anchor_weight > 0.0:
                with torch.no_grad():
                    snapshot_anchor_pred = snapshot_anchor_model(x)
            sup_student_elementwise = F.mse_loss(pred, y, reduction="none")
            sup_peer_elementwise = F.mse_loss(peer_pred, y, reduction="none")
            sup_teacher_student = F.mse_loss(teacher_pred_student, y, reduction="none")
            sup_teacher_peer = F.mse_loss(teacher_pred_peer, y, reduction="none")
            peer_supervised_loss = sup_peer_elementwise.reshape(sup_peer_elementwise.shape[0], -1).mean(dim=1)
            zero = pred.new_tensor(0.0)

            if ssml_guidance_mode in {"corrective", "delta_fusion"}:
                if correction_gate is None:
                    raise ValueError(
                        "correction_gate is required when ssml_guidance_mode is corrective or delta_fusion"
                    )
                teacher_pred = peer_pred.detach()
                if ssml_guidance_mode == "corrective":
                    student_pred_for_correction = pred.detach() if correction_freeze_student else pred
                    correction_focus_mask = build_correction_focus_mask(
                        x,
                        pred.detach(),
                        tail_start_ratio=ssml_correction_tail_start_ratio,
                        regime_focus_quantile=ssml_correction_regime_focus_quantile,
                        student_pred=pred.detach(),
                        teacher_pred=teacher_pred.detach(),
                        target=y.detach(),
                        peer_advantage_quantile=ssml_correction_peer_advantage_quantile,
                        peer_advantage_min=ssml_correction_peer_advantage_min,
                        peer_advantage_smoothing_kernel=ssml_correction_peer_advantage_smoothing_kernel,
                    )
                    effective_correction_scale = guidance_scale * correction_apply_scale
                    corrected_pred, correction_gate_values, correction_gate_logits = compute_corrective_prediction(
                        x,
                        student_pred_for_correction,
                        teacher_pred,
                        correction_gate,
                        guidance_scale=effective_correction_scale,
                        feature_mode=ssml_correction_feature_mode,
                        use_regime_features=ssml_correction_use_regime_features,
                        decomposition_kernel=ssml_correction_decomposition_kernel,
                        trend_scale=ssml_correction_trend_scale,
                        residual_scale=ssml_correction_residual_scale,
                        budget_ratio=ssml_correction_budget_ratio,
                        focus_mask=correction_focus_mask,
                        horizon_router_weights=ssml_student_horizon_router_weights,
                        horizon_router_bin_endpoints=ssml_router_bin_endpoints,
                        trend_only_teaching=ssml_trend_only_teaching,
                    )
                    corrected_sup_elementwise = F.mse_loss(corrected_pred, y, reduction="none")
                    correction_focus_weights = 1.0 + ssml_correction_focus_loss_alpha * correction_focus_mask.to(
                        dtype=corrected_sup_elementwise.dtype
                    )
                    supervised_term_student = weighted_mean(corrected_sup_elementwise, correction_focus_weights)
                    teacher_better_mask, branch_teacher_better_mask = compute_corrective_branch_masks(
                        pred.detach(),
                        teacher_pred.detach(),
                        y.detach(),
                        margin=margin,
                        feature_mode=ssml_correction_feature_mode,
                        decomposition_kernel=ssml_correction_decomposition_kernel,
                    )
                    focused_teacher_better_mask = teacher_better_mask * correction_focus_mask.to(
                        dtype=teacher_better_mask.dtype
                    )
                    if branch_teacher_better_mask.ndim > correction_focus_mask.ndim:
                        branch_focus_mask = correction_focus_mask.unsqueeze(-1).expand_as(branch_teacher_better_mask)
                    else:
                        branch_focus_mask = correction_focus_mask
                    usefulness_loss = F.binary_cross_entropy_with_logits(
                        correction_gate_logits,
                        branch_teacher_better_mask * branch_focus_mask.to(dtype=branch_teacher_better_mask.dtype),
                        weight=1.0
                        + ssml_correction_focus_loss_alpha * branch_focus_mask.to(dtype=correction_gate_logits.dtype),
                    )
                    sparsity_penalty = correction_gate_values.mean()
                    imitation_term_student = usefulness_loss
                    loss = (
                        supervised_term_student
                        + lambda_imitation * imitation_term_student
                        + ssml_correction_sparsity_weight * sparsity_penalty
                    )
                    selected_mask = (correction_gate_values > ssml_correction_threshold) & correction_focus_mask
                    if snapshot_anchor_pred is not None and snapshot_anchor_weight > 0.0:
                        snapshot_anchor_mask = (
                            selected_mask
                            if snapshot_anchor_mask_mode == "selected"
                            else None
                        )
                        snapshot_anchor_penalty = compute_prediction_anchor_penalty(
                            pred,
                            snapshot_anchor_pred,
                            mask=snapshot_anchor_mask,
                        )
                        snapshot_anchor_loss_metric = float(snapshot_anchor_penalty.item())
                        loss = loss + snapshot_anchor_weight * snapshot_anchor_penalty

                    loss.backward()
                    optimizer.step()

                    mean_weight_metric = float(correction_gate_values.mean().item())
                    if effective_correction_scale <= 0.0:
                        active_mask = torch.zeros_like(correction_gate_values, dtype=torch.bool)
                    else:
                        active_mask = selected_mask
                    active_ratio_metric = mask_ratio(active_mask)
                    correction_focus_ratio_metric = mask_ratio(correction_focus_mask)
                    student_positive_ratio = mask_ratio(focused_teacher_better_mask > 0)
                    student_selected_ratio = active_ratio_metric
                    if student_positive_ratio > 0.0:
                        student_selected_of_positive_ratio = student_selected_ratio / student_positive_ratio
                    student_selected_score_mean = masked_tensor_mean(correction_gate_values, active_mask)
                    student_hotspot_error_mean = masked_tensor_mean(corrected_sup_elementwise.detach(), active_mask)
                    student_background_error_mean = masked_tensor_mean(corrected_sup_elementwise.detach(), ~active_mask)
                    student_hotspot_gap_mean = masked_tensor_mean(
                        (sup_student_elementwise.detach() - sup_peer_elementwise.detach()),
                        active_mask,
                    )
                    student_hotspot_error_share = masked_tensor_mean(
                        focused_teacher_better_mask,
                        active_mask,
                    )
                    student_error_mean = float(corrected_sup_elementwise.detach().mean().item())
                    student_score_p90 = safe_quantile(correction_gate_values, 0.9)
                    student_worse_ratio = mask_ratio(focused_teacher_better_mask > 0)
                    student_worse_update_ratio = masked_tensor_mean(
                        focused_teacher_better_mask,
                        active_mask,
                    )
                    student_update_ratio = active_ratio_metric
                    anchor_loss_metric = float(sparsity_penalty.item())
                    peer_error_mean = float(sup_peer_elementwise.detach().mean().item())
                    if ssml_router_bin_endpoints:
                        student_router_gain_metric = compute_horizon_bin_relative_gains(
                            sup_student_elementwise.detach(),
                            sup_peer_elementwise.detach(),
                            ssml_router_bin_endpoints,
                        )
                        peer_router_gain_metric = compute_horizon_bin_relative_gains(
                            sup_peer_elementwise.detach(),
                            sup_student_elementwise.detach(),
                            ssml_router_bin_endpoints,
                        )
                else:
                    fused_pred, fusion_gate_values, fusion_gate_logits, fusion_focus_mask = compute_delta_fusion_prediction(
                        x,
                        pred.detach() if correction_freeze_student else pred,
                        teacher_pred,
                        correction_gate,
                        guidance_scale=guidance_scale * correction_apply_scale,
                        feature_mode=ssml_correction_feature_mode,
                        use_regime_features=ssml_correction_use_regime_features,
                        decomposition_kernel=ssml_correction_decomposition_kernel,
                        tail_start_ratio=ssml_fusion_tail_start_ratio,
                        fusion_max_scale=ssml_fusion_max_scale,
                        budget_ratio=ssml_correction_budget_ratio,
                        horizon_router_weights=ssml_student_horizon_router_weights,
                        horizon_router_bin_endpoints=ssml_router_bin_endpoints,
                    )
                    fused_sup_elementwise = F.mse_loss(fused_pred, y, reduction="none")
                    delta_pred = pred - teacher_pred
                    delta_target = y - teacher_pred
                    delta_loss_elementwise = F.mse_loss(delta_pred, delta_target.detach(), reduction="none")
                    fusion_focus_weights = 1.0 + ssml_correction_focus_loss_alpha * fusion_focus_mask.to(
                        dtype=fused_sup_elementwise.dtype
                    )
                    supervised_term_student = weighted_mean(fused_sup_elementwise, fusion_focus_weights)
                    delta_weights = fusion_gate_values * fusion_focus_mask.to(dtype=fusion_gate_values.dtype)
                    imitation_term_student = weighted_mean(delta_loss_elementwise, delta_weights)
                    sparsity_penalty = fusion_gate_values.mean()
                    loss = (
                        supervised_term_student
                        + lambda_imitation * imitation_term_student
                        + ssml_correction_sparsity_weight * sparsity_penalty
                    )
                    selected_mask = (fusion_gate_values > ssml_correction_threshold) & fusion_focus_mask
                    if snapshot_anchor_pred is not None and snapshot_anchor_weight > 0.0:
                        snapshot_anchor_mask = (
                            selected_mask
                            if snapshot_anchor_mask_mode == "selected"
                            else None
                        )
                        snapshot_anchor_penalty = compute_prediction_anchor_penalty(
                            pred,
                            snapshot_anchor_pred,
                            mask=snapshot_anchor_mask,
                        )
                        snapshot_anchor_loss_metric = float(snapshot_anchor_penalty.item())
                        loss = loss + snapshot_anchor_weight * snapshot_anchor_penalty
                    if ssml_anchor_weight > 0.0 and anchor_params:
                        anchor_penalty = compute_anchor_penalty(model, anchor_params)
                        anchor_loss_metric = float(anchor_penalty.item())
                        loss = loss + ssml_anchor_weight * anchor_penalty
                    else:
                        anchor_loss_metric = float(sparsity_penalty.item())

                    loss.backward()
                    optimizer.step()

                    if guidance_scale * correction_apply_scale <= 0.0:
                        active_mask = torch.zeros_like(fusion_gate_values, dtype=torch.bool)
                    else:
                        active_mask = selected_mask
                    mean_weight_metric = float(fusion_gate_values.mean().item())
                    active_ratio_metric = mask_ratio(active_mask)
                    correction_focus_ratio_metric = mask_ratio(fusion_focus_mask)
                    student_positive_ratio = mask_ratio(fusion_focus_mask)
                    student_selected_ratio = active_ratio_metric
                    if student_positive_ratio > 0.0:
                        student_selected_of_positive_ratio = student_selected_ratio / student_positive_ratio
                    student_selected_score_mean = masked_tensor_mean(fusion_gate_values, active_mask)
                    student_hotspot_error_mean = masked_tensor_mean(fused_sup_elementwise.detach(), active_mask)
                    student_background_error_mean = masked_tensor_mean(fused_sup_elementwise.detach(), ~active_mask)
                    student_hotspot_gap_mean = masked_tensor_mean(
                        (sup_student_elementwise.detach() - sup_peer_elementwise.detach()),
                        active_mask,
                    )
                    student_hotspot_error_share = masked_tensor_mean(delta_loss_elementwise.detach(), active_mask)
                    student_error_mean = float(fused_sup_elementwise.detach().mean().item())
                    student_score_p90 = safe_quantile(fusion_gate_values, 0.9)
                    student_worse_ratio = mask_ratio(sup_student_elementwise.detach() > sup_peer_elementwise.detach())
                    student_worse_update_ratio = masked_tensor_mean(
                        (sup_student_elementwise.detach() > sup_peer_elementwise.detach()).to(
                            dtype=sup_student_elementwise.dtype
                        ),
                        active_mask,
                    )
                    student_update_ratio = active_ratio_metric
                    peer_error_mean = float(sup_peer_elementwise.detach().mean().item())
                    if ssml_router_bin_endpoints:
                        student_router_gain_metric = compute_horizon_bin_relative_gains(
                            fused_sup_elementwise.detach(),
                            sup_peer_elementwise.detach(),
                            ssml_router_bin_endpoints,
                        )
                        peer_router_gain_metric = compute_horizon_bin_relative_gains(
                            sup_peer_elementwise.detach(),
                            fused_sup_elementwise.detach(),
                            ssml_router_bin_endpoints,
                        )
                continue_batch = False
            else:
                continue_batch = True

            if continue_batch:
                sup_student = sup_student_elementwise.detach()
                sup_peer = sup_peer_elementwise.detach()
                score_sup_student = smooth_time_series_scores(sup_student, ssml_window_score_kernel)
                score_sup_peer = smooth_time_series_scores(sup_peer, ssml_window_score_kernel)
                score_teacher_student = smooth_time_series_scores(
                    sup_teacher_student.detach(),
                    ssml_window_score_kernel,
                )
                score_teacher_peer = smooth_time_series_scores(
                    sup_teacher_peer.detach(),
                    ssml_window_score_kernel,
                )
                error_gap_student = sup_student - sup_teacher_student.detach()
                error_gap_peer = sup_peer - sup_teacher_peer.detach()

                student_scores, _ = compute_ssml_element_scores(
                    score_sup_student,
                    score_teacher_student,
                    margin=margin,
                    score_mode=ssml_gate_score_mode,
                    score_transform=ssml_score_transform,
                )
                peer_scores, _ = compute_ssml_element_scores(
                    score_sup_peer,
                    score_teacher_peer,
                    margin=margin,
                    score_mode=ssml_gate_score_mode,
                    score_transform=ssml_score_transform,
                )
                worse_student_mask = sup_student > sup_peer
                worse_peer_mask = sup_peer > sup_student
                if ssml_worse_only_update:
                    student_scores = student_scores * worse_student_mask.to(dtype=student_scores.dtype)
                    peer_scores = peer_scores * worse_peer_mask.to(dtype=peer_scores.dtype)

                student_positive_ratio_raw = mask_ratio(student_scores > 0)
                peer_positive_ratio_raw = mask_ratio(peer_scores > 0)
                student_dense_cfg = resolve_adaptive_dense_ssml_params(
                    positive_ratio=student_positive_ratio_raw,
                    topk_ratio=ssml_topk_ratio,
                    topk_scope=ssml_topk_scope,
                    max_selected_ratio=ssml_max_selected_ratio,
                    score_smoothing_kernel=ssml_score_smoothing_kernel,
                    window_expand_kernel=ssml_window_expand_kernel,
                    adaptive_dense_threshold=ssml_adaptive_dense_threshold,
                    adaptive_dense_topk_ratio=ssml_adaptive_dense_topk_ratio,
                    adaptive_dense_topk_scope=ssml_adaptive_dense_topk_scope,
                    adaptive_dense_max_selected_ratio=ssml_adaptive_dense_max_selected_ratio,
                    adaptive_dense_score_smoothing_kernel=ssml_adaptive_dense_score_smoothing_kernel,
                    adaptive_dense_window_expand_kernel=ssml_adaptive_dense_window_expand_kernel,
                )
                peer_dense_cfg = resolve_adaptive_dense_ssml_params(
                    positive_ratio=peer_positive_ratio_raw,
                    topk_ratio=ssml_topk_ratio,
                    topk_scope=ssml_topk_scope,
                    max_selected_ratio=ssml_max_selected_ratio,
                    score_smoothing_kernel=ssml_score_smoothing_kernel,
                    window_expand_kernel=ssml_window_expand_kernel,
                    adaptive_dense_threshold=ssml_adaptive_dense_threshold,
                    adaptive_dense_topk_ratio=ssml_adaptive_dense_topk_ratio,
                    adaptive_dense_topk_scope=ssml_adaptive_dense_topk_scope,
                    adaptive_dense_max_selected_ratio=ssml_adaptive_dense_max_selected_ratio,
                    adaptive_dense_score_smoothing_kernel=ssml_adaptive_dense_score_smoothing_kernel,
                    adaptive_dense_window_expand_kernel=ssml_adaptive_dense_window_expand_kernel,
                )
                student_scores = smooth_time_series_scores(
                    student_scores,
                    int(student_dense_cfg["score_smoothing_kernel"]),
                )
                peer_scores = smooth_time_series_scores(
                    peer_scores,
                    int(peer_dense_cfg["score_smoothing_kernel"]),
                )
                if ssml_tail_start_ratio > 0.0:
                    tail_mask_student = build_tail_horizon_mask(student_scores, ssml_tail_start_ratio)
                    tail_mask_peer = build_tail_horizon_mask(peer_scores, ssml_tail_start_ratio)
                    student_scores = student_scores * tail_mask_student.to(dtype=student_scores.dtype)
                    peer_scores = peer_scores * tail_mask_peer.to(dtype=peer_scores.dtype)
                if ssml_router_bin_endpoints:
                    student_scores = student_scores * build_horizon_router_tensor(
                        student_scores,
                        ssml_student_horizon_router_weights,
                        ssml_router_bin_endpoints,
                    )
                    peer_scores = peer_scores * build_horizon_router_tensor(
                        peer_scores,
                        ssml_peer_horizon_router_weights,
                        ssml_router_bin_endpoints,
                    )

            if continue_batch:
                mask_student_imitate = build_topk_element_mask(
                    student_scores,
                    float(student_dense_cfg["topk_ratio"]),
                    scope=str(student_dense_cfg["topk_scope"]),
                    positive_upper_quantile=ssml_positive_upper_quantile,
                )
                mask_peer_imitate = build_topk_element_mask(
                    peer_scores,
                    float(peer_dense_cfg["topk_ratio"]),
                    scope=str(peer_dense_cfg["topk_scope"]),
                    positive_upper_quantile=ssml_positive_upper_quantile,
                )
                mask_student_imitate = expand_time_series_mask(
                    mask_student_imitate,
                    int(student_dense_cfg["window_expand_kernel"]),
                )
                mask_peer_imitate = expand_time_series_mask(
                    mask_peer_imitate,
                    int(peer_dense_cfg["window_expand_kernel"]),
                )
                mask_student_imitate = limit_element_mask_by_ratio(
                    student_scores,
                    mask_student_imitate,
                    float(student_dense_cfg["max_selected_ratio"]),
                )
                mask_peer_imitate = limit_element_mask_by_ratio(
                    peer_scores,
                    mask_peer_imitate,
                    float(peer_dense_cfg["max_selected_ratio"]),
                )

                if guidance_scale <= 0.0:
                    mask_student_imitate = torch.zeros_like(mask_student_imitate, dtype=torch.bool)
                    mask_peer_imitate = torch.zeros_like(mask_peer_imitate, dtype=torch.bool)
                elif lambda_imitation <= 0.0 and ssml_guidance_mode != "reweight_only":
                    mask_student_imitate = torch.zeros_like(mask_student_imitate, dtype=torch.bool)
                    mask_peer_imitate = torch.zeros_like(mask_peer_imitate, dtype=torch.bool)
                elif hetero_ssml_one_way and ssml_guidance_mode != "reweight_only":
                    student_imitates, peer_imitates = choose_one_way_imitation_from_scores(
                        student_scores,
                        peer_scores,
                    )
                    if not student_imitates:
                        mask_student_imitate = torch.zeros_like(mask_student_imitate, dtype=torch.bool)
                    if not peer_imitates:
                        mask_peer_imitate = torch.zeros_like(mask_peer_imitate, dtype=torch.bool)
                if peer_update_disabled:
                    mask_peer_imitate = torch.zeros_like(mask_peer_imitate, dtype=torch.bool)

                student_teacher_target = build_residual_teacher_target(pred, teacher_pred_student, ssml_residual_beta)
                if ssml_trend_only_teaching:
                    student_teacher_target = build_residual_teacher_target(
                        pred,
                        teacher_pred_student,
                        ssml_residual_beta,
                        trend_only=True,
                        trend_kernel=ssml_correction_decomposition_kernel,
                    )
                history_target = x[:, :, : pred.shape[-1]]
                imit_student_source = build_imitation_representation(
                    pred,
                    history_target,
                    ssml_imitation_space,
                    ssml_residual_space_kernel,
                )
                imit_student_target = build_imitation_representation(
                    student_teacher_target,
                    history_target,
                    ssml_imitation_space,
                    ssml_residual_space_kernel,
                )
                imit_student = elementwise_imitation_loss_fn(imit_student_source, imit_student_target)
                imitation_weight_student = build_elementwise_score_weights(
                    imit_student,
                    student_scores,
                    mask_student_imitate,
                )
                hotspot_weight_student = build_elementwise_hotspot_weights(
                    sup_student_elementwise,
                    student_scores,
                    mask_student_imitate,
                    ssml_supervised_hotspot_alpha * guidance_scale,
                    mode=ssml_supervised_weight_mode,
                )
                supervised_term_student = weighted_mean(sup_student_elementwise, hotspot_weight_student)
                anchor_penalty = zero
                if ssml_anchor_weight > 0.0 and anchor_params:
                    anchor_penalty = compute_anchor_penalty(model, anchor_params)
                    anchor_loss_metric = float(anchor_penalty.item())
                supervised_objective_student = supervised_term_student + ssml_anchor_weight * anchor_penalty
                if ssml_guidance_mode == "reweight_only":
                    imitation_term_student = zero
                    imitation_objective_student = zero
                    loss = supervised_objective_student
                else:
                    imitation_term_student = weighted_mean(imit_student, imitation_weight_student)
                    imitation_objective_student = lambda_imitation * imitation_term_student
                    loss = supervised_objective_student + imitation_objective_student
                if snapshot_anchor_pred is not None and snapshot_anchor_weight > 0.0:
                    snapshot_anchor_mask = (
                        mask_student_imitate
                        if snapshot_anchor_mask_mode == "selected"
                        else None
                    )
                    snapshot_anchor_penalty = compute_prediction_anchor_penalty(
                        pred,
                        snapshot_anchor_pred,
                        mask=snapshot_anchor_mask,
                    )
                    snapshot_anchor_loss_metric = float(snapshot_anchor_penalty.item())
                    loss = loss + snapshot_anchor_weight * snapshot_anchor_penalty

                hotspot_weight_peer = build_elementwise_hotspot_weights(
                    sup_peer_elementwise,
                    peer_scores,
                    mask_peer_imitate,
                    ssml_supervised_hotspot_alpha * guidance_scale,
                    mode=ssml_supervised_weight_mode,
                )
                supervised_term_peer = weighted_mean(sup_peer_elementwise, hotspot_weight_peer)
                if peer_update_disabled:
                    imitation_term_peer = zero
                    peer_loss = zero
                elif ssml_guidance_mode == "reweight_only":
                    imitation_term_peer = zero
                    peer_loss = supervised_term_peer
                else:
                    peer_teacher_target = build_residual_teacher_target(peer_pred, teacher_pred_peer, ssml_residual_beta)
                    if ssml_trend_only_teaching:
                        peer_teacher_target = build_residual_teacher_target(
                            peer_pred,
                            teacher_pred_peer,
                            ssml_residual_beta,
                            trend_only=True,
                            trend_kernel=ssml_correction_decomposition_kernel,
                        )
                    peer_history_target = x[:, :, : peer_pred.shape[-1]]
                    imit_peer_source = build_imitation_representation(
                        peer_pred,
                        peer_history_target,
                        ssml_imitation_space,
                        ssml_residual_space_kernel,
                    )
                    imit_peer_target = build_imitation_representation(
                        peer_teacher_target,
                        peer_history_target,
                        ssml_imitation_space,
                        ssml_residual_space_kernel,
                    )
                    imit_peer = elementwise_imitation_loss_fn(imit_peer_source, imit_peer_target)
                    imitation_weight_peer = build_elementwise_score_weights(
                        imit_peer,
                        peer_scores,
                        mask_peer_imitate,
                    )
                    imitation_term_peer = weighted_mean(imit_peer, imitation_weight_peer)
                    peer_loss = supervised_term_peer + lambda_imitation * imitation_term_peer

                if peer_update_disabled:
                    if (
                        ssml_conflict_aware_projection
                        and ssml_guidance_mode != "reweight_only"
                        and float(imitation_objective_student.detach().item()) > 0.0
                    ):
                        trainable_params = [param for param in model.parameters() if param.requires_grad]
                        _, conflict_cosine_metric, projection_applied = conflict_project_gradients(
                            trainable_params,
                            supervised_objective_student,
                            imitation_objective_student,
                        )
                        conflict_projection_applied_ratio = 1.0 if projection_applied else 0.0
                        optimizer.step()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    (loss + peer_loss).backward()
                    optimizer.step()
                    peer_optimizer.step()
                    update_ema_model(ema_model, model, ssml_ema_decay)
                    update_ema_model(ema_peer_model, peer_model, ssml_ema_decay)

                mean_weight_metric = float(student_scores[mask_student_imitate].mean().item()) if bool(mask_student_imitate.any().item()) else 0.0
                active_ratio_metric = mask_ratio(mask_student_imitate)
                student_positive_ratio = student_positive_ratio_raw
                peer_positive_ratio = peer_positive_ratio_raw
                student_selected_ratio = mask_ratio(mask_student_imitate)
                peer_selected_ratio = mask_ratio(mask_peer_imitate)
                if student_positive_ratio > 0.0:
                    student_selected_of_positive_ratio = student_selected_ratio / student_positive_ratio
                if peer_positive_ratio > 0.0:
                    peer_selected_of_positive_ratio = peer_selected_ratio / peer_positive_ratio
                student_selected_score_mean = masked_tensor_mean(student_scores, mask_student_imitate)
                peer_selected_score_mean = masked_tensor_mean(peer_scores, mask_peer_imitate)
                student_hotspot_error_mean = masked_tensor_mean(sup_student, mask_student_imitate)
                student_background_error_mean = masked_tensor_mean(sup_student, ~mask_student_imitate)
                peer_hotspot_error_mean = masked_tensor_mean(sup_peer, mask_peer_imitate)
                peer_background_error_mean = masked_tensor_mean(sup_peer, ~mask_peer_imitate)
                student_hotspot_gap_mean = masked_tensor_mean(error_gap_student, mask_student_imitate)
                peer_hotspot_gap_mean = masked_tensor_mean(error_gap_peer, mask_peer_imitate)
                student_error_mean = float(sup_student.mean().item())
                peer_error_mean = float(sup_peer.mean().item())
                student_score_p90 = safe_quantile(student_scores, 0.9)
                peer_score_p90 = safe_quantile(peer_scores, 0.9)
                student_worse_ratio = mask_ratio(worse_student_mask)
                peer_worse_ratio = mask_ratio(worse_peer_mask)
                student_worse_update_ratio = masked_tensor_mean(
                    worse_student_mask.to(dtype=sup_student.dtype),
                    mask_student_imitate,
                )
                peer_worse_update_ratio = masked_tensor_mean(
                    worse_peer_mask.to(dtype=sup_peer.dtype),
                    mask_peer_imitate,
                )
                student_update_ratio = mask_ratio(mask_student_imitate)
                peer_update_ratio = mask_ratio(mask_peer_imitate)
                student_total_error = float(sup_student.sum().item())
                peer_total_error = float(sup_peer.sum().item())
                student_dense_mode_ratio = 1.0 if bool(student_dense_cfg["dense_mode"]) else 0.0
                peer_dense_mode_ratio = 1.0 if bool(peer_dense_cfg["dense_mode"]) else 0.0
                if student_total_error > 0.0 and bool(mask_student_imitate.any().item()):
                    student_hotspot_error_share = float((sup_student[mask_student_imitate].sum() / sup_student.sum()).item())
                if peer_total_error > 0.0 and bool(mask_peer_imitate.any().item()):
                    peer_hotspot_error_share = float((sup_peer[mask_peer_imitate].sum() / sup_peer.sum()).item())
                if ssml_router_bin_endpoints:
                    student_router_gain_metric = compute_horizon_bin_relative_gains(
                        sup_student.detach(),
                        score_teacher_student.detach(),
                        ssml_router_bin_endpoints,
                    )
                    peer_router_gain_metric = compute_horizon_bin_relative_gains(
                        sup_peer.detach(),
                        score_teacher_peer.detach(),
                        ssml_router_bin_endpoints,
                    )

        else:
            raise ValueError(f"Unsupported method '{method}'")

        batch_size = x.size(0)
        total_train_objective += float(loss.item()) * batch_size
        total_supervised += float(supervised_term_student.item()) * batch_size
        total_imitation += float(imitation_term_student.item()) * batch_size
        total_mean_weight += mean_weight_metric * batch_size
        total_active_ratio += active_ratio_metric * batch_size
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
        total_student_error_mean += student_error_mean * batch_size
        total_peer_error_mean += peer_error_mean * batch_size
        total_student_score_p90 += student_score_p90 * batch_size
        total_peer_score_p90 += peer_score_p90 * batch_size
        total_student_worse_ratio += student_worse_ratio * batch_size
        total_peer_worse_ratio += peer_worse_ratio * batch_size
        total_student_worse_update_ratio += student_worse_update_ratio * batch_size
        total_peer_worse_update_ratio += peer_worse_update_ratio * batch_size
        total_student_update_ratio += student_update_ratio * batch_size
        total_peer_update_ratio += peer_update_ratio * batch_size
        total_student_dense_mode_ratio += student_dense_mode_ratio * batch_size
        total_peer_dense_mode_ratio += peer_dense_mode_ratio * batch_size
        total_anchor_loss += anchor_loss_metric * batch_size
        total_snapshot_anchor_loss += snapshot_anchor_loss_metric * batch_size
        total_conflict_cosine += conflict_cosine_metric * batch_size
        total_conflict_projection_applied_ratio += conflict_projection_applied_ratio * batch_size
        total_correction_focus_ratio += correction_focus_ratio_metric * batch_size
        if ssml_router_bin_endpoints and student_router_gain_metric is not None:
            if student_router_gain_total is None:
                student_router_gain_total = student_router_gain_metric.detach() * batch_size
                peer_router_gain_total = peer_router_gain_metric.detach() * batch_size if peer_router_gain_metric is not None else None
            else:
                student_router_gain_total += student_router_gain_metric.detach() * batch_size
                if peer_router_gain_metric is not None:
                    if peer_router_gain_total is None:
                        peer_router_gain_total = peer_router_gain_metric.detach() * batch_size
                    else:
                        peer_router_gain_total += peer_router_gain_metric.detach() * batch_size
        total_count += batch_size

    denom = max(total_count, 1)
    return {
        "train_total": total_train_objective / denom,
        "supervised_loss_mean": total_supervised / denom,
        "imitation_loss_mean": total_imitation / denom,
        "mean_imitation_weight": total_mean_weight / denom,
        "active_imitation_ratio": total_active_ratio / denom,
        "student_positive_score_ratio": total_student_positive_ratio / denom,
        "peer_positive_score_ratio": total_peer_positive_ratio / denom,
        "student_selected_ratio": total_student_selected_ratio / denom,
        "peer_selected_ratio": total_peer_selected_ratio / denom,
        "student_selected_of_positive_ratio": total_student_selected_of_positive_ratio / denom,
        "peer_selected_of_positive_ratio": total_peer_selected_of_positive_ratio / denom,
        "student_selected_score_mean": total_student_selected_score_mean / denom,
        "peer_selected_score_mean": total_peer_selected_score_mean / denom,
        "student_hotspot_error_mean": total_student_hotspot_error_mean / denom,
        "student_background_error_mean": total_student_background_error_mean / denom,
        "peer_hotspot_error_mean": total_peer_hotspot_error_mean / denom,
        "peer_background_error_mean": total_peer_background_error_mean / denom,
        "student_hotspot_gap_mean": total_student_hotspot_gap_mean / denom,
        "peer_hotspot_gap_mean": total_peer_hotspot_gap_mean / denom,
        "student_hotspot_error_share": total_student_hotspot_error_share / denom,
        "peer_hotspot_error_share": total_peer_hotspot_error_share / denom,
        "student_error_mean": total_student_error_mean / denom,
        "peer_error_mean": total_peer_error_mean / denom,
        "student_score_p90": total_student_score_p90 / denom,
        "peer_score_p90": total_peer_score_p90 / denom,
        "student_worse_ratio": total_student_worse_ratio / denom,
        "peer_worse_ratio": total_peer_worse_ratio / denom,
        "student_worse_update_ratio": total_student_worse_update_ratio / denom,
        "peer_worse_update_ratio": total_peer_worse_update_ratio / denom,
        "student_update_ratio": total_student_update_ratio / denom,
        "peer_update_ratio": total_peer_update_ratio / denom,
        "student_dense_mode_ratio": total_student_dense_mode_ratio / denom,
        "peer_dense_mode_ratio": total_peer_dense_mode_ratio / denom,
        "anchor_loss_mean": total_anchor_loss / denom,
        "snapshot_anchor_loss_mean": total_snapshot_anchor_loss / denom,
        "conflict_cosine": total_conflict_cosine / denom,
        "conflict_projection_applied_ratio": total_conflict_projection_applied_ratio / denom,
        "correction_focus_ratio": total_correction_focus_ratio / denom,
        "student_horizon_router_relative_gains": (
            (student_router_gain_total / denom).cpu().tolist()
            if student_router_gain_total is not None
            else []
        ),
        "peer_horizon_router_relative_gains": (
            (peer_router_gain_total / denom).cpu().tolist()
            if peer_router_gain_total is not None
            else []
        ),
    }


@torch.no_grad()
def compute_guided_evaluation_prediction(
    x: torch.Tensor,
    y: torch.Tensor,
    student_pred: torch.Tensor,
    peer_pred: torch.Tensor,
    correction_gate: Optional[TimeSeriesCorrectionGate],
    *,
    guidance_mode: str,
    guidance_scale: float,
    correction_apply_scale: float,
    correction_tail_start_ratio: float,
    correction_regime_focus_quantile: float,
    correction_peer_advantage_quantile: float,
    correction_peer_advantage_min: float,
    correction_peer_advantage_smoothing_kernel: int,
    correction_budget_ratio: float,
    horizon_router_weights: Optional[torch.Tensor],
    horizon_router_bin_endpoints: Optional[list[int]],
    trend_only_teaching: bool,
    fusion_tail_start_ratio: float,
    fusion_max_scale: float,
    correction_feature_mode: str,
    correction_use_regime_features: bool,
    correction_decomposition_kernel: int,
    correction_trend_scale: float,
    correction_residual_scale: float,
) -> torch.Tensor:
    if correction_gate is None:
        return student_pred
    if guidance_mode == "delta_fusion":
        guided_pred, _, _, _ = compute_delta_fusion_prediction(
            x,
            student_pred,
            peer_pred,
            correction_gate,
            guidance_scale=guidance_scale * correction_apply_scale,
            feature_mode=correction_feature_mode,
            use_regime_features=correction_use_regime_features,
            decomposition_kernel=correction_decomposition_kernel,
            tail_start_ratio=fusion_tail_start_ratio,
            fusion_max_scale=fusion_max_scale,
            budget_ratio=correction_budget_ratio,
            horizon_router_weights=horizon_router_weights,
            horizon_router_bin_endpoints=horizon_router_bin_endpoints,
        )
        return guided_pred

    correction_focus_mask = build_correction_focus_mask(
        x,
        student_pred.detach(),
        tail_start_ratio=correction_tail_start_ratio,
        regime_focus_quantile=correction_regime_focus_quantile,
        student_pred=student_pred.detach(),
        teacher_pred=peer_pred.detach(),
        target=y.detach(),
        peer_advantage_quantile=correction_peer_advantage_quantile,
        peer_advantage_min=correction_peer_advantage_min,
        peer_advantage_smoothing_kernel=correction_peer_advantage_smoothing_kernel,
    )
    guided_pred, _, _ = compute_corrective_prediction(
        x,
        student_pred,
        peer_pred,
        correction_gate,
        guidance_scale=guidance_scale * correction_apply_scale,
        focus_mask=correction_focus_mask,
        horizon_router_weights=horizon_router_weights,
        horizon_router_bin_endpoints=horizon_router_bin_endpoints,
        trend_only_teaching=trend_only_teaching,
        feature_mode=correction_feature_mode,
        use_regime_features=correction_use_regime_features,
        decomposition_kernel=correction_decomposition_kernel,
        trend_scale=correction_trend_scale,
        residual_scale=correction_residual_scale,
        budget_ratio=correction_budget_ratio,
    )
    return guided_pred


def evaluate(
    model,
    loader,
    device,
    *,
    peer_model: Optional[torch.nn.Module] = None,
    correction_gate: Optional[TimeSeriesCorrectionGate] = None,
    guidance_mode: str = "corrective",
    guidance_scale: float = 1.0,
    correction_apply_scale: float = 1.0,
    correction_tail_start_ratio: float = 0.0,
    correction_regime_focus_quantile: float = 0.0,
    correction_peer_advantage_quantile: float = 0.0,
    correction_peer_advantage_min: float = 0.0,
    correction_peer_advantage_smoothing_kernel: int = 1,
    correction_budget_ratio: float = 0.0,
    horizon_router_weights: Optional[torch.Tensor] = None,
    horizon_router_bin_endpoints: Optional[list[int]] = None,
    trend_only_teaching: bool = False,
    fusion_tail_start_ratio: float = 0.0,
    fusion_max_scale: float = 1.0,
    correction_feature_mode: str = "basic",
    correction_use_regime_features: bool = False,
    correction_decomposition_kernel: int = 9,
    correction_trend_scale: float = 1.0,
    correction_residual_scale: float = 1.0,
) -> dict[str, dict[str, float]]:
    model.eval()
    if correction_gate is not None:
        correction_gate.eval()
    if peer_model is not None:
        peer_model.eval()
    branch_totals: dict[str, dict[str, float]] = {
        "student": {"mse": 0.0, "mae": 0.0},
        "guided": {"mse": 0.0, "mae": 0.0},
    }
    if peer_model is not None:
        branch_totals["peer"] = {"mse": 0.0, "mae": 0.0}
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        student_pred = model(x)
        guided_pred = student_pred
        peer_pred = None
        if peer_model is not None:
            peer_pred = peer_model(x)
            guided_pred = compute_guided_evaluation_prediction(
                x,
                y,
                student_pred,
                peer_pred,
                correction_gate,
                guidance_mode=guidance_mode,
                guidance_scale=guidance_scale,
                correction_apply_scale=correction_apply_scale,
                correction_tail_start_ratio=correction_tail_start_ratio,
                correction_regime_focus_quantile=correction_regime_focus_quantile,
                correction_peer_advantage_quantile=correction_peer_advantage_quantile,
                correction_peer_advantage_min=correction_peer_advantage_min,
                correction_peer_advantage_smoothing_kernel=correction_peer_advantage_smoothing_kernel,
                correction_budget_ratio=correction_budget_ratio,
                horizon_router_weights=horizon_router_weights,
                horizon_router_bin_endpoints=horizon_router_bin_endpoints,
                trend_only_teaching=trend_only_teaching,
                fusion_tail_start_ratio=fusion_tail_start_ratio,
                fusion_max_scale=fusion_max_scale,
                correction_feature_mode=correction_feature_mode,
                correction_use_regime_features=correction_use_regime_features,
                correction_decomposition_kernel=correction_decomposition_kernel,
                correction_trend_scale=correction_trend_scale,
                correction_residual_scale=correction_residual_scale,
            )
        branch_preds = {
            "student": student_pred,
            "guided": guided_pred,
        }
        if peer_pred is not None:
            branch_preds["peer"] = peer_pred
        batch_size = x.size(0)
        for branch_name, branch_pred in branch_preds.items():
            mse = F.mse_loss(branch_pred, y)
            mae = F.l1_loss(branch_pred, y)
            branch_totals[branch_name]["mse"] += float(mse.item()) * batch_size
            branch_totals[branch_name]["mae"] += float(mae.item()) * batch_size
        total_count += batch_size
    denom = max(total_count, 1)
    return {
        branch_name: {
            "mse": totals["mse"] / denom,
            "mae": totals["mae"] / denom,
        }
        for branch_name, totals in branch_totals.items()
    }


def select_reported_eval_metrics(
    branch_metrics: dict[str, dict[str, float]],
    output_mode: str,
) -> tuple[str, float, float]:
    if output_mode == "guided":
        source = "guided"
    elif output_mode == "student":
        source = "student"
    elif output_mode == "peer":
        if "peer" not in branch_metrics:
            raise ValueError("ssml_eval_output_mode='peer' requires a peer_model")
        source = "peer"
    elif output_mode == "best_branch":
        preferred_order = {"guided": 0, "peer": 1, "student": 2}
        source = min(
            branch_metrics,
            key=lambda name: (
                branch_metrics[name]["mse"],
                preferred_order.get(name, 99),
            ),
        )
    else:
        raise ValueError(f"Unsupported ssml_eval_output_mode: {output_mode}")
    metrics = branch_metrics[source]
    return source, metrics["mse"], metrics["mae"]


def main():
    args = parse_args()
    args.method = canonicalize_method_name(args.method)
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

    peer_model_name = (args.peer_model or args.model) if uses_peer_model(args.method) else None
    pair_meta = build_pair_metadata(args.model, peer_model_name)
    model = build_time_series_model(
        model_name=args.model,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_features=int(meta["num_features"]),
        num_targets=int(meta["num_targets"]),
    ).to(device)
    corrective_mode = args.method == "ssml" and args.ssml_guidance_mode == "corrective"
    gated_guidance_mode = args.method == "ssml" and args.ssml_guidance_mode in {"corrective", "delta_fusion"}
    peer_model = None
    peer_optimizer = None
    ema_model = None
    ema_peer_model = None
    correction_gate = None
    snapshot_anchor_model = None
    if uses_peer_model(args.method):
        peer_model = build_time_series_model(
            model_name=pair_meta["peer_model"],
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_features=int(meta["num_features"]),
            num_targets=int(meta["num_targets"]),
        ).to(device)
    loaded_init_checkpoint = load_model_checkpoint(model, args.init_checkpoint, "init")
    loaded_peer_init_checkpoint = None
    if peer_model is not None:
        loaded_peer_init_checkpoint = load_model_checkpoint(peer_model, args.peer_init_checkpoint, "peer_init")
    if gated_guidance_mode:
        correction_input_dim, correction_output_dim = resolve_correction_gate_shape(
            args.ssml_correction_feature_mode,
            args.ssml_correction_use_regime_features,
        )
        if args.ssml_guidance_mode == "delta_fusion":
            correction_output_dim = 1
        correction_gate = TimeSeriesCorrectionGate(
            input_dim=correction_input_dim,
            output_dim=correction_output_dim,
            hidden_dim=args.ssml_correction_gate_hidden_dim,
            dropout=args.ssml_correction_gate_dropout,
            init_bias=args.ssml_correction_init_bias,
        ).to(device)
    correction_only = corrective_mode and args.ssml_correction_only
    ssml_student_only = args.method == "ssml" and (args.ssml_student_only or gated_guidance_mode) and uses_peer_model(args.method)
    ssml_freeze_peer = args.method == "ssml" and (args.ssml_freeze_peer or gated_guidance_mode) and uses_peer_model(args.method)
    peer_update_disabled = ssml_student_only or ssml_freeze_peer
    if correction_only:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
    if peer_model is not None and peer_update_disabled:
        for param in peer_model.parameters():
            param.requires_grad_(False)
        peer_model.eval()
    if args.method == "ssml" and args.ssml_ema_decay > 0.0 and peer_model is not None and not peer_update_disabled:
        ema_model = clone_ema_model(model)
        ema_peer_model = clone_ema_model(peer_model)
    anchor_params = None
    if args.method == "ssml" and args.ssml_anchor_weight > 0.0 and not correction_only:
        anchor_params = snapshot_trainable_parameters(model)

    optimizer_params = [] if correction_only else list(model.parameters())
    if correction_gate is not None:
        optimizer_params.extend(correction_gate.parameters())
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    if peer_model is not None and not peer_update_disabled:
        peer_optimizer = torch.optim.AdamW(peer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    supervised_loss_fn = build_regression_imitation_loss_fn("mse")
    imitation_loss_fn = build_regression_imitation_loss_fn(args.regression_imitation_loss)
    elementwise_imitation_loss_fn = build_regression_elementwise_loss_fn(args.regression_imitation_loss)

    summary_peer_model = pair_meta["peer_model"]
    run_dir = make_run_dir(
        args.output_dir,
        "time_series",
        args.dataset,
        f"{pair_meta['pair_tag']}_{args.method}_{args.regression_imitation_loss}_seed{args.seed}",
    )
    model_param_count = count_parameters(model)
    correction_gate_param_count = count_parameters(correction_gate) if correction_gate is not None else 0
    print(f"[time_series] run_dir={run_dir}")
    print(f"[time_series] params={model_param_count + correction_gate_param_count}")
    if correction_gate is not None:
        print(f"[time_series] correction_gate_params={correction_gate_param_count}")

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
    primary_val_mse_curve = []
    primary_val_mae_curve = []
    student_val_mse_curve = []
    student_val_mae_curve = []
    peer_val_mse_curve = []
    peer_val_mae_curve = []
    best_reported_val_mse = float("inf")
    best_primary_val_mse = float("inf")
    best_student_val_mse = float("inf")
    best_peer_val_mse = float("inf")
    best_epoch = None
    best_epoch1 = None
    best_student_epoch = None
    best_peer_epoch = None
    best_metric_output_source = None
    last_metric_output_source = "guided"
    first_active_epoch = None
    snapshot_anchor_enabled_epoch = None
    best_val_before_activation = float("inf")
    best_val_before_activation_epoch = None
    best_val_after_activation = float("inf")
    best_val_after_activation_epoch = None
    early_stop_enabled = args.early_stop_patience > 0
    early_stop_bad_epochs = 0
    early_stop_min_epochs = max(args.early_stop_min_epochs, 0)
    stopped_early = False
    stop_epoch = None
    stop_reason = None
    last_train_stats = {
        "train_total": 0.0,
        "supervised_loss_mean": 0.0,
        "imitation_loss_mean": 0.0,
        "mean_imitation_weight": 0.0,
        "active_imitation_ratio": 0.0,
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
        "student_error_mean": 0.0,
        "peer_error_mean": 0.0,
        "student_score_p90": 0.0,
        "peer_score_p90": 0.0,
        "student_worse_ratio": 0.0,
        "peer_worse_ratio": 0.0,
        "student_worse_update_ratio": 0.0,
        "peer_worse_update_ratio": 0.0,
        "student_update_ratio": 0.0,
        "peer_update_ratio": 0.0,
        "student_dense_mode_ratio": 0.0,
        "peer_dense_mode_ratio": 0.0,
        "anchor_loss_mean": 0.0,
        "snapshot_anchor_loss_mean": 0.0,
        "conflict_cosine": 0.0,
        "conflict_projection_applied_ratio": 0.0,
        "correction_focus_ratio": 1.0,
        "student_horizon_router_relative_gains": [],
        "peer_horizon_router_relative_gains": [],
    }
    hetero_ssml_one_way = args.hetero_ssml_one_way and pair_meta["is_heterogeneous_pair"]
    router_enabled = bool(args.ssml_router_bin_endpoints.strip()) or args.ssml_router_ema_decay > 0.0
    router_bin_endpoints = parse_horizon_router_bin_endpoints(args.ssml_router_bin_endpoints, args.pred_len) if router_enabled else []
    student_router_gain_state = torch.zeros(len(router_bin_endpoints), dtype=torch.float32) if router_enabled else None
    peer_router_gain_state = torch.zeros(len(router_bin_endpoints), dtype=torch.float32) if router_enabled else None
    student_router_weights = build_horizon_router_weights(student_router_gain_state) if router_enabled else None
    peer_router_weights = build_horizon_router_weights(peer_router_gain_state) if router_enabled else None
    adaptive_budget_enabled = (
        gated_guidance_mode
        and args.ssml_active_ratio_adapt_rate > 0.0
        and args.ssml_target_active_ratio_start >= 0.0
        and args.ssml_target_active_ratio_end >= 0.0
    )
    effective_correction_budget_ratio = float(args.ssml_correction_budget_ratio)
    if adaptive_budget_enabled:
        seed_budget = (
            effective_correction_budget_ratio
            if effective_correction_budget_ratio > 0.0
            else args.ssml_target_active_ratio_start
        )
        effective_correction_budget_ratio = clamp_budget_ratio(seed_budget)
    last_effective_budget_ratio = effective_correction_budget_ratio
    teacher_guidance_weight = 1.0 if args.method == "ssml" else 0.0
    current_target_active_ratio = None
    ssml_eval_output_mode = args.ssml_eval_output_mode if args.method == "ssml" else "guided"

    for epoch in range(1, args.epochs + 1):
        current_target_active_ratio = (
            compute_linear_epoch_schedule(
                epoch,
                args.epochs,
                args.ssml_target_active_ratio_start,
                args.ssml_target_active_ratio_end,
            )
            if adaptive_budget_enabled
            else None
        )
        teacher_guidance_weight = (
            compute_peer_taper_weight(epoch=epoch, taper_end_epoch=args.ssml_peer_taper_end_epoch)
            if args.method == "ssml"
            else 0.0
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
        correction_apply_scale = (
            compute_correction_ramp_scale(
                epoch=epoch,
                start_epoch=args.ssml_correction_ramp_start_epoch,
                end_epoch=args.ssml_correction_ramp_end_epoch,
            )
            if gated_guidance_mode
            else 1.0
        )
        effective_lambda, guidance_scale, correction_apply_scale, handoff_applied = apply_ssml_handoff(
            epoch=epoch,
            handoff_end_epoch=args.ssml_handoff_end_epoch,
            lambda_imitation=effective_lambda,
            guidance_scale=guidance_scale,
            correction_apply_scale=correction_apply_scale,
        )
        if args.method == "ssml":
            effective_lambda = effective_lambda * teacher_guidance_weight
            guidance_scale = guidance_scale * teacher_guidance_weight
        correction_freeze_student = (
            corrective_mode and epoch <= max(args.ssml_correction_freeze_student_epochs, 0)
        )
        correction_backbone_frozen = (
            corrective_mode
            and (
                args.ssml_correction_only
                or (
                    args.ssml_correction_student_train_end_epoch >= 0
                    and epoch > args.ssml_correction_student_train_end_epoch
                )
            )
        )
        epoch_student_router_weights = student_router_weights.clone() if student_router_weights is not None else None
        epoch_peer_router_weights = peer_router_weights.clone() if peer_router_weights is not None else None
        snapshot_anchor_enabled = (
            args.method == "ssml"
            and snapshot_anchor_model is not None
            and args.ssml_snapshot_anchor_weight > 0.0
            and args.ssml_snapshot_anchor_start_epoch >= 0
            and epoch >= args.ssml_snapshot_anchor_start_epoch
        )
        if snapshot_anchor_enabled and snapshot_anchor_enabled_epoch is None:
            snapshot_anchor_enabled_epoch = epoch

        train_stats = train_one_epoch(
            model,
            peer_model,
            ema_model,
            ema_peer_model,
            correction_gate,
            train_loader,
            optimizer,
            peer_optimizer,
            device,
            supervised_loss_fn=supervised_loss_fn,
            imitation_loss_fn=imitation_loss_fn,
            elementwise_imitation_loss_fn=elementwise_imitation_loss_fn,
            lambda_imitation=effective_lambda,
            margin=args.margin,
            ssml_handoff_end_epoch=args.ssml_handoff_end_epoch,
            ssml_topk_ratio=args.ssml_topk_ratio,
            ssml_supervised_hotspot_alpha=args.ssml_supervised_hotspot_alpha,
            ssml_supervised_weight_mode=args.ssml_supervised_weight_mode,
            ssml_topk_scope=args.ssml_topk_scope,
            ssml_max_selected_ratio=args.ssml_max_selected_ratio,
            ssml_adaptive_dense_threshold=args.ssml_adaptive_dense_threshold,
            ssml_adaptive_dense_topk_ratio=args.ssml_adaptive_dense_topk_ratio,
            ssml_adaptive_dense_topk_scope=args.ssml_adaptive_dense_topk_scope,
            ssml_adaptive_dense_max_selected_ratio=args.ssml_adaptive_dense_max_selected_ratio,
            ssml_adaptive_dense_score_smoothing_kernel=args.ssml_adaptive_dense_score_smoothing_kernel,
            ssml_adaptive_dense_window_expand_kernel=args.ssml_adaptive_dense_window_expand_kernel,
            ssml_gate_score_mode=args.ssml_gate_score_mode,
            ssml_score_transform=args.ssml_score_transform,
            ssml_positive_upper_quantile=args.ssml_positive_upper_quantile,
            ssml_score_smoothing_kernel=args.ssml_score_smoothing_kernel,
            ssml_window_score_kernel=args.ssml_window_score_kernel,
            ssml_window_expand_kernel=args.ssml_window_expand_kernel,
            ssml_tail_start_ratio=args.ssml_tail_start_ratio,
            ssml_residual_beta=args.ssml_residual_beta,
            ssml_ema_decay=args.ssml_ema_decay,
            ssml_imitation_space=args.ssml_imitation_space,
            ssml_residual_space_kernel=args.ssml_residual_space_kernel,
            ssml_conflict_aware_projection=args.ssml_conflict_aware_projection,
            ssml_guidance_mode=args.ssml_guidance_mode,
            ssml_correction_sparsity_weight=args.ssml_correction_sparsity_weight,
            ssml_correction_threshold=args.ssml_correction_threshold,
            ssml_correction_only=args.ssml_correction_only,
            ssml_correction_tail_start_ratio=args.ssml_correction_tail_start_ratio,
            ssml_correction_regime_focus_quantile=args.ssml_correction_regime_focus_quantile,
            ssml_correction_focus_loss_alpha=args.ssml_correction_focus_loss_alpha,
            ssml_correction_peer_advantage_quantile=args.ssml_correction_peer_advantage_quantile,
            ssml_correction_peer_advantage_min=args.ssml_correction_peer_advantage_min,
            ssml_correction_peer_advantage_smoothing_kernel=args.ssml_correction_peer_advantage_smoothing_kernel,
            ssml_correction_budget_ratio=effective_correction_budget_ratio,
            ssml_router_bin_endpoints=router_bin_endpoints,
            ssml_student_horizon_router_weights=epoch_student_router_weights,
            ssml_peer_horizon_router_weights=epoch_peer_router_weights,
            ssml_trend_only_teaching=args.ssml_trend_only_teaching,
            ssml_fusion_tail_start_ratio=args.ssml_fusion_tail_start_ratio,
            ssml_fusion_max_scale=args.ssml_fusion_max_scale,
            ssml_correction_feature_mode=args.ssml_correction_feature_mode,
            ssml_correction_use_regime_features=args.ssml_correction_use_regime_features,
            ssml_correction_decomposition_kernel=args.ssml_correction_decomposition_kernel,
            ssml_correction_trend_scale=args.ssml_correction_trend_scale,
            ssml_correction_residual_scale=args.ssml_correction_residual_scale,
            guidance_scale=guidance_scale,
            correction_apply_scale=correction_apply_scale,
            correction_freeze_student=correction_freeze_student,
            correction_backbone_frozen=correction_backbone_frozen,
            method=args.method,
            hetero_ssml_one_way=hetero_ssml_one_way,
            ssml_student_only=ssml_student_only,
            ssml_freeze_peer=ssml_freeze_peer,
            ssml_worse_only_update=args.ssml_worse_only_update,
            ssml_anchor_weight=args.ssml_anchor_weight,
            anchor_params=anchor_params,
            snapshot_anchor_model=snapshot_anchor_model if snapshot_anchor_enabled else None,
            snapshot_anchor_weight=args.ssml_snapshot_anchor_weight if snapshot_anchor_enabled else 0.0,
            snapshot_anchor_mask_mode=args.ssml_snapshot_anchor_mask_mode,
        )
        last_train_stats = train_stats
        eval_metrics = evaluate(
            model,
            val_loader,
            device,
            peer_model=peer_model,
            correction_gate=correction_gate if gated_guidance_mode else None,
            guidance_mode=args.ssml_guidance_mode,
            guidance_scale=guidance_scale,
            correction_apply_scale=correction_apply_scale,
            correction_tail_start_ratio=args.ssml_correction_tail_start_ratio,
            correction_regime_focus_quantile=args.ssml_correction_regime_focus_quantile,
            correction_peer_advantage_quantile=args.ssml_correction_peer_advantage_quantile,
            correction_peer_advantage_min=args.ssml_correction_peer_advantage_min,
            correction_peer_advantage_smoothing_kernel=args.ssml_correction_peer_advantage_smoothing_kernel,
            correction_budget_ratio=effective_correction_budget_ratio,
            horizon_router_weights=epoch_student_router_weights,
            horizon_router_bin_endpoints=router_bin_endpoints,
            trend_only_teaching=args.ssml_trend_only_teaching,
            fusion_tail_start_ratio=args.ssml_fusion_tail_start_ratio,
            fusion_max_scale=args.ssml_fusion_max_scale,
            correction_feature_mode=args.ssml_correction_feature_mode,
            correction_use_regime_features=args.ssml_correction_use_regime_features,
            correction_decomposition_kernel=args.ssml_correction_decomposition_kernel,
            correction_trend_scale=args.ssml_correction_trend_scale,
            correction_residual_scale=args.ssml_correction_residual_scale,
        )
        student_va_mse = eval_metrics["student"]["mse"]
        student_va_mae = eval_metrics["student"]["mae"]
        primary_va_mse = eval_metrics["guided"]["mse"]
        primary_va_mae = eval_metrics["guided"]["mae"]
        reported_source, va_mse, va_mae = select_reported_eval_metrics(
            eval_metrics,
            ssml_eval_output_mode,
        )
        last_metric_output_source = reported_source
        if router_enabled:
            student_router_gain_state = update_horizon_router_state(
                student_router_gain_state,
                torch.tensor(train_stats["student_horizon_router_relative_gains"], dtype=torch.float32),
                args.ssml_router_ema_decay,
            )
            peer_router_gain_state = update_horizon_router_state(
                peer_router_gain_state,
                torch.tensor(train_stats["peer_horizon_router_relative_gains"], dtype=torch.float32),
                args.ssml_router_ema_decay,
            )
            student_router_weights = build_horizon_router_weights(student_router_gain_state)
            peer_router_weights = build_horizon_router_weights(peer_router_gain_state)
        logged_effective_budget_ratio = effective_correction_budget_ratio
        last_effective_budget_ratio = logged_effective_budget_ratio
        next_effective_correction_budget_ratio = effective_correction_budget_ratio
        if adaptive_budget_enabled and guidance_scale > 0.0 and teacher_guidance_weight > 0.0:
            next_effective_correction_budget_ratio = adapt_effective_budget_ratio(
                effective_correction_budget_ratio,
                observed_active_ratio=train_stats["active_imitation_ratio"],
                target_active_ratio=current_target_active_ratio,
                adapt_rate=args.ssml_active_ratio_adapt_rate,
            )
        peer_metrics = eval_metrics.get("peer")
        peer_va_mse = peer_metrics["mse"] if peer_metrics is not None else None
        peer_va_mae = peer_metrics["mae"] if peer_metrics is not None else None
        train_total_curve.append(train_stats["train_total"])
        train_mse_curve.append(train_stats["supervised_loss_mean"])
        train_sup_curve.append(train_stats["supervised_loss_mean"])
        train_imitation_curve.append(train_stats["imitation_loss_mean"])
        train_mean_weight_curve.append(train_stats["mean_imitation_weight"])
        train_active_ratio_curve.append(train_stats["active_imitation_ratio"])
        val_mse_curve.append(va_mse)
        val_mae_curve.append(va_mae)
        primary_val_mse_curve.append(primary_va_mse)
        primary_val_mae_curve.append(primary_va_mae)
        student_val_mse_curve.append(student_va_mse)
        student_val_mae_curve.append(student_va_mae)
        primary_improved = primary_va_mse < (best_primary_val_mse - args.early_stop_min_delta)
        if primary_improved:
            best_primary_val_mse = primary_va_mse
            best_epoch1 = epoch
            early_stop_bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            if correction_gate is not None:
                torch.save(correction_gate.state_dict(), run_dir / "best_correction_gate.pt")
            if args.method == "ssml":
                snapshot_anchor_model = refresh_frozen_snapshot(snapshot_anchor_model, model)
                torch.save(model.state_dict(), run_dir / "best_snapshot_anchor_model.pt")
        elif early_stop_enabled and epoch >= early_stop_min_epochs:
            early_stop_bad_epochs += 1
        if student_va_mse < best_student_val_mse:
            best_student_val_mse = student_va_mse
            best_student_epoch = epoch
        reported_improved = va_mse < (best_reported_val_mse - args.early_stop_min_delta)
        if reported_improved:
            best_reported_val_mse = va_mse
            best_epoch = epoch
            best_metric_output_source = reported_source
        if peer_va_mse is not None and peer_va_mae is not None:
            peer_val_mse_curve.append(peer_va_mse)
            peer_val_mae_curve.append(peer_va_mae)
            if peer_va_mse < best_peer_val_mse:
                best_peer_val_mse = peer_va_mse
                best_peer_epoch = epoch
                torch.save(peer_model.state_dict(), run_dir / "best_peer_model.pt")
        effective_correction_budget_ratio = next_effective_correction_budget_ratio

        guidance_is_active = (
            args.method == "ssml"
            and guidance_scale > 0.0
            and (
                train_stats["student_update_ratio"] > 0.0
                or train_stats["peer_update_ratio"] > 0.0
            )
        )
        if first_active_epoch is None and guidance_is_active:
            first_active_epoch = epoch
        if first_active_epoch is None:
            if primary_va_mse < best_val_before_activation:
                best_val_before_activation = primary_va_mse
                best_val_before_activation_epoch = epoch
        else:
            if primary_va_mse < best_val_after_activation:
                best_val_after_activation = primary_va_mse
                best_val_after_activation_epoch = epoch

        status = (
            f"[time_series][epoch {epoch:03d}] "
            f"lambda={effective_lambda:.4f} "
            f"g_scale={guidance_scale:.3f} "
            f"teacher_w={teacher_guidance_weight:.3f} "
            f"handoff={int(handoff_applied)} "
            f"corr_scale={correction_apply_scale:.3f} "
            f"budget={logged_effective_budget_ratio:.3f} "
            f"target_active={(current_target_active_ratio if current_target_active_ratio is not None else -1.0):.3f} "
            f"freeze_student={int(correction_freeze_student)} "
            f"freeze_backbone={int(correction_backbone_frozen)} "
            f"focus={train_stats['correction_focus_ratio']:.4f} "
            f"train_total={train_stats['train_total']:.8f} "
            f"train_sup={train_stats['supervised_loss_mean']:.8f} "
            f"train_imit={train_stats['imitation_loss_mean']:.8f} "
            f"snap={train_stats['snapshot_anchor_loss_mean']:.8f} "
            f"snap_w={(args.ssml_snapshot_anchor_weight if snapshot_anchor_enabled else 0.0):.3f} "
            f"mean_w={train_stats['mean_imitation_weight']:.4f} "
            f"active={train_stats['active_imitation_ratio']:.4f} "
            f"s_dense={train_stats['student_dense_mode_ratio']:.2f} "
            f"s_pos={train_stats['student_positive_score_ratio']:.4f} "
            f"s_sel={train_stats['student_selected_ratio']:.4f} "
            f"s_sel_pos={train_stats['student_selected_of_positive_ratio']:.4f} "
            f"s_hot_err={train_stats['student_hotspot_error_mean']:.6f} "
            f"s_bg_err={train_stats['student_background_error_mean']:.6f} "
            f"s_gap={train_stats['student_hotspot_gap_mean']:.6f} "
            f"conf_cos={train_stats['conflict_cosine']:.4f} "
            f"conf_proj={train_stats['conflict_projection_applied_ratio']:.2f} "
            f"val_mse={va_mse:.8f} val_mae={va_mae:.8f} "
            f"eval={reported_source}"
        )
        if reported_source != "guided" or abs(primary_va_mse - va_mse) > 1e-12:
            status += (
                f" guided_val_mse={primary_va_mse:.8f} "
                f"student_val_mse={student_va_mse:.8f}"
            )
        if peer_va_mse is not None and peer_va_mae is not None:
            status += (
                f" | p_pos={train_stats['peer_positive_score_ratio']:.4f} "
                f"p_dense={train_stats['peer_dense_mode_ratio']:.2f} "
                f"p_sel={train_stats['peer_selected_ratio']:.4f} "
                f"p_sel_pos={train_stats['peer_selected_of_positive_ratio']:.4f} "
                f"p_hot_err={train_stats['peer_hotspot_error_mean']:.6f} "
                f"p_bg_err={train_stats['peer_background_error_mean']:.6f} "
                f"p_gap={train_stats['peer_hotspot_gap_mean']:.6f} "
                f"peer_val_mse={peer_va_mse:.8f} peer_val_mae={peer_va_mae:.8f}"
            )
        print(status)
        one_way_rule = (
            "disabled_in_reweight_only"
            if args.ssml_guidance_mode == "reweight_only"
            else "hotspot_presence_bidirectional_fallback"
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
                "hetero_ssml_one_way": hetero_ssml_one_way,
                "ssml_one_way_rule": one_way_rule,
                "ssml_student_only": ssml_student_only,
                "ssml_freeze_peer": ssml_freeze_peer,
                "ssml_worse_only_update": args.ssml_worse_only_update,
                "ssml_anchor_weight": args.ssml_anchor_weight,
                "ssml_snapshot_anchor_start_epoch": args.ssml_snapshot_anchor_start_epoch,
                "ssml_snapshot_anchor_weight": args.ssml_snapshot_anchor_weight,
                "ssml_snapshot_anchor_mask_mode": args.ssml_snapshot_anchor_mask_mode,
                "ssml_peer_taper_end_epoch": args.ssml_peer_taper_end_epoch,
                "ssml_target_active_ratio_start": args.ssml_target_active_ratio_start,
                "ssml_target_active_ratio_end": args.ssml_target_active_ratio_end,
                "ssml_active_ratio_adapt_rate": args.ssml_active_ratio_adapt_rate,
                "ssml_topk_ratio": args.ssml_topk_ratio,
                "ssml_topk_scope": args.ssml_topk_scope,
                "ssml_max_selected_ratio": args.ssml_max_selected_ratio,
                "ssml_adaptive_dense_threshold": args.ssml_adaptive_dense_threshold,
                "ssml_adaptive_dense_topk_ratio": args.ssml_adaptive_dense_topk_ratio,
                "ssml_adaptive_dense_topk_scope": args.ssml_adaptive_dense_topk_scope,
                "ssml_adaptive_dense_max_selected_ratio": args.ssml_adaptive_dense_max_selected_ratio,
                "ssml_adaptive_dense_score_smoothing_kernel": args.ssml_adaptive_dense_score_smoothing_kernel,
                "ssml_adaptive_dense_window_expand_kernel": args.ssml_adaptive_dense_window_expand_kernel,
                "ssml_supervised_hotspot_alpha": args.ssml_supervised_hotspot_alpha,
                "ssml_supervised_weight_mode": args.ssml_supervised_weight_mode,
                "ssml_gate_score_mode": args.ssml_gate_score_mode,
                "ssml_score_transform": args.ssml_score_transform,
                "ssml_positive_upper_quantile": args.ssml_positive_upper_quantile,
                "ssml_score_smoothing_kernel": args.ssml_score_smoothing_kernel,
                "ssml_window_score_kernel": args.ssml_window_score_kernel,
                "ssml_window_expand_kernel": args.ssml_window_expand_kernel,
                "ssml_tail_start_ratio": args.ssml_tail_start_ratio,
                "ssml_residual_beta": args.ssml_residual_beta,
                "ssml_ema_decay": args.ssml_ema_decay,
                "ssml_imitation_space": args.ssml_imitation_space,
                "ssml_residual_space_kernel": args.ssml_residual_space_kernel,
                "ssml_handoff_end_epoch": args.ssml_handoff_end_epoch,
                "ssml_router_bin_endpoints": router_bin_endpoints,
                "ssml_router_ema_decay": args.ssml_router_ema_decay,
                "ssml_trend_only_teaching": args.ssml_trend_only_teaching,
                "ssml_fusion_tail_start_ratio": args.ssml_fusion_tail_start_ratio,
                "ssml_fusion_max_scale": args.ssml_fusion_max_scale,
                "ssml_conflict_aware_projection": args.ssml_conflict_aware_projection,
                "ssml_guidance_mode": args.ssml_guidance_mode,
                "ssml_eval_output_mode": ssml_eval_output_mode,
                "ssml_correction_init_bias": args.ssml_correction_init_bias,
                "ssml_correction_ramp_start_epoch": args.ssml_correction_ramp_start_epoch,
                "ssml_correction_ramp_end_epoch": args.ssml_correction_ramp_end_epoch,
                "ssml_correction_freeze_student_epochs": args.ssml_correction_freeze_student_epochs,
                "ssml_correction_only": args.ssml_correction_only,
                "ssml_correction_tail_start_ratio": args.ssml_correction_tail_start_ratio,
                "ssml_correction_regime_focus_quantile": args.ssml_correction_regime_focus_quantile,
                "ssml_correction_focus_loss_alpha": args.ssml_correction_focus_loss_alpha,
                "ssml_correction_peer_advantage_quantile": args.ssml_correction_peer_advantage_quantile,
                "ssml_correction_peer_advantage_min": args.ssml_correction_peer_advantage_min,
                "ssml_correction_peer_advantage_smoothing_kernel": args.ssml_correction_peer_advantage_smoothing_kernel,
                "ssml_correction_budget_ratio": args.ssml_correction_budget_ratio,
                "ssml_correction_feature_mode": args.ssml_correction_feature_mode,
                "ssml_correction_use_regime_features": args.ssml_correction_use_regime_features,
                "ssml_correction_decomposition_kernel": args.ssml_correction_decomposition_kernel,
                "ssml_correction_trend_scale": args.ssml_correction_trend_scale,
                "ssml_correction_residual_scale": args.ssml_correction_residual_scale,
                "guidance_scale": guidance_scale,
                "teacher_guidance_weight": teacher_guidance_weight,
                "handoff_applied": handoff_applied,
                "correction_apply_scale": correction_apply_scale,
                "effective_budget_ratio": logged_effective_budget_ratio,
                "target_active_ratio": current_target_active_ratio,
                "correction_freeze_student": correction_freeze_student,
                "correction_backbone_frozen": correction_backbone_frozen,
                "train_total": train_stats["train_total"],
                "supervised_loss_mean": train_stats["supervised_loss_mean"],
                "imitation_loss_mean": train_stats["imitation_loss_mean"],
                "mean_imitation_weight": train_stats["mean_imitation_weight"],
                "active_imitation_ratio": train_stats["active_imitation_ratio"],
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
                "student_error_mean": train_stats["student_error_mean"],
                "peer_error_mean": train_stats["peer_error_mean"],
                "student_score_p90": train_stats["student_score_p90"],
                "peer_score_p90": train_stats["peer_score_p90"],
                "student_worse_ratio": train_stats["student_worse_ratio"],
                "peer_worse_ratio": train_stats["peer_worse_ratio"],
                "student_worse_update_ratio": train_stats["student_worse_update_ratio"],
                "peer_worse_update_ratio": train_stats["peer_worse_update_ratio"],
                "student_update_ratio": train_stats["student_update_ratio"],
                "peer_update_ratio": train_stats["peer_update_ratio"],
                "student_dense_mode_ratio": train_stats["student_dense_mode_ratio"],
                "peer_dense_mode_ratio": train_stats["peer_dense_mode_ratio"],
                "anchor_loss_mean": train_stats["anchor_loss_mean"],
                "snapshot_anchor_loss_mean": train_stats["snapshot_anchor_loss_mean"],
                "conflict_cosine": train_stats["conflict_cosine"],
                "conflict_projection_applied_ratio": train_stats["conflict_projection_applied_ratio"],
                "correction_focus_ratio": train_stats["correction_focus_ratio"],
                "student_horizon_router_relative_gains": train_stats["student_horizon_router_relative_gains"],
                "peer_horizon_router_relative_gains": train_stats["peer_horizon_router_relative_gains"],
                "student_horizon_router_weights": epoch_student_router_weights.tolist() if epoch_student_router_weights is not None else [],
                "peer_horizon_router_weights": epoch_peer_router_weights.tolist() if epoch_peer_router_weights is not None else [],
                "snapshot_anchor_enabled": snapshot_anchor_enabled,
                "snapshot_anchor_enabled_epoch_so_far": snapshot_anchor_enabled_epoch,
                "first_active_epoch_so_far": first_active_epoch,
                "best_val_mse_so_far": best_reported_val_mse,
                "best_epoch_so_far": best_epoch,
                "best_primary_val_mse_so_far": best_primary_val_mse,
                "best_primary_epoch_so_far": best_epoch1,
                "best_student_val_mse_so_far": best_student_val_mse,
                "best_student_epoch_so_far": best_student_epoch,
                "best_val_mse_before_activation_so_far": None if math.isinf(best_val_before_activation) else best_val_before_activation,
                "best_val_mse_before_activation_epoch_so_far": best_val_before_activation_epoch,
                "best_val_mse_after_activation_so_far": None if math.isinf(best_val_after_activation) else best_val_after_activation,
                "best_val_mse_after_activation_epoch_so_far": best_val_after_activation_epoch,
                "val_output_source": reported_source,
                "val_mse": va_mse,
                "val_mae": va_mae,
                "val_mse_reported": va_mse,
                "val_mae_reported": va_mae,
                "val_mse_guided": primary_va_mse,
                "val_mae_guided": primary_va_mae,
                "val_mse_student": student_va_mse,
                "val_mae_student": student_va_mae,
                "peer_val_mse": peer_va_mse,
                "peer_val_mae": peer_va_mae,
            },
        )

        should_stop = (
            early_stop_enabled
            and epoch >= early_stop_min_epochs
            and early_stop_bad_epochs >= args.early_stop_patience
        )
        if should_stop:
            stopped_early = True
            stop_epoch = epoch
            stop_reason = (
                f"no val_mse improvement greater than {args.early_stop_min_delta:.6g} "
                f"for {args.early_stop_patience} epoch(s)"
            )
            print(f"[time_series][epoch {epoch:03d}] early stop: {stop_reason}")

        if epoch % args.live_plot_interval == 0 or epoch == args.epochs or should_stop:
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
                val_mse_reported=val_mse_curve,
                val_mae_reported=val_mae_curve,
                train_total1=train_total_curve,
                train_mse1=train_mse_curve,
                train_sup1=train_sup_curve,
                train_imitation_loss1=train_imitation_curve,
                train_mean_imitation_weight1=train_mean_weight_curve,
                train_active_imitation_ratio1=train_active_ratio_curve,
                val_mse1=primary_val_mse_curve,
                val_mae1=primary_val_mae_curve,
                val_mse_guided=primary_val_mse_curve,
                val_mae_guided=primary_val_mae_curve,
                val_mse_student=student_val_mse_curve,
                val_mae_student=student_val_mae_curve,
                val_mse2=peer_val_mse_curve,
                val_mae2=peer_val_mae_curve,
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
        if should_stop:
            break
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
        val_mse_reported=val_mse_curve,
        val_mae_reported=val_mae_curve,
        train_total1=train_total_curve,
        train_mse1=train_mse_curve,
        train_sup1=train_sup_curve,
        train_imitation_loss1=train_imitation_curve,
        train_mean_imitation_weight1=train_mean_weight_curve,
        train_active_imitation_ratio1=train_active_ratio_curve,
        val_mse1=primary_val_mse_curve,
        val_mae1=primary_val_mae_curve,
        val_mse_guided=primary_val_mse_curve,
        val_mae_guided=primary_val_mae_curve,
        val_mse_student=student_val_mse_curve,
        val_mae_student=student_val_mae_curve,
        val_mse2=peer_val_mse_curve,
        val_mae2=peer_val_mae_curve,
    )
    if args.method == "ssml":
        before_text = "none" if math.isinf(best_val_before_activation) else f"{best_val_before_activation:.8f}@{best_val_before_activation_epoch}"
        after_text = "none" if math.isinf(best_val_after_activation) else f"{best_val_after_activation:.8f}@{best_val_after_activation_epoch}"
        print(
            "[time_series][ssml_diag] "
            f"gate_mode={args.ssml_gate_score_mode} "
            f"score_transform={args.ssml_score_transform} "
            f"window_score_k={args.ssml_window_score_kernel} "
            f"smooth_k={args.ssml_score_smoothing_kernel} "
            f"expand_k={args.ssml_window_expand_kernel} "
            f"tail_start={args.ssml_tail_start_ratio:.2f} "
            f"residual_beta={args.ssml_residual_beta:.3f} "
            f"ema_decay={args.ssml_ema_decay:.3f} "
            f"weight_mode={args.ssml_supervised_weight_mode} "
            f"freeze_peer={ssml_freeze_peer} "
            f"worse_only={args.ssml_worse_only_update} "
            f"anchor_w={args.ssml_anchor_weight:.5f} "
            f"snapshot_anchor_start={args.ssml_snapshot_anchor_start_epoch} "
            f"snapshot_anchor_w={args.ssml_snapshot_anchor_weight:.5f} "
            f"snapshot_anchor_mask={args.ssml_snapshot_anchor_mask_mode} "
            f"peer_taper_end={args.ssml_peer_taper_end_epoch} "
            f"target_active={args.ssml_target_active_ratio_start:.3f}->{args.ssml_target_active_ratio_end:.3f} "
            f"adapt_rate={args.ssml_active_ratio_adapt_rate:.3f} "
            f"imit_space={args.ssml_imitation_space} "
            f"residual_space_k={args.ssml_residual_space_kernel} "
            f"conflict_proj={int(args.ssml_conflict_aware_projection)} "
            f"topk_scope={args.ssml_topk_scope} "
            f"max_sel={args.ssml_max_selected_ratio:.3f} "
            f"adaptive_dense_thr={args.ssml_adaptive_dense_threshold:.3f} "
            f"positive_upper_q={args.ssml_positive_upper_quantile:.3f} "
            f"guidance_schedule=warmup_then_decay "
            f"handoff_end={args.ssml_handoff_end_epoch} "
            f"correction_init_bias={args.ssml_correction_init_bias:.3f} "
            f"correction_ramp={args.ssml_correction_ramp_start_epoch}->{args.ssml_correction_ramp_end_epoch} "
            f"correction_freeze_student_epochs={args.ssml_correction_freeze_student_epochs} "
            f"correction_student_train_end_epoch={args.ssml_correction_student_train_end_epoch} "
            f"correction_only={int(args.ssml_correction_only)} "
            f"correction_tail_start={args.ssml_correction_tail_start_ratio:.2f} "
            f"correction_regime_q={args.ssml_correction_regime_focus_quantile:.2f} "
            f"correction_focus_alpha={args.ssml_correction_focus_loss_alpha:.2f} "
            f"correction_peer_adv_q={args.ssml_correction_peer_advantage_quantile:.2f} "
            f"correction_peer_adv_min={args.ssml_correction_peer_advantage_min:.4f} "
            f"correction_peer_adv_k={args.ssml_correction_peer_advantage_smoothing_kernel} "
            f"correction_budget_ratio={last_effective_budget_ratio:.3f} "
            f"router_bins={router_bin_endpoints} "
            f"router_ema={args.ssml_router_ema_decay:.3f} "
            f"trend_only={int(args.ssml_trend_only_teaching)} "
            f"fusion_tail={args.ssml_fusion_tail_start_ratio:.2f} "
            f"fusion_max_scale={args.ssml_fusion_max_scale:.2f} "
            f"correction_feature_mode={args.ssml_correction_feature_mode} "
            f"correction_regime={int(args.ssml_correction_use_regime_features)} "
            f"correction_decomp_k={args.ssml_correction_decomposition_kernel} "
            f"correction_trend_scale={args.ssml_correction_trend_scale:.3f} "
            f"correction_residual_scale={args.ssml_correction_residual_scale:.3f} "
            f"eval_output={ssml_eval_output_mode} "
            f"first_active_epoch={first_active_epoch} "
            f"best_before_active={before_text} "
            f"best_after_active={after_text}"
        )
    summary = {
        "task": "time_series",
        "dataset": args.dataset,
        "method": args.method,
        "model": args.model,
        "peer_model": summary_peer_model,
        "pair_tag": pair_meta["pair_tag"],
        "pair_type": pair_meta["pair_type"],
        "is_joint_training": pair_meta["is_joint_training"],
        "is_heterogeneous_pair": pair_meta["is_heterogeneous_pair"],
        "mean_imitation_weight": last_train_stats["mean_imitation_weight"],
        "active_imitation_ratio": last_train_stats["active_imitation_ratio"],
        "supervised_loss_mean": last_train_stats["supervised_loss_mean"],
        "imitation_loss_mean": last_train_stats["imitation_loss_mean"],
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
        "student_error_mean": last_train_stats["student_error_mean"],
        "peer_error_mean": last_train_stats["peer_error_mean"],
        "student_score_p90": last_train_stats["student_score_p90"],
        "peer_score_p90": last_train_stats["peer_score_p90"],
        "student_worse_ratio": last_train_stats["student_worse_ratio"],
        "peer_worse_ratio": last_train_stats["peer_worse_ratio"],
        "student_worse_update_ratio": last_train_stats["student_worse_update_ratio"],
        "peer_worse_update_ratio": last_train_stats["peer_worse_update_ratio"],
        "student_update_ratio": last_train_stats["student_update_ratio"],
        "peer_update_ratio": last_train_stats["peer_update_ratio"],
        "student_dense_mode_ratio": last_train_stats["student_dense_mode_ratio"],
        "peer_dense_mode_ratio": last_train_stats["peer_dense_mode_ratio"],
        "anchor_loss_mean": last_train_stats["anchor_loss_mean"],
        "snapshot_anchor_loss_mean": last_train_stats["snapshot_anchor_loss_mean"],
        "conflict_cosine": last_train_stats["conflict_cosine"],
        "conflict_projection_applied_ratio": last_train_stats["conflict_projection_applied_ratio"],
        "correction_focus_ratio": last_train_stats["correction_focus_ratio"],
        "curve_mode": "pair" if peer_model is not None else "single",
        "model_idx": 1,
        "model1": args.model,
        "model2": summary_peer_model,
        "regression_imitation_loss": args.regression_imitation_loss,
        "lambda_imitation": args.lambda_imitation,
        "margin": args.margin,
        "warmup_epochs": args.warmup_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_epochs": args.early_stop_min_epochs,
        "early_stop_min_delta": args.early_stop_min_delta,
        "epochs_completed": len(val_mse_curve),
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "teacher_guidance_weight": teacher_guidance_weight,
        "effective_budget_ratio": last_effective_budget_ratio,
        "target_active_ratio": current_target_active_ratio,
        "imitation_decay_start_epoch": args.imitation_decay_start_epoch,
        "imitation_decay_end_epoch": args.imitation_decay_end_epoch,
        "imitation_decay_min_scale": args.imitation_decay_min_scale,
        "ssml_handoff_end_epoch": args.ssml_handoff_end_epoch,
        "hetero_ssml_one_way": hetero_ssml_one_way,
        "ssml_one_way_rule": (
            "disabled_in_reweight_only"
            if args.ssml_guidance_mode == "reweight_only"
            else "hotspot_presence_bidirectional_fallback"
        ),
        "ssml_student_only": ssml_student_only,
        "ssml_freeze_peer": ssml_freeze_peer,
        "ssml_worse_only_update": args.ssml_worse_only_update,
        "ssml_anchor_weight": args.ssml_anchor_weight,
        "ssml_snapshot_anchor_start_epoch": args.ssml_snapshot_anchor_start_epoch,
        "ssml_snapshot_anchor_weight": args.ssml_snapshot_anchor_weight,
        "ssml_snapshot_anchor_mask_mode": args.ssml_snapshot_anchor_mask_mode,
        "ssml_peer_taper_end_epoch": args.ssml_peer_taper_end_epoch,
        "ssml_target_active_ratio_start": args.ssml_target_active_ratio_start,
        "ssml_target_active_ratio_end": args.ssml_target_active_ratio_end,
        "ssml_active_ratio_adapt_rate": args.ssml_active_ratio_adapt_rate,
        "ssml_topk_ratio": args.ssml_topk_ratio,
        "ssml_topk_scope": args.ssml_topk_scope,
        "ssml_max_selected_ratio": args.ssml_max_selected_ratio,
        "ssml_adaptive_dense_threshold": args.ssml_adaptive_dense_threshold,
        "ssml_adaptive_dense_topk_ratio": args.ssml_adaptive_dense_topk_ratio,
        "ssml_adaptive_dense_topk_scope": args.ssml_adaptive_dense_topk_scope,
        "ssml_adaptive_dense_max_selected_ratio": args.ssml_adaptive_dense_max_selected_ratio,
        "ssml_adaptive_dense_score_smoothing_kernel": args.ssml_adaptive_dense_score_smoothing_kernel,
        "ssml_adaptive_dense_window_expand_kernel": args.ssml_adaptive_dense_window_expand_kernel,
        "ssml_supervised_hotspot_alpha": args.ssml_supervised_hotspot_alpha,
        "ssml_supervised_weight_mode": args.ssml_supervised_weight_mode,
        "ssml_gate_score_mode": args.ssml_gate_score_mode,
        "ssml_score_transform": args.ssml_score_transform,
        "ssml_positive_upper_quantile": args.ssml_positive_upper_quantile,
        "ssml_score_smoothing_kernel": args.ssml_score_smoothing_kernel,
        "ssml_window_score_kernel": args.ssml_window_score_kernel,
        "ssml_window_expand_kernel": args.ssml_window_expand_kernel,
        "ssml_tail_start_ratio": args.ssml_tail_start_ratio,
        "ssml_residual_beta": args.ssml_residual_beta,
        "ssml_ema_decay": args.ssml_ema_decay,
        "ssml_imitation_space": args.ssml_imitation_space,
        "ssml_residual_space_kernel": args.ssml_residual_space_kernel,
        "ssml_router_bin_endpoints": router_bin_endpoints,
        "ssml_router_ema_decay": args.ssml_router_ema_decay,
        "ssml_trend_only_teaching": args.ssml_trend_only_teaching,
        "ssml_fusion_tail_start_ratio": args.ssml_fusion_tail_start_ratio,
        "ssml_fusion_max_scale": args.ssml_fusion_max_scale,
        "ssml_conflict_aware_projection": args.ssml_conflict_aware_projection,
        "ssml_guidance_mode": args.ssml_guidance_mode,
        "ssml_eval_output_mode": ssml_eval_output_mode,
        "ssml_correction_gate_hidden_dim": args.ssml_correction_gate_hidden_dim,
        "ssml_correction_gate_dropout": args.ssml_correction_gate_dropout,
        "ssml_correction_init_bias": args.ssml_correction_init_bias,
        "ssml_correction_sparsity_weight": args.ssml_correction_sparsity_weight,
        "ssml_correction_threshold": args.ssml_correction_threshold,
        "ssml_correction_ramp_start_epoch": args.ssml_correction_ramp_start_epoch,
        "ssml_correction_ramp_end_epoch": args.ssml_correction_ramp_end_epoch,
        "ssml_correction_freeze_student_epochs": args.ssml_correction_freeze_student_epochs,
        "ssml_correction_student_train_end_epoch": args.ssml_correction_student_train_end_epoch,
        "ssml_correction_only": args.ssml_correction_only,
        "ssml_correction_tail_start_ratio": args.ssml_correction_tail_start_ratio,
        "ssml_correction_regime_focus_quantile": args.ssml_correction_regime_focus_quantile,
        "ssml_correction_focus_loss_alpha": args.ssml_correction_focus_loss_alpha,
        "ssml_correction_peer_advantage_quantile": args.ssml_correction_peer_advantage_quantile,
        "ssml_correction_peer_advantage_min": args.ssml_correction_peer_advantage_min,
        "ssml_correction_peer_advantage_smoothing_kernel": args.ssml_correction_peer_advantage_smoothing_kernel,
        "ssml_correction_budget_ratio": args.ssml_correction_budget_ratio,
        "ssml_correction_feature_mode": args.ssml_correction_feature_mode,
        "ssml_correction_use_regime_features": args.ssml_correction_use_regime_features,
        "ssml_correction_decomposition_kernel": args.ssml_correction_decomposition_kernel,
        "ssml_correction_trend_scale": args.ssml_correction_trend_scale,
        "ssml_correction_residual_scale": args.ssml_correction_residual_scale,
        "ssml_gate_rule": f"{args.ssml_gate_score_mode}_windowwise_topk",
        "ssml_supervised_rule": (
            "binary_hotspot_reweighting"
            if args.ssml_supervised_weight_mode == "binary"
            else "score_weighted_hotspot_reweighting"
        ),
        "ssml_directionality": (
            "frozen_backbone_sparse_correction_gate"
            if correction_only
            else "peer_base_delta_fusion_with_frozen_teacher"
            if args.ssml_guidance_mode == "delta_fusion"
            else "late_handoff_backbone_then_correction_gate"
            if corrective_mode and args.ssml_correction_student_train_end_epoch >= 0
            else "primary_model_with_frozen_teacher_correction_gate"
            if corrective_mode
            else
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
        "init_checkpoint": loaded_init_checkpoint,
        "peer_init_checkpoint": loaded_peer_init_checkpoint,
        "epoch_log_path": str(epoch_log_path),
        "best_val_mse": best_reported_val_mse,
        "best_epoch": best_epoch,
        "best_metric_output_source": best_metric_output_source,
        "final_metric_output_source": last_metric_output_source,
        "reported_model_idx": 2 if best_metric_output_source == "peer" else 1,
        "first_active_epoch": first_active_epoch,
        "snapshot_anchor_enabled_epoch": snapshot_anchor_enabled_epoch,
        "final_correction_backbone_frozen": correction_backbone_frozen,
        "best_model_path": str(run_dir / "best_model.pt"),
        "best_correction_gate_path": str(run_dir / "best_correction_gate.pt") if correction_gate is not None else None,
        "correction_gate_path": str(run_dir / "correction_gate.pt") if correction_gate is not None else None,
        "best_reported_model_path": (
            str(run_dir / "best_peer_model.pt")
            if best_metric_output_source == "peer"
            else str(run_dir / "best_model.pt")
        ),
        "best_snapshot_anchor_model_path": str(run_dir / "best_snapshot_anchor_model.pt"),
        "best_val_mse_before_activation": None if math.isinf(best_val_before_activation) else best_val_before_activation,
        "best_val_mse_before_activation_epoch": best_val_before_activation_epoch,
        "best_val_mse_after_activation": None if math.isinf(best_val_after_activation) else best_val_after_activation,
        "best_val_mse_after_activation_epoch": best_val_after_activation_epoch,
        "final_val_mse": val_mse_curve[-1],
        "final_val_mae": val_mae_curve[-1],
        "best_final_gap": val_mse_curve[-1] - best_reported_val_mse,
        "best_metric": best_reported_val_mse,
        "best_metric_key": "mse",
        "final_metric": val_mse_curve[-1],
        "best_metric1": best_primary_val_mse,
        "best_epoch1": best_epoch1,
        "final_metric1": primary_val_mse_curve[-1],
        "best_val_mse1": best_primary_val_mse,
        "final_val_mse1": primary_val_mse_curve[-1],
        "final_val_mae1": primary_val_mae_curve[-1],
        "final_val1": primary_val_mse_curve[-1],
        "best_metric_student": best_student_val_mse,
        "best_epoch_student": best_student_epoch,
        "final_metric_student": student_val_mse_curve[-1],
        "final_val_mae_student": student_val_mae_curve[-1],
        "best_metric_guided": best_primary_val_mse,
        "best_epoch_guided": best_epoch1,
        "final_metric_guided": primary_val_mse_curve[-1],
        "final_val_mae_guided": primary_val_mae_curve[-1],
        "num_parameters": model_param_count + correction_gate_param_count,
        "num_parameters1": model_param_count + correction_gate_param_count,
        "num_parameters_correction_gate": correction_gate_param_count,
        "num_parameters_reported": (
            count_parameters(peer_model)
            if best_metric_output_source == "peer" and peer_model is not None
            else model_param_count + correction_gate_param_count
        ),
        "student_horizon_router_relative_gains": last_train_stats["student_horizon_router_relative_gains"],
        "peer_horizon_router_relative_gains": last_train_stats["peer_horizon_router_relative_gains"],
        "student_horizon_router_weights": student_router_weights.tolist() if student_router_weights is not None else [],
        "peer_horizon_router_weights": peer_router_weights.tolist() if peer_router_weights is not None else [],
        "meta": meta,
    }
    if peer_model is not None:
        summary.update(
            {
                "best_metric2": best_peer_val_mse,
                "best_epoch2": best_peer_epoch,
                "best_peer_model_path": str(run_dir / "best_peer_model.pt"),
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
    if correction_gate is not None:
        torch.save(correction_gate.state_dict(), run_dir / "correction_gate.pt")
    print("[time_series] done")


if __name__ == "__main__":
    main()
