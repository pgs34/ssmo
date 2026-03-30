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
    p.add_argument("--hetero-ssml-one-way", action="store_true")
    p.add_argument("--ssml-student-only", action="store_true")
    p.add_argument("--ssml-freeze-peer", action="store_true")
    p.add_argument("--ssml-worse-only-update", action="store_true")
    p.add_argument("--ssml-anchor-weight", type=float, default=0.0)
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
        choices=["hybrid", "reweight_only"],
    )
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--peer-init-checkpoint", type=str, default=None)
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
) -> torch.Tensor:
    base = student_pred.detach()
    teacher = teacher_pred.detach()
    beta = float(max(0.0, min(1.0, beta)))
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


def build_forecast_residual_representation(
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
    trend = trend.reshape(batch, *trailing_shape, horizon).permute(0, forecast.ndim - 1, *range(1, forecast.ndim - 1))
    return forecast - trend


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
    ema_model: Optional[torch.nn.Module],
    ema_peer_model: Optional[torch.nn.Module],
    loader,
    optimizer,
    peer_optimizer: Optional[torch.optim.Optimizer],
    device,
    supervised_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    elementwise_imitation_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_imitation: float,
    margin: float,
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
    guidance_scale: float,
    method: str,
    hetero_ssml_one_way: bool = False,
    ssml_student_only: bool = False,
    ssml_freeze_peer: bool = False,
    ssml_worse_only_update: bool = False,
    ssml_anchor_weight: float = 0.0,
    anchor_params: Optional[list[torch.Tensor]] = None,
):
    method = canonicalize_method_name(method)
    peer_update_disabled = ssml_student_only or ssml_freeze_peer
    model.train()
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
    total_conflict_cosine = 0.0
    total_conflict_projection_applied_ratio = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if peer_optimizer is not None:
            peer_optimizer.zero_grad(set_to_none=True)
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
        conflict_cosine_metric = 0.0
        conflict_projection_applied_ratio = 0.0

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
            sup_student_elementwise = F.mse_loss(pred, y, reduction="none")
            sup_peer_elementwise = F.mse_loss(peer_pred, y, reduction="none")
            sup_teacher_student = F.mse_loss(teacher_pred_student, y, reduction="none")
            sup_teacher_peer = F.mse_loss(teacher_pred_peer, y, reduction="none")
            peer_supervised_loss = sup_peer_elementwise.reshape(sup_peer_elementwise.shape[0], -1).mean(dim=1)
            zero = pred.new_tensor(0.0)

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
        total_conflict_cosine += conflict_cosine_metric * batch_size
        total_conflict_projection_applied_ratio += conflict_projection_applied_ratio * batch_size
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
        "conflict_cosine": total_conflict_cosine / denom,
        "conflict_projection_applied_ratio": total_conflict_projection_applied_ratio / denom,
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
    peer_model = None
    peer_optimizer = None
    ema_model = None
    ema_peer_model = None
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
    ssml_student_only = args.method == "ssml" and args.ssml_student_only and uses_peer_model(args.method)
    ssml_freeze_peer = args.method == "ssml" and args.ssml_freeze_peer and uses_peer_model(args.method)
    peer_update_disabled = ssml_student_only or ssml_freeze_peer
    if peer_model is not None and peer_update_disabled:
        for param in peer_model.parameters():
            param.requires_grad_(False)
        peer_model.eval()
    if args.method == "ssml" and args.ssml_ema_decay > 0.0 and peer_model is not None and not peer_update_disabled:
        ema_model = clone_ema_model(model)
        ema_peer_model = clone_ema_model(peer_model)
    anchor_params = None
    if args.method == "ssml" and args.ssml_anchor_weight > 0.0:
        anchor_params = snapshot_trainable_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    peer_val_mse_curve = []
    peer_val_mae_curve = []
    best_val_mse = float("inf")
    best_peer_val_mse = float("inf")
    best_epoch = None
    best_peer_epoch = None
    first_active_epoch = None
    best_val_before_activation = float("inf")
    best_val_before_activation_epoch = None
    best_val_after_activation = float("inf")
    best_val_after_activation_epoch = None
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
        "conflict_cosine": 0.0,
        "conflict_projection_applied_ratio": 0.0,
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
            ema_model,
            ema_peer_model,
            train_loader,
            optimizer,
            peer_optimizer,
            device,
            supervised_loss_fn=supervised_loss_fn,
            imitation_loss_fn=imitation_loss_fn,
            elementwise_imitation_loss_fn=elementwise_imitation_loss_fn,
            lambda_imitation=effective_lambda,
            margin=args.margin,
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
            guidance_scale=guidance_scale,
            method=args.method,
            hetero_ssml_one_way=hetero_ssml_one_way,
            ssml_student_only=ssml_student_only,
            ssml_freeze_peer=ssml_freeze_peer,
            ssml_worse_only_update=args.ssml_worse_only_update,
            ssml_anchor_weight=args.ssml_anchor_weight,
            anchor_params=anchor_params,
        )
        last_train_stats = train_stats
        va_mse, va_mae = evaluate(model, val_loader, device)
        peer_va_mse = None
        peer_va_mae = None
        if peer_model is not None:
            peer_va_mse, peer_va_mae = evaluate(peer_model, val_loader, device)
        train_total_curve.append(train_stats["train_total"])
        train_mse_curve.append(train_stats["supervised_loss_mean"])
        train_sup_curve.append(train_stats["supervised_loss_mean"])
        train_imitation_curve.append(train_stats["imitation_loss_mean"])
        train_mean_weight_curve.append(train_stats["mean_imitation_weight"])
        train_active_ratio_curve.append(train_stats["active_imitation_ratio"])
        val_mse_curve.append(va_mse)
        val_mae_curve.append(va_mae)
        if va_mse < best_val_mse:
            best_val_mse = va_mse
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        if peer_va_mse is not None and peer_va_mae is not None:
            peer_val_mse_curve.append(peer_va_mse)
            peer_val_mae_curve.append(peer_va_mae)
            if peer_va_mse < best_peer_val_mse:
                best_peer_val_mse = peer_va_mse
                best_peer_epoch = epoch
                torch.save(peer_model.state_dict(), run_dir / "best_peer_model.pt")

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
            if va_mse < best_val_before_activation:
                best_val_before_activation = va_mse
                best_val_before_activation_epoch = epoch
        else:
            if va_mse < best_val_after_activation:
                best_val_after_activation = va_mse
                best_val_after_activation_epoch = epoch

        status = (
            f"[time_series][epoch {epoch:03d}] "
            f"lambda={effective_lambda:.4f} "
            f"g_scale={guidance_scale:.3f} "
            f"train_total={train_stats['train_total']:.8f} "
            f"train_sup={train_stats['supervised_loss_mean']:.8f} "
            f"train_imit={train_stats['imitation_loss_mean']:.8f} "
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
            f"val_mse={va_mse:.8f} val_mae={va_mae:.8f}"
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
                "ssml_conflict_aware_projection": args.ssml_conflict_aware_projection,
                "ssml_guidance_mode": args.ssml_guidance_mode,
                "guidance_scale": guidance_scale,
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
                "conflict_cosine": train_stats["conflict_cosine"],
                "conflict_projection_applied_ratio": train_stats["conflict_projection_applied_ratio"],
                "first_active_epoch_so_far": first_active_epoch,
                "best_val_mse_so_far": best_val_mse,
                "best_epoch_so_far": best_epoch,
                "best_val_mse_before_activation_so_far": None if math.isinf(best_val_before_activation) else best_val_before_activation,
                "best_val_mse_before_activation_epoch_so_far": best_val_before_activation_epoch,
                "best_val_mse_after_activation_so_far": None if math.isinf(best_val_after_activation) else best_val_after_activation,
                "best_val_mse_after_activation_epoch_so_far": best_val_after_activation_epoch,
                "val_mse": va_mse,
                "val_mae": va_mae,
                "peer_val_mse": peer_va_mse,
                "peer_val_mae": peer_va_mae,
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
                train_total1=train_total_curve,
                train_mse1=train_mse_curve,
                train_sup1=train_sup_curve,
                train_imitation_loss1=train_imitation_curve,
                train_mean_imitation_weight1=train_mean_weight_curve,
                train_active_imitation_ratio1=train_active_ratio_curve,
                val_mse1=val_mse_curve,
                val_mae1=val_mae_curve,
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
        train_total1=train_total_curve,
        train_mse1=train_mse_curve,
        train_sup1=train_sup_curve,
        train_imitation_loss1=train_imitation_curve,
        train_mean_imitation_weight1=train_mean_weight_curve,
        train_active_imitation_ratio1=train_active_ratio_curve,
        val_mse1=val_mse_curve,
        val_mae1=val_mae_curve,
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
            f"imit_space={args.ssml_imitation_space} "
            f"residual_space_k={args.ssml_residual_space_kernel} "
            f"conflict_proj={int(args.ssml_conflict_aware_projection)} "
            f"topk_scope={args.ssml_topk_scope} "
            f"max_sel={args.ssml_max_selected_ratio:.3f} "
            f"adaptive_dense_thr={args.ssml_adaptive_dense_threshold:.3f} "
            f"positive_upper_q={args.ssml_positive_upper_quantile:.3f} "
            f"guidance_schedule=warmup_then_decay "
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
        "conflict_cosine": last_train_stats["conflict_cosine"],
        "conflict_projection_applied_ratio": last_train_stats["conflict_projection_applied_ratio"],
        "curve_mode": "pair" if peer_model is not None else "single",
        "model_idx": 1,
        "model1": args.model,
        "model2": summary_peer_model,
        "regression_imitation_loss": args.regression_imitation_loss,
        "lambda_imitation": args.lambda_imitation,
        "margin": args.margin,
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
        "ssml_freeze_peer": ssml_freeze_peer,
        "ssml_worse_only_update": args.ssml_worse_only_update,
        "ssml_anchor_weight": args.ssml_anchor_weight,
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
        "ssml_conflict_aware_projection": args.ssml_conflict_aware_projection,
        "ssml_guidance_mode": args.ssml_guidance_mode,
        "ssml_gate_rule": f"{args.ssml_gate_score_mode}_windowwise_topk",
        "ssml_supervised_rule": (
            "binary_hotspot_reweighting"
            if args.ssml_supervised_weight_mode == "binary"
            else "score_weighted_hotspot_reweighting"
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
        "init_checkpoint": loaded_init_checkpoint,
        "peer_init_checkpoint": loaded_peer_init_checkpoint,
        "epoch_log_path": str(epoch_log_path),
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "first_active_epoch": first_active_epoch,
        "best_model_path": str(run_dir / "best_model.pt"),
        "best_val_mse_before_activation": None if math.isinf(best_val_before_activation) else best_val_before_activation,
        "best_val_mse_before_activation_epoch": best_val_before_activation_epoch,
        "best_val_mse_after_activation": None if math.isinf(best_val_after_activation) else best_val_after_activation,
        "best_val_mse_after_activation_epoch": best_val_after_activation_epoch,
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
    print("[time_series] done")


if __name__ == "__main__":
    main()
