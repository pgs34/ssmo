from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT

COLORS = {
    "Independent": "#4c78a8",
    "DML": "#d95f02",
    "SSML": "#1b9e77",
    "T-Indep": "#4c78a8",
    "DLinear-Indep": "#7f7f7f",
}


def compact_sci_label(value: float, _pos: float | None = None) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def load_curve(path: str | Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data[key]).reshape(-1)


def mean_curve(paths: list[str | Path], key: str, transform=None) -> np.ndarray:
    curves = []
    for path in paths:
        curve = load_curve(path, key)
        if transform is not None:
            curve = transform(curve)
        curves.append(curve)
    min_len = min(curve.size for curve in curves)
    stacked = np.stack([curve[:min_len] for curve in curves], axis=0)
    return stacked.mean(axis=0)


def pad_curve_to_length(curve: np.ndarray, target_len: int) -> np.ndarray:
    if curve.size >= target_len:
        return curve[:target_len]
    pad_width = target_len - curve.size
    return np.pad(curve, (0, pad_width), mode="edge")


def prefer_existing_paths(*path_sets: list[str | Path]) -> list[Path]:
    normalized_sets = [[Path(path) for path in path_set] for path_set in path_sets]
    for path_set in normalized_sets:
        if all(path.exists() for path in path_set):
            return path_set
    return normalized_sets[-1]


def mean_summary_metric(summary_paths: list[str | Path], key: str) -> float:
    values = []
    for path in summary_paths:
        with Path(path).open() as f:
            summary = json.load(f)
        values.append(float(summary[key]))
    return float(np.mean(values))


def neg_log10_curve(curve: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.log10(np.clip(curve, eps, None))


def select_best_completed_path_set(
    candidates: list[tuple[list[str | Path], list[str | Path]]],
    *,
    summary_key: str = "best_val_mse",
    maximize: bool = False,
) -> list[Path]:
    normalized_candidates = [
        ([Path(path) for path in curve_paths], [Path(path) for path in summary_paths])
        for curve_paths, summary_paths in candidates
    ]

    best_paths: list[Path] | None = None
    best_score: float | None = None
    fallback_paths: list[Path] | None = None

    for curve_paths, summary_paths in normalized_candidates:
        if not all(path.exists() for path in curve_paths):
            continue
        if fallback_paths is None:
            fallback_paths = curve_paths
        if summary_paths and all(path.exists() for path in summary_paths):
            score = mean_summary_metric(summary_paths, summary_key)
            if best_score is None or (score > best_score if maximize else score < best_score):
                best_score = score
                best_paths = curve_paths

    if best_paths is not None:
        return best_paths
    if fallback_paths is not None:
        return fallback_paths
    return normalized_candidates[-1][0]


def all_paths_exist(paths: list[str | Path]) -> bool:
    return all(Path(path).exists() for path in paths)


def load_optional_json(path: str | Path) -> dict | None:
    resolved = Path(path)
    if not resolved.exists():
        return None
    with resolved.open() as f:
        return json.load(f)


def add_tail_zoom_inset(
    ax,
    series: list[tuple[str, np.ndarray]],
    *,
    focus_labels: list[str],
    start_epoch: int,
    end_epoch: int | None = None,
    bounds: tuple[float, float, float, float] = (0.08, 0.52, 0.42, 0.36),
    title: str = "tail zoom",
    yscale: str | None = None,
) -> None:
    series_map = {label: curve for label, curve in series}
    zoom_series = [(label, series_map[label]) for label in focus_labels if label in series_map]
    if not zoom_series:
        return

    max_epoch = min(curve.size for _, curve in zoom_series)
    start_epoch = max(1, min(start_epoch, max_epoch))
    end_epoch = max_epoch if end_epoch is None else max(start_epoch, min(end_epoch, max_epoch))

    inset_ax = ax.inset_axes(bounds)
    inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.90))

    all_segments = []
    for label, curve in zoom_series:
        epochs = np.arange(start_epoch, end_epoch + 1)
        segment = curve[start_epoch - 1 : end_epoch]
        all_segments.append(segment)
        inset_ax.plot(
            epochs,
            segment,
            color=COLORS.get(label, None),
            linewidth=2.0,
        )
        inset_ax.scatter(
            [epochs[-1]],
            [segment[-1]],
            color=COLORS.get(label, None),
            s=20,
            zorder=3,
        )

    inset_ax.set_xlim(start_epoch, end_epoch)
    merged = np.concatenate(all_segments)
    if yscale == "log":
        positive = merged[merged > 0.0]
        if positive.size == 0:
            return
        y_min = float(positive.min())
        y_max = float(positive.max())
        y_low = y_min / 1.18
        y_high = y_max * 1.18
        inset_ax.set_yscale("log")
        inset_ax.set_ylim(y_low, y_high)
        inset_ax.yaxis.set_major_locator(FixedLocator(np.geomspace(y_low, y_high, 3)))
        inset_ax.yaxis.set_major_formatter(FuncFormatter(compact_sci_label))
        inset_ax.yaxis.set_minor_locator(NullLocator())
    else:
        y_min = float(merged.min())
        y_max = float(merged.max())
        y_pad = max((y_max - y_min) * 0.35, max(y_max, 1e-12) * 0.01)
        inset_ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    inset_ax.set_xticks([start_epoch, end_epoch])
    inset_ax.tick_params(labelsize=7)
    inset_ax.grid(alpha=0.2)
    inset_ax.set_title(title, fontsize=8, pad=2)

    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#555555")

    ax.indicate_inset_zoom(inset_ax, edgecolor="#555555", alpha=0.9)


def plot_panel(
    ax,
    title: str,
    ylabel: str,
    series: list[tuple[str, np.ndarray]],
    *,
    mark_final: bool = False,
    mark_best: bool = True,
    annotate_best: bool = True,
    yscale: str | None = None,
    inset: dict | None = None,
    xlim: tuple[int, int] | None = None,
    legend_loc: str = "best",
) -> None:
    for label, curve in series:
        epochs = np.arange(1, curve.size + 1)
        ax.plot(
            epochs,
            curve,
            label=label,
            color=COLORS.get(label, None),
            linewidth=2.0,
        )
        best_idx = int(np.argmin(curve))
        best_epoch = int(epochs[best_idx])
        best_value = float(curve[best_idx])
        if mark_best:
            ax.scatter(
                [best_epoch],
                [best_value],
                color=COLORS.get(label, None),
                s=28,
                zorder=3,
            )
        if annotate_best:
            ax.annotate(
                f"{label} best@{best_epoch}",
                (best_epoch, best_value),
                textcoords="offset points",
                xytext=(4, -12),
                fontsize=7,
                color=COLORS.get(label, None),
            )
        if mark_final:
            final_epoch = int(epochs[-1])
            final_value = float(curve[-1])
            ax.scatter(
                [final_epoch],
                [final_value],
                color=COLORS.get(label, None),
                s=28,
                marker="x",
                zorder=3,
            )
            ax.annotate(
                f"{label} final@{final_epoch}",
                (final_epoch, final_value),
                textcoords="offset points",
                xytext=(-46, 6),
                fontsize=7,
                color=COLORS.get(label, None),
            )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc=legend_loc)
    if inset is not None:
        add_tail_zoom_inset(ax, series, yscale=yscale, **inset)


def plot_zoom_panel(
    ax,
    title: str,
    ylabel: str,
    series: list[tuple[str, np.ndarray]],
    *,
    start_epoch: int,
    end_epoch: int,
    yscale: str | None = None,
) -> None:
    for label, curve in series:
        epochs = np.arange(start_epoch, end_epoch + 1)
        segment = curve[start_epoch - 1 : end_epoch]
        ax.plot(
            epochs,
            segment,
            label=label,
            color=COLORS.get(label, None),
            linewidth=2.0,
        )
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlim(start_epoch, end_epoch)
    ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def build_classification_plot() -> Path:
    out_path = OUT_DIR / "test_error_classification.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)

    c10_indep = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed0/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed1/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed0/curves.npz",
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed1/curves.npz",
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c10_dml = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed0/curves.npz",
                ROOT / "results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed1/curves.npz",
                ROOT / "results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_dml_kl_seed0/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_dml_kl_seed1/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_dml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_dml_kl_seed0/curves.npz",
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_dml_kl_seed1/curves.npz",
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_dml_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c10_ssml = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed0/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed1/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed0/curves.npz",
                ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed1/curves.npz",
                ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][0],
        "CIFAR-10 / ResNet18",
        "validation error",
        [("Independent", c10_indep), ("DML", c10_dml), ("SSML", c10_ssml)],
    )

    followup_report = load_optional_json(
        ROOT / "results/logs/classification_cifar100_strict128_followup_v1/node0/narrow_exploit_report.json"
    )
    followup_preview_case = (
        str(followup_report.get("preview_case"))
        if isinstance(followup_report, dict) and followup_report.get("preview_case")
        else "uh_sched_mem_v2"
    )
    followup_preview_mode = (
        str(followup_report.get("preview_mode"))
        if isinstance(followup_report, dict) and followup_report.get("preview_mode")
        else "aggressive_diagnostic"
    )

    def followup_case_root(label: str) -> Path:
        if label == "pcu_sched_df10_x05_r30_60":
            return ROOT / "results/classification_cifar100_strict128_followup_v1/probes"
        return ROOT / "results/classification_cifar100_strict128_followup_v1"

    def followup_curve_paths(label: str, method: str, seeds: tuple[int, ...]) -> list[Path]:
        root = followup_case_root(label) / label / "classification/cifar100"
        return [
            root / f"resnet34_gelu_{method}_kl_seed{seed}/curves.npz"
            for seed in seeds
        ]

    aggressive_indep_paths = [
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/strict128_independent_v2/classification/cifar100/resnet34_gelu_independent_kl_seed0/curves.npz",
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/strict128_independent_v2/classification/cifar100/resnet34_gelu_independent_kl_seed1/curves.npz",
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/strict128_independent_v2/classification/cifar100/resnet34_gelu_independent_kl_seed2/curves.npz",
    ]
    aggressive_dml_paths = [
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/strict128_dml_v2/classification/cifar100/resnet34_gelu_dml_kl_seed0/curves.npz",
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/strict128_dml_v2/classification/cifar100/resnet34_gelu_dml_kl_seed1/curves.npz",
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/strict128_dml_v2/classification/cifar100/resnet34_gelu_dml_kl_seed2/curves.npz",
    ]
    aggressive_ssml_paths = [
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/uh_sched_mem/classification/cifar100/resnet34_gelu_ssml_kl_seed0/curves.npz",
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/uh_sched_mem/classification/cifar100/resnet34_gelu_ssml_kl_seed1/curves.npz",
        ROOT / "results/classification_cifar100_strict128_aggressive_v1/uh_sched_mem/classification/cifar100/resnet34_gelu_ssml_kl_seed2/curves.npz",
    ]
    followup_indep_paths = followup_curve_paths("strict128_independent_v3", "independent", (0, 1, 2))
    followup_dml_paths = followup_curve_paths("strict128_dml_v3", "dml", (0, 1, 2))
    followup_ssml_paths = followup_curve_paths(followup_preview_case, "ssml", (0, 1, 2))
    use_followup_preview = (
        followup_preview_mode == "corrected_followup_3seed"
        and all_paths_exist(followup_indep_paths)
        and all_paths_exist(followup_dml_paths)
        and all_paths_exist(followup_ssml_paths)
    )

    if use_followup_preview:
        c100_indep_paths = followup_indep_paths
        c100_dml_paths = followup_dml_paths
        c100_ssml_paths = followup_ssml_paths
    else:
        c100_indep_paths = aggressive_indep_paths
        c100_dml_paths = aggressive_dml_paths
        c100_ssml_paths = aggressive_ssml_paths

    c100_indep = mean_curve(
        c100_indep_paths,
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c100_dml = mean_curve(
        c100_dml_paths,
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c100_ssml_strict128 = mean_curve(
        c100_ssml_paths,
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][1],
        "CIFAR-100 / ResNet34_GELU",
        "validation error",
        [("Independent", c100_indep), ("DML", c100_dml), ("SSML", c100_ssml_strict128)],
    )

    fig.suptitle("Classification Validation Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_classification_cifar100_followup_partial_plot() -> Path:
    out_path = OUT_DIR / "test_error_classification_cifar100_followup_partial.png"
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.8), squeeze=False)
    followup_report = load_optional_json(
        ROOT / "results/logs/classification_cifar100_strict128_followup_v1/node0/narrow_exploit_report.json"
    )
    preview_candidates = []
    if isinstance(followup_report, dict) and followup_report.get("preview_case"):
        preview_candidates.append(str(followup_report["preview_case"]))
    if "uh_sched_mem_v2" not in preview_candidates:
        preview_candidates.append("uh_sched_mem_v2")

    def followup_case_root(label: str) -> Path:
        if label == "pcu_sched_df10_x05_r30_60":
            return ROOT / "results/classification_cifar100_strict128_followup_v1/probes"
        return ROOT / "results/classification_cifar100_strict128_followup_v1"

    def followup_curve_paths(label: str, method: str, seeds: tuple[int, ...]) -> list[Path]:
        root = followup_case_root(label) / label / "classification/cifar100"
        return [
            root / f"resnet34_gelu_{method}_kl_seed{seed}/curves.npz"
            for seed in seeds
        ]

    selected_preview_case = "uh_sched_mem_v2"
    selected_seeds = (0, 1)
    for label in preview_candidates:
        for seeds in ((0, 1, 2), (0, 1)):
            indep_paths = followup_curve_paths("strict128_independent_v3", "independent", seeds)
            dml_paths = followup_curve_paths("strict128_dml_v3", "dml", seeds)
            ssml_paths = followup_curve_paths(label, "ssml", seeds)
            if all_paths_exist(indep_paths) and all_paths_exist(dml_paths) and all_paths_exist(ssml_paths):
                selected_preview_case = label
                selected_seeds = seeds
                break
        else:
            continue
        break

    followup_indep = mean_curve(
        followup_curve_paths("strict128_independent_v3", "independent", selected_seeds),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    followup_dml = mean_curve(
        followup_curve_paths("strict128_dml_v3", "dml", selected_seeds),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    followup_ssml = mean_curve(
        followup_curve_paths(selected_preview_case, "ssml", selected_seeds),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    seed_label = "3-seed" if len(selected_seeds) == 3 else "partial 2-seed"
    plot_panel(
        ax[0][0],
        f"CIFAR-100 / strict128 follow-up ({seed_label})",
        "validation error",
        [("Independent", followup_indep), ("DML", followup_dml), ("SSML", followup_ssml)],
    )
    fig.suptitle("CIFAR-100 Follow-up Validation Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_classification_cifar100_only_plot() -> Path:
    out_path = OUT_DIR / "test_error_cifar100_only.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)

    strict128_indep = mean_curve(
        [
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed0/curves.npz",
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed1/curves.npz",
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    strict128_dml = mean_curve(
        [
            ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed0/curves.npz",
            ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed1/curves.npz",
            ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    strict128_ssml = mean_curve(
        [
            ROOT / "results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed0/curves.npz",
            ROOT / "results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed1/curves.npz",
            ROOT / "results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][0],
        "CIFAR-100 / strict128 official",
        "validation error",
        [("Independent", strict128_indep), ("DML", strict128_dml), ("SSML", strict128_ssml)],
    )

    cifarstem_report = load_optional_json(
        ROOT / "results/logs/classification_cifar100_cifarstem_followup_v1/node0/cifarstem_followup_report.json"
    )
    preview_case = None
    preview_seeds = ()
    preview_mode = "pending"
    if isinstance(cifarstem_report, dict):
        if cifarstem_report.get("preview_case"):
            preview_case = str(cifarstem_report["preview_case"])
        preview_mode = str(cifarstem_report.get("preview_mode", "pending"))
        preview_seeds = tuple(int(seed) for seed in cifarstem_report.get("preview_seeds", []))

    def cifarstem_curve_paths(label: str, method: str, seeds: tuple[int, ...]) -> list[Path]:
        root = ROOT / "results/classification_cifar100_cifarstem_followup_v1" / label / "classification/cifar100"
        return [
            root / f"resnet34_cifar_gelu_{method}_kl_seed{seed}/curves.npz"
            for seed in seeds
        ]

    if preview_case and preview_seeds:
        cifarstem_indep_paths = cifarstem_curve_paths("cifarstem_independent_v1", "independent", preview_seeds)
        cifarstem_dml_paths = cifarstem_curve_paths("cifarstem_dml_v1", "dml", preview_seeds)
        cifarstem_ssml_paths = cifarstem_curve_paths(preview_case, "ssml", preview_seeds)
    else:
        cifarstem_indep_paths = []
        cifarstem_dml_paths = []
        cifarstem_ssml_paths = []

    if (
        preview_case
        and preview_seeds
        and all_paths_exist(cifarstem_indep_paths)
        and all_paths_exist(cifarstem_dml_paths)
        and all_paths_exist(cifarstem_ssml_paths)
    ):
        cifarstem_indep = mean_curve(
            cifarstem_indep_paths,
            "val_acc",
            transform=lambda x: 1.0 - x,
        )
        cifarstem_dml = mean_curve(
            cifarstem_dml_paths,
            "val_acc",
            transform=lambda x: 1.0 - x,
        )
        cifarstem_ssml = mean_curve(
            cifarstem_ssml_paths,
            "val_acc",
            transform=lambda x: 1.0 - x,
        )
        seed_label = "3-seed" if len(preview_seeds) == 3 else "seed2 probe"
        plot_panel(
            axes[0][1],
            f"CIFAR-100 / cifarstem follow-up ({seed_label})",
            "validation error",
            [("Independent", cifarstem_indep), ("DML", cifarstem_dml), ("SSML", cifarstem_ssml)],
        )
    else:
        ax = axes[0][1]
        ax.set_title("CIFAR-100 / cifarstem follow-up")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation error")
        ax.grid(alpha=0.25)
        ax.text(
            0.5,
            0.56,
            "pending matched preview",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.40,
            f"mode = {preview_mode}",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
            color="#555555",
        )
        if preview_case:
            ax.text(
                0.5,
                0.28,
                f"preview case = {preview_case}",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
                color="#555555",
            )

    fig.suptitle("CIFAR-100 Validation Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_final_wrapup_classification_plot() -> Path:
    out_path = OUT_DIR / "final_wrapup_classification.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)

    c10_indep = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed0/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed1/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed0/curves.npz",
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed1/curves.npz",
                ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c10_dml = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed0/curves.npz",
                ROOT / "results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed1/curves.npz",
                ROOT / "results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_dml_kl_seed0/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_dml_kl_seed1/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_dml_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c10_ssml = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed0/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed1/curves.npz",
                ROOT / "results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed0/curves.npz",
                ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed1/curves.npz",
                ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][0],
        "CIFAR-10 / ResNet18",
        "validation error",
        [("Independent", c10_indep), ("DML", c10_dml), ("SSML", c10_ssml)],
        mark_best=False,
        annotate_best=False,
    )

    cifarstem_report = load_optional_json(
        ROOT / "results/logs/classification_cifar100_cifarstem_followup_v1/node0/cifarstem_followup_report.json"
    )
    preview_case = "oxtra42_cifarstem_v1"
    preview_seeds = (0, 1, 2)
    if isinstance(cifarstem_report, dict):
        if cifarstem_report.get("preview_case"):
            preview_case = str(cifarstem_report["preview_case"])
        if cifarstem_report.get("preview_seeds"):
            preview_seeds = tuple(int(seed) for seed in cifarstem_report["preview_seeds"])

    def cifarstem_curve_paths(label: str, method: str, seeds: tuple[int, ...]) -> list[Path]:
        root = ROOT / "results/classification_cifar100_cifarstem_followup_v1" / label / "classification/cifar100"
        return [
            root / f"resnet34_cifar_gelu_{method}_kl_seed{seed}/curves.npz"
            for seed in seeds
        ]

    c100_indep = mean_curve(
        cifarstem_curve_paths("cifarstem_independent_v1", "independent", preview_seeds),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c100_dml = mean_curve(
        cifarstem_curve_paths("cifarstem_dml_v1", "dml", preview_seeds),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c100_ssml = mean_curve(
        cifarstem_curve_paths(preview_case, "ssml", preview_seeds),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][1],
        "CIFAR-100 / ResNet34_CIFAR_GELU",
        "validation error",
        [("Independent", c100_indep), ("DML", c100_dml), ("SSML", c100_ssml)],
        mark_best=False,
        annotate_best=False,
        xlim=(1, 40),
    )

    fig.suptitle("Final Wrap-up: Classification Validation Error")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_time_series_plot() -> Path:
    out_path = OUT_DIR / "test_error_time_series.png"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), squeeze=False)

    etth1_dml_paths = [
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed0/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed1/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed2/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed3/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed4/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed5/curves.npz",
    ]
    etth1_t_indep_paths = [
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/transformer_independent_huber_seed0/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/transformer_independent_huber_seed1/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/transformer_independent_huber_seed2/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/transformer_independent_huber_seed3/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/transformer_independent_huber_seed4/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/transformer_independent_huber_seed5/curves.npz",
    ]
    etth1_dlinear_indep_paths = [
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed0/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed1/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed2/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed3/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed4/curves.npz",
        ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed5/curves.npz",
    ]
    etth1_ssml_paths = select_best_completed_path_set(
        [
            (
                [
                    ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
                ],
                [
                    ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json",
                    ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/summary.json",
                    ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/summary.json",
                ],
            ),
            (
                [
                    ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
                ],
                [
                    ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json",
                    ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/summary.json",
                    ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/summary.json",
                ],
            ),
            (
                [
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q85_top18_a40_h22_r75_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q85_top18_a40_h22_r75_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q85_top18_a40_h22_r75_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
                ],
                [
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q85_top18_a40_h22_r75_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q85_top18_a40_h22_r75_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/summary.json",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q85_top18_a40_h22_r75_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/summary.json",
                ],
            ),
            (
                [
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q90_top15_a45_h18_r80_lr15e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q90_top15_a45_h18_r80_lr15e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q90_top15_a45_h18_r80_lr15e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
                ],
                [
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q90_top15_a45_h18_r80_lr15e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q90_top15_a45_h18_r80_lr15e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/summary.json",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q90_top15_a45_h18_r80_lr15e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/summary.json",
                ],
            ),
            (
                [
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q80_top20_a35_h26_r70_lr25e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q80_top20_a35_h26_r70_lr25e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q80_top20_a35_h26_r70_lr25e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
                ],
                [
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q80_top20_a35_h26_r70_lr25e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q80_top20_a35_h26_r70_lr25e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/summary.json",
                    ROOT / "results/time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1/worker0_gpu0/twr_q80_top20_a35_h26_r70_lr25e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/summary.json",
                ],
            ),
            (
                [
                    ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
                    ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
                    ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
                    ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed3/curves.npz",
                    ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed4/curves.npz",
                    ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed5/curves.npz",
                ],
                [],
            ),
        ]
    )
    etth1_t_indep = mean_curve(etth1_t_indep_paths, "val_mse")
    etth1_dlinear_indep = mean_curve(etth1_dlinear_indep_paths, "val_mse")
    etth1_dml = mean_curve(etth1_dml_paths, "val_mse")
    etth1_ssml = mean_curve(etth1_ssml_paths, "val_mse")
    plot_panel(
        axes[0][0],
        "ETTh1 / transformer:dlinear",
        "validation MSE",
        [
            ("DLinear", etth1_dlinear_indep),
            ("SSML", etth1_ssml),
            ("Transformer", etth1_t_indep),
            ("DML", etth1_dml),
        ],
    )

    weather_indep = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    weather_dml = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    weather_ssml = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][1],
        "Weather / transformer:dlinear",
        "validation MSE",
        [("Independent", weather_indep), ("DML", weather_dml), ("SSML", weather_ssml)],
    )

    elec_dml = mean_curve(
        [
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    elec_ssml_paths = prefer_existing_paths(
        [
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_do10_l20_sp10e4/time_series/electricity/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_do10_l20_sp10e4/time_series/electricity/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_do10_l20_sp10e4/time_series/electricity/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/paper_rerun_canonical/time_series/time_series/electricity/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/paper_rerun_canonical/time_series/time_series/electricity/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/paper_rerun_canonical/time_series/time_series/electricity/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
    )
    elec_ssml = mean_curve(elec_ssml_paths, "val_mse")
    # Electricity also uses the strongest single baseline as Independent.
    elec_indep = mean_curve(
        [
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/dlinear_independent_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/dlinear_independent_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/dlinear_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    elec_target_len = max(elec_indep.size, elec_dml.size, elec_ssml.size)
    elec_indep = pad_curve_to_length(elec_indep, elec_target_len)
    elec_dml = pad_curve_to_length(elec_dml, elec_target_len)
    elec_ssml = pad_curve_to_length(elec_ssml, elec_target_len)
    plot_panel(
        axes[0][2],
        "Electricity",
        "validation MSE",
        [("Independent", elec_indep), ("DML", elec_dml), ("SSML", elec_ssml)],
        mark_final=True,
    )

    fig.suptitle("Time-Series Validation Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_final_wrapup_time_series_plot() -> Path:
    out_path = OUT_DIR / "final_wrapup_time_series.png"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), squeeze=False)

    etth1_dml = mean_curve(
        [
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed2/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed3/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed4/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed5/curves.npz",
        ],
        "val_mse",
    )
    etth1_dlinear = mean_curve(
        [
            ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed2/curves.npz",
            ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed3/curves.npz",
            ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed4/curves.npz",
            ROOT / "results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed5/curves.npz",
        ],
        "val_mse",
    )
    etth1_ssml = mean_curve(
        [
            ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][0],
        "ETTh1 / transformer:dlinear",
        "validation MSE",
        [("Independent", etth1_dlinear), ("DML", etth1_dml), ("SSML", etth1_ssml)],
        mark_best=False,
        annotate_best=False,
        xlim=(1, 20),
    )

    weather_indep = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    weather_dml = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    weather_ssml = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][1],
        "Weather / transformer:dlinear",
        "validation MSE",
        [("Independent", weather_indep), ("DML", weather_dml), ("SSML", weather_ssml)],
        mark_best=False,
        annotate_best=False,
        xlim=(1, 20),
    )

    elec_dml = mean_curve(
        [
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    elec_ssml = mean_curve(
        [
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    elec_indep = mean_curve(
        [
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/dlinear_independent_mse_seed0/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/dlinear_independent_mse_seed1/curves.npz",
            ROOT / "results/time_series_electricity_followup_v1/best_known/time_series/electricity/dlinear_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][2],
        "Electricity / transformer:dlinear",
        "validation MSE",
        [("Independent", elec_indep), ("DML", elec_dml), ("SSML", elec_ssml)],
        mark_best=False,
        annotate_best=False,
        xlim=(1, 20),
    )

    fig.suptitle("Final Wrap-up: Time-Series Validation Error")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_operator_plot() -> Path:
    out_path = OUT_DIR / "test_error_operator.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)

    burgers_indep_curve_paths = prefer_existing_paths(
        [
            ROOT / "results/operator_burgers_polish_aggressive_v4/ctrl_cos_lr4e4_w10_min02_clip1/operator/burgers/fno_independent_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/ctrl_cos_lr4e4_w10_min02_clip1/operator/burgers/fno_independent_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/ctrl_cos_lr4e4_w10_min02_clip1/operator/burgers/fno_independent_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed2/curves.npz",
        ],
    )
    burgers_indep = mean_curve(burgers_indep_curve_paths, "val_mse")
    burgers_dml = mean_curve(
        [
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    burgers_ssml_curve_paths = prefer_existing_paths(
        [
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_full_l0012_s20_70_40_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_full_l0012_s20_70_40_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_full_l0012_s20_70_40_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_coarse_l0008_s20_70_50_element_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_coarse_l0008_s20_70_50_element_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_coarse_l0008_s20_70_50_element_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_full_l0010_w20_d90_150_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_full_l0010_w20_d90_150_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_full_l0010_w20_d90_150_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/operator_burgers_fno_polish_fair_v3/worker3_gpu0/fno_polish_ultra_full_l0012_w30_d110_180/operator/burgers/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_fno_polish_fair_v3/worker3_gpu0/fno_polish_ultra_full_l0012_w30_d110_180/operator/burgers/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_fno_polish_fair_v3/worker3_gpu0/fno_polish_ultra_full_l0012_w30_d110_180/operator/burgers/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/operator_burgers_fno_polish_fair_v2/worker3_gpu0/fno_polish_coarse_l002_w24_d90_170/operator/burgers/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_fno_polish_fair_v2/worker3_gpu0/fno_polish_coarse_l002_w24_d90_170/operator/burgers/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_fno_polish_fair_v2/worker3_gpu0/fno_polish_coarse_l002_w24_d90_170/operator/burgers/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        [
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
    )
    burgers_ssml = mean_curve(burgers_ssml_curve_paths, "val_mse")
    plot_panel(
        axes[0][0],
        "Burgers / FNO:DeepONet",
        "validation MSE",
        [("Independent", burgers_indep), ("DML", burgers_dml), ("SSML", burgers_ssml)],
        mark_best=False,
        annotate_best=False,
        yscale="log",
        inset={
            "focus_labels": ["Independent", "SSML"],
            "start_epoch": 151,
            "end_epoch": 180,
            "bounds": (0.08, 0.54, 0.42, 0.34),
            "title": "tail zoom (SSML vs Indep)",
        },
    )

    op_indep = mean_curve(
        [
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno_independent_mse_seed0/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno_independent_mse_seed1/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    op_dml = mean_curve(
        [
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_dml_mse_seed0/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_dml_mse_seed1/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    op_ssml = mean_curve(
        [
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][1],
        "Darcy / FNO:DeepONet",
        "validation MSE",
        [("Independent", op_indep), ("DML", op_dml), ("SSML", op_ssml)],
        mark_best=False,
        annotate_best=False,
        yscale="log",
        inset={
            "focus_labels": ["Independent", "SSML"],
            "start_epoch": 121,
            "end_epoch": 150,
            "bounds": (0.08, 0.54, 0.42, 0.34),
            "title": "tail zoom (SSML vs Indep)",
        },
    )
    fig.suptitle("Operator Validation Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_final_wrapup_operator_plot() -> Path:
    out_path = OUT_DIR / "final_wrapup_operator.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), squeeze=False)

    burgers_indep = mean_curve(
        [
            ROOT / "results/operator_burgers_polish_aggressive_v4/ctrl_cos_lr4e4_w10_min02_clip1/operator/burgers/fno_independent_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/ctrl_cos_lr4e4_w10_min02_clip1/operator/burgers/fno_independent_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/ctrl_cos_lr4e4_w10_min02_clip1/operator/burgers/fno_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    burgers_dml = mean_curve(
        [
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    burgers_ssml = mean_curve(
        [
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_full_l0012_s20_70_40_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_full_l0012_s20_70_40_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_polish_aggressive_v4/cos_relay_full_l0012_s20_70_40_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][0],
        "Burgers / FNO:DeepONet",
        "validation MSE",
        [("Independent", burgers_indep), ("DML", burgers_dml), ("SSML", burgers_ssml)],
        mark_best=False,
        annotate_best=False,
        yscale="log",
        xlim=(1, 100),
        legend_loc="lower left",
        inset={
            "focus_labels": ["Independent", "SSML"],
            "start_epoch": 86,
            "end_epoch": 100,
            "bounds": (0.60, 0.72, 0.30, 0.18),
            "title": "late-stage gap",
        },
    )

    darcy_indep = mean_curve(
        [
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno_independent_mse_seed0/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno_independent_mse_seed1/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    darcy_dml = mean_curve(
        [
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_dml_mse_seed0/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_dml_mse_seed1/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    darcy_ssml = mean_curve(
        [
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_ssml_mse_seed0/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_ssml_mse_seed1/curves.npz",
            ROOT / "results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_ssml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][1],
        "Darcy / FNO:DeepONet",
        "validation MSE",
        [("Independent", darcy_indep), ("DML", darcy_dml), ("SSML", darcy_ssml)],
        mark_best=False,
        annotate_best=False,
        yscale="log",
        xlim=(1, 100),
        legend_loc="lower left",
        inset={
            "focus_labels": ["Independent", "SSML"],
            "start_epoch": 82,
            "end_epoch": 100,
            "bounds": (0.56, 0.60, 0.35, 0.24),
            "title": "late-stage gap",
        },
    )

    fig.suptitle("Final Wrap-up: Operator Validation Error")
    fig.subplots_adjust(top=0.86, wspace=0.18)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> int:
    paths = [
        build_classification_plot(),
        build_classification_cifar100_followup_partial_plot(),
        build_classification_cifar100_only_plot(),
        build_final_wrapup_classification_plot(),
        build_time_series_plot(),
        build_final_wrapup_time_series_plot(),
        build_operator_plot(),
        build_final_wrapup_operator_plot(),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
