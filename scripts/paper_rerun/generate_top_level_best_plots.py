from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT

COLORS = {
    "Independent": "#4c78a8",
    "DML": "#d95f02",
    "SSML": "#1b9e77",
    "T-Indep": "#4c78a8",
    "DLinear-Indep": "#7f7f7f",
}


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


def plot_panel(ax, title: str, ylabel: str, series: list[tuple[str, np.ndarray]]) -> None:
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
        ax.scatter(
            [best_epoch],
            [best_value],
            color=COLORS.get(label, None),
            s=28,
            zorder=3,
        )
        ax.annotate(
            f"{label} best@{best_epoch}",
            (best_epoch, best_value),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=7,
            color=COLORS.get(label, None),
        )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def build_classification_plot() -> Path:
    out_path = OUT_DIR / "test_error_classification.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)

    c10_indep = mean_curve(
        [
            ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed0/curves.npz",
            ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed1/curves.npz",
            ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_independent_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c10_dml = mean_curve(
        [
            ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_dml_kl_seed0/curves.npz",
            ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_dml_kl_seed1/curves.npz",
            ROOT / "results/classification_ssml_topk_sweep_cifar10_v3/cifar10/baseline_dml/classification/cifar10/resnet18_dml_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c10_ssml = mean_curve(
        [
            ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed0/curves.npz",
            ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed1/curves.npz",
            ROOT / "results/classification_ssml_reweight_cifar10_v4/cifar10/ssml_reweight_t0p1/classification/cifar10/resnet18_ssml_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][0],
        "CIFAR-10 / ResNet18",
        "test error",
        [("Independent", c10_indep), ("DML", c10_dml), ("SSML", c10_ssml)],
    )

    c100_indep = mean_curve(
        [
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed0/curves.npz",
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed1/curves.npz",
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c100_ssml = mean_curve(
        [
            ROOT / "results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/cifar100/resnet34_gelu_ssml_kl_seed0/curves.npz",
            ROOT / "results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/cifar100/resnet34_gelu_ssml_kl_seed1/curves.npz",
            ROOT / "results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/cifar100/resnet34_gelu_ssml_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][1],
        "CIFAR-100 / ResNet34_GELU",
        "test error",
        [("Independent", c100_indep), ("SSML", c100_ssml)],
    )

    fig.suptitle("Classification Test Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_time_series_plot() -> Path:
    out_path = OUT_DIR / "test_error_time_series.png"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), squeeze=False)

    etth1_indep = mean_curve(
        [
            ROOT / "results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer_independent_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer_independent_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer_independent_huber_seed2/curves.npz",
        ],
        "val_mse",
    )
    etth1_dml = mean_curve(
        [
            ROOT / "results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer__dlinear_dml_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer__dlinear_dml_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer__dlinear_dml_huber_seed2/curves.npz",
        ],
        "val_mse",
    )
    etth1_ssml = mean_curve(
        [
            ROOT / "results/time_series_etth1_rescue_v3/a0p5/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_rescue_v3/a0p5/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_rescue_v3/a0p5/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][0],
        "ETTh1 / transformer:dlinear",
        "test error (MSE)",
        [("Independent", etth1_indep), ("DML", etth1_dml), ("SSML", etth1_ssml)],
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
        "test error (MSE)",
        [("Independent", weather_indep), ("DML", weather_dml), ("SSML", weather_ssml)],
    )

    elec_t_indep = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer_independent_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer_independent_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    elec_dml = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_dml_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_dml_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_dml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    elec_ssml = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_ssml_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_ssml_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_ssml_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    elec_dlinear_indep = mean_curve(
        [
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/dlinear_independent_mse_seed0/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/dlinear_independent_mse_seed1/curves.npz",
            ROOT / "results/instruction_matrix_v1/time_series/time_series/electricity/dlinear_independent_mse_seed2/curves.npz",
        ],
        "val_mse",
    )
    plot_panel(
        axes[0][2],
        "Electricity",
        "test error (MSE)",
        [
            ("T-Indep", elec_t_indep),
            ("DML", elec_dml),
            ("SSML", elec_ssml),
            ("DLinear-Indep", elec_dlinear_indep),
        ],
    )

    fig.suptitle("Time-Series Test Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_operator_plot() -> Path:
    out_path = OUT_DIR / "test_error_operator.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

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
        ax,
        "Darcy / FNO:DeepONet",
        "test error (MSE)",
        [("Independent", op_indep), ("DML", op_dml), ("SSML", op_ssml)],
    )
    fig.suptitle("Operator Test Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> int:
    paths = [
        build_classification_plot(),
        build_time_series_plot(),
        build_operator_plot(),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
