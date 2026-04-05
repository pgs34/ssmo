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


def prefer_existing_paths(*path_sets: list[str | Path]) -> list[Path]:
    normalized_sets = [[Path(path) for path in path_set] for path_set in path_sets]
    for path_set in normalized_sets:
        if all(path.exists() for path in path_set):
            return path_set
    return normalized_sets[-1]


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

    c100_indep = mean_curve(
        [
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed0/curves.npz",
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed1/curves.npz",
            ROOT / "results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed2/curves.npz",
        ],
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c100_dml = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed0/curves.npz",
                ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed1/curves.npz",
                ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t4/classification/cifar100/resnet34_gelu_dml_kl_seed0/curves.npz",
                ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t4/classification/cifar100/resnet34_gelu_dml_kl_seed1/curves.npz",
                ROOT / "results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t4/classification/cifar100/resnet34_gelu_dml_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    c100_ssml = mean_curve(
        prefer_existing_paths(
            [
                ROOT / "results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed0/curves.npz",
                ROOT / "results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed1/curves.npz",
                ROOT / "results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_cifar100_alt_focus_v2/worker3_queue/conf_pb26_thr33_aw4e4/classification/cifar100/resnet34_gelu_ssml_kl_seed0/curves.npz",
                ROOT / "results/classification_cifar100_alt_focus_v2/worker3_queue/conf_pb26_thr33_aw4e4/classification/cifar100/resnet34_gelu_ssml_kl_seed1/curves.npz",
                ROOT / "results/classification_cifar100_alt_focus_v2/worker3_queue/conf_pb26_thr33_aw4e4/classification/cifar100/resnet34_gelu_ssml_kl_seed2/curves.npz",
            ],
            [
                ROOT / "results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/cifar100/resnet34_gelu_ssml_kl_seed0/curves.npz",
                ROOT / "results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/cifar100/resnet34_gelu_ssml_kl_seed1/curves.npz",
                ROOT / "results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/cifar100/resnet34_gelu_ssml_kl_seed2/curves.npz",
            ],
        ),
        "val_acc",
        transform=lambda x: 1.0 - x,
    )
    plot_panel(
        axes[0][1],
        "CIFAR-100 / ResNet34_GELU",
        "validation error",
        [("Independent", c100_indep), ("DML", c100_dml), ("SSML", c100_ssml)],
    )

    fig.suptitle("Classification Validation Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_time_series_plot() -> Path:
    out_path = OUT_DIR / "test_error_time_series.png"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), squeeze=False)

    # Keep ETTh1 aligned with the latest best rescue in Results_Summary.md.
    etth1_indep_paths = [
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed0/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed1/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed2/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed3/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed4/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed5/curves.npz",
    ]
    etth1_dml_paths = [
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed0/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed1/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed2/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed3/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed4/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed5/curves.npz",
    ]
    etth1_ssml_paths = prefer_existing_paths(
        [
            ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
        ],
        [
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed0/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed1/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed2/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed3/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed4/curves.npz",
            ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_ssml_huber_seed5/curves.npz",
        ],
    )
    etth1_dlinear_indep_paths = [
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/dlinear_independent_huber_seed0/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/dlinear_independent_huber_seed1/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/dlinear_independent_huber_seed2/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/dlinear_independent_huber_seed3/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/dlinear_independent_huber_seed4/curves.npz",
        ROOT / "results/time_series_etth1_all_methods_long_v3/time_series/etth1/dlinear_independent_huber_seed5/curves.npz",
    ]

    # For the top-level comparison, use a single Independent curve only.
    # ETTh1's strongest single baseline is the DLinear independent run.
    etth1_indep = mean_curve(etth1_dlinear_indep_paths, "val_mse")
    etth1_dml = mean_curve(etth1_dml_paths, "val_mse")
    etth1_ssml = mean_curve(etth1_ssml_paths, "val_mse")
    plot_panel(
        axes[0][0],
        "ETTh1 / transformer:dlinear",
        "validation MSE",
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
    elec_ssml = mean_curve(
        prefer_existing_paths(
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
        ),
        "val_mse",
    )
    # Electricity also uses the strongest single baseline as Independent.
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
        "Electricity",
        "validation MSE",
        [("Independent", elec_indep), ("DML", elec_dml), ("SSML", elec_ssml)],
    )

    fig.suptitle("Time-Series Validation Error by Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_operator_plot() -> Path:
    out_path = OUT_DIR / "test_error_operator.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)

    burgers_indep = mean_curve(
        [
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed0/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed1/curves.npz",
            ROOT / "results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed2/curves.npz",
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
        prefer_existing_paths(
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
        ),
        "val_mse",
    )
    plot_panel(
        axes[0][0],
        "Burgers / FNO:DeepONet",
        "validation MSE",
        [("Independent", burgers_indep), ("DML", burgers_dml), ("SSML", burgers_ssml)],
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
    )
    fig.suptitle("Operator Validation Error by Epoch")
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
