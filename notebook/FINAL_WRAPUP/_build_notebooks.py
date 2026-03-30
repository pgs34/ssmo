from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")


def notebook(cells: list, title: str):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "title": title,
    }
    return nb


SETUP_CELL = """
from pathlib import Path
import sys

def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "notebook").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current working directory.")

REPO_ROOT = find_repo_root(Path.cwd().resolve())
NOTEBOOK_ROOT = REPO_ROOT / "notebook" / "FINAL_WRAPUP"
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from _shared.io import load_curve_file, load_epoch_metrics, load_pks_results, load_run_tree
from _shared.plotting import (
    DATASET_ORDER,
    METHOD_COLORS,
    METHOD_ORDER,
    apply_report_style,
    pretty_dataset,
    pretty_method,
    pretty_model,
    pretty_pair,
    save_figure,
    save_table,
)

apply_report_style()
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 180)

NOTEBOOK_ROOT
"""


CURVE_HELPERS_CELL = """
def stack_curves(curves):
    prepared = []
    for curve in curves:
        if curve is None:
            continue
        arr = np.asarray(curve, dtype=float).reshape(-1)
        if arr.size:
            prepared.append(arr)
    if not prepared:
        raise ValueError("No non-empty curves were provided.")
    min_len = min(len(arr) for arr in prepared)
    return np.vstack([arr[:min_len] for arr in prepared])

def mean_and_std(curves):
    stacked = stack_curves(curves)
    return stacked.mean(axis=0), stacked.std(axis=0)
"""


def build_index_notebook():
    cells = [
        md(
            """
            # Final Wrap-Up Index

            보고서형 최종 notebook 진입점입니다. 아래 표는 실제 데이터 소스 기준 인벤토리이며,
            각 task notebook은 `Setup -> Load -> Normalize -> Summary Table -> Main Figures -> Secondary Figures -> Export -> Notes`
            순서를 따릅니다.

            - `01_time_series_wrapup.ipynb`
            - `02_classification_wrapup.ipynb`
            - `03_operator_wrapup.ipynb`
            """
        ),
        code(SETUP_CELL),
        md(
            """
            ## Load

            PKS 결과와 내 실험 결과 루트를 확인하고, notebook별 예상 row/file 개수를 표로 정리합니다.
            """
        ),
        code(
            """
            PKS_ROOT = REPO_ROOT / "notebook" / "results_from_pks" / "result_zip"
            TIME_SERIES_ROOT = REPO_ROOT / "results" / "paper_rerun_canonical" / "time_series" / "time_series"

            inventory_rows = []

            pks = load_pks_results(PKS_ROOT)
            inventory_rows.append({
                "source": "PKS classification",
                "expected_artifacts": 15,
                "observed_artifacts": pks.loc[pks["task"] == "classification", "source_path"].nunique(),
                "root": str(PKS_ROOT),
            })
            inventory_rows.append({
                "source": "PKS operator",
                "expected_artifacts": 18,
                "observed_artifacts": pks.loc[pks["task"] == "operator", "source_path"].nunique(),
                "root": str(PKS_ROOT),
            })

            time_series = load_run_tree(TIME_SERIES_ROOT)
            inventory_rows.append({
                "source": "Time-series summary.json",
                "expected_artifacts": 36,
                "observed_artifacts": len(time_series),
                "root": str(TIME_SERIES_ROOT),
            })

            inventory = pd.DataFrame(inventory_rows)
            display(inventory)
            inventory
            """
        ),
        md("## Export"),
        code(
            """
            export_path = save_table(inventory, "index", "source_inventory")
            display(Markdown(f"Saved inventory table to `{export_path}`."))
            """
        ),
        md(
            """
            ## Notes

            - `navierstokes` PKS 폴더는 현재 비어 있으므로 operator wrap-up에서 unavailable 상태를 명시합니다.
            - time-series는 `results/paper_rerun_canonical/time_series/time_series`만 공식 입력으로 사용합니다.
            """
        ),
    ]
    return notebook(cells, "Final Wrap-Up Index")


def build_time_series_notebook():
    cells = [
        md(
            """
            # Time-Series Wrap-Up

            최종 메시지는 **final MSE/MAE 비교**이며, 수렴 곡선과 imitation dynamics는 보조 해석으로 둡니다.

            공식 입력 소스:
            - `results/paper_rerun_canonical/time_series/time_series`
            """
        ),
        code(SETUP_CELL),
        md("## Load"),
        code(
            """
            TIME_SERIES_ROOT = REPO_ROOT / "results" / "paper_rerun_canonical" / "time_series" / "time_series"
            time_runs = load_run_tree(TIME_SERIES_ROOT)
            if len(time_runs) != 36:
                raise AssertionError(f"Expected 36 time-series runs, found {len(time_runs)}")

            expected_datasets = {"etth1", "electricity", "weather"}
            observed_datasets = set(time_runs["dataset"].unique())
            if observed_datasets != expected_datasets:
                raise AssertionError(f"Unexpected time-series dataset set: {observed_datasets}")

            display(time_runs.head())
            time_runs[["dataset", "model", "peer_model", "method", "loss_name"]].drop_duplicates().sort_values(
                ["dataset", "model", "peer_model", "method"]
            )
            """
        ),
        md("## Normalize"),
        code(
            CURVE_HELPERS_CELL
            + textwrap.dedent(
                """

                def pair_category(row):
                    peer = row["peer_model"]
                    if pd.isna(peer) or peer is None:
                        return "single"
                    if peer == row["model"]:
                        return "homogeneous"
                    return "heterogeneous"

                time_runs = time_runs.copy()
                time_runs["pair_category"] = time_runs.apply(pair_category, axis=1)
                time_runs["dataset_label"] = time_runs["dataset"].map(pretty_dataset)
                time_runs["model_label"] = time_runs["model"].map(pretty_model)
                time_runs["peer_label"] = time_runs["peer_model"].map(pretty_model)
                time_runs["method_label"] = time_runs["method"].map(pretty_method)

                def config_label(row):
                    if row["pair_category"] == "single":
                        return f"{pretty_model(row['model'])} (single)"
                    return f"{pretty_model(row['model'])} | peer={pretty_model(row['peer_model'])}"

                time_runs["config_label"] = time_runs.apply(config_label, axis=1)
                time_runs["dataset_order"] = time_runs["dataset"].map(DATASET_ORDER)
                time_runs["method_order"] = time_runs["method"].map(METHOD_ORDER)
                time_runs["pair_order"] = time_runs["pair_category"].map({"single": 0, "homogeneous": 1, "heterogeneous": 2})

                summary_columns = [
                    "dataset",
                    "pair_category",
                    "model",
                    "peer_model",
                    "method",
                    "final_metric",
                    "best_metric",
                    "final_val_mae",
                    "mean_imitation_weight",
                    "active_imitation_ratio",
                ]
                display(
                    time_runs
                    .sort_values(["dataset_order", "pair_order", "model", "peer_model", "method_order"])[summary_columns]
                    .reset_index(drop=True)
                )
                """
            )
        ),
        md("## Summary Table"),
        code(
            """
            time_summary = (
                time_runs[
                    [
                        "dataset",
                        "dataset_label",
                        "pair_category",
                        "config_label",
                        "method",
                        "method_label",
                        "final_metric",
                        "best_metric",
                        "final_val_mae",
                        "mean_imitation_weight",
                        "active_imitation_ratio",
                    ]
                ]
                .sort_values(["dataset", "pair_category", "config_label", "method"])
                .reset_index(drop=True)
            )

            display(time_summary)
            raw_export_path = save_table(time_summary, "time_series", "time_series_summary")
            display(Markdown(f"Saved summary table to `{raw_export_path}`."))
            """
        ),
        md("## Main Figures"),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, sorted(time_runs["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))):
                ds = (
                    time_runs[time_runs["dataset"] == dataset]
                    .sort_values(["pair_order", "model", "peer_model", "method_order"])
                    .reset_index(drop=True)
                )
                x = np.arange(len(ds))
                colors = [METHOD_COLORS.get(method, "#4c566a") for method in ds["method"]]
                ax.bar(x, ds["final_metric"], color=colors)
                ax.set_title(pretty_dataset(dataset))
                ax.set_ylabel("Final MSE")
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [f"{label}\\n{method}" for label, method in zip(ds["config_label"], ds["method_label"])],
                    rotation=60,
                    ha="right",
                )

                categories = ds["pair_category"].tolist()
                for idx in range(1, len(categories)):
                    if categories[idx] != categories[idx - 1]:
                        ax.axvline(idx - 0.5, color="#d8dee9", linestyle="--", linewidth=1.0)

                category_positions = ds.groupby("pair_category").agg(start=("pair_order", "min"), count=("pair_order", "size"))
                for category, cat_df in ds.groupby("pair_category"):
                    start = cat_df.index.min()
                    stop = cat_df.index.max()
                    midpoint = (start + stop) / 2.0
                    ax.text(midpoint, ax.get_ylim()[1] * 1.02, category, ha="center", va="bottom", fontsize=9)

            fig.suptitle("Dataset-wise Final MSE Comparison", fontsize=16)
            fig.tight_layout()
            main_path = save_figure(fig, "time_series", "time_series_final_mse")
            display(Markdown(f"Saved main figure to `{main_path}`."))
            plt.show()
            """
        ),
        md("## Secondary Figures"),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, sorted(time_runs["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))):
                ds = (
                    time_runs[time_runs["dataset"] == dataset]
                    .sort_values(["pair_order", "model", "peer_model", "method_order"])
                    .reset_index(drop=True)
                )
                x = np.arange(len(ds))
                colors = [METHOD_COLORS.get(method, "#4c566a") for method in ds["method"]]
                ax.bar(x, ds["final_val_mae"], color=colors)
                ax.set_title(pretty_dataset(dataset))
                ax.set_ylabel("Final MAE")
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [f"{label}\\n{method}" for label, method in zip(ds["config_label"], ds["method_label"])],
                    rotation=60,
                    ha="right",
                )

            fig.suptitle("Dataset-wise Final MAE Comparison", fontsize=16)
            fig.tight_layout()
            mae_path = save_figure(fig, "time_series", "time_series_final_mae")
            display(Markdown(f"Saved MAE figure to `{mae_path}`."))
            plt.show()
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, sorted(time_runs["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))):
                subset = time_runs[
                    (time_runs["dataset"] == dataset)
                    & (
                        ((time_runs["model"] == "transformer") & (time_runs["peer_model"] == "dlinear"))
                        | ((time_runs["model"] == "transformer") & (time_runs["method"] == "independent"))
                    )
                ].copy()
                subset["curve_label"] = subset.apply(
                    lambda row: "Transformer independent"
                    if row["method"] == "independent"
                    else f"Transformer + DLinear ({pretty_method(row['method'])})",
                    axis=1,
                )

                for _, row in subset.sort_values(["method_order"]).iterrows():
                    curves = load_curve_file(row["curve_path"])
                    ax.plot(
                        curves["val_mse"],
                        label=row["curve_label"],
                        color=METHOD_COLORS.get(row["method"], "#4c566a"),
                        linewidth=2.0,
                    )

                ax.set_title(pretty_dataset(dataset))
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Validation MSE")
                ax.legend(fontsize=8)

            fig.suptitle("Representative Validation MSE Curves", fontsize=16)
            fig.tight_layout()
            curve_path = save_figure(fig, "time_series", "time_series_representative_val_mse_curves")
            display(Markdown(f"Saved representative curve figure to `{curve_path}`."))
            plt.show()
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=False)

            for col, dataset in enumerate(sorted(time_runs["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))):
                subset = time_runs[
                    (time_runs["dataset"] == dataset)
                    & (time_runs["model"] == "transformer")
                    & (time_runs["peer_model"] == "dlinear")
                ].copy()

                top_ax = axes[0, col]
                bottom_ax = axes[1, col]

                for _, row in subset.sort_values(["method_order"]).iterrows():
                    metrics = load_epoch_metrics(row["epoch_metrics_path"])
                    top_ax.plot(
                        metrics["epoch"],
                        metrics["mean_imitation_weight"],
                        label=pretty_method(row["method"]),
                        color=METHOD_COLORS.get(row["method"], "#4c566a"),
                        linewidth=2.0,
                    )
                    bottom_ax.plot(
                        metrics["epoch"],
                        metrics["active_imitation_ratio"],
                        label=pretty_method(row["method"]),
                        color=METHOD_COLORS.get(row["method"], "#4c566a"),
                        linewidth=2.0,
                    )

                top_ax.set_title(pretty_dataset(dataset))
                top_ax.set_ylabel("Mean imitation weight")
                bottom_ax.set_xlabel("Epoch")
                bottom_ax.set_ylabel("Active ratio")

            axes[0, 0].legend(fontsize=8)
            fig.suptitle("Heterogeneous Pair Dynamics (Transformer + DLinear)", fontsize=16)
            fig.tight_layout()
            dynamics_path = save_figure(fig, "time_series", "time_series_heterogeneous_dynamics")
            display(Markdown(f"Saved dynamics figure to `{dynamics_path}`."))
            plt.show()
            """
        ),
        md("## Export"),
        code(
            """
            export_frame = time_runs.drop(
                columns=["summary_path", "curve_path", "epoch_metrics_path", "epoch_log_path"],
                errors="ignore",
            ).copy()
            export_path = save_table(export_frame, "time_series", "time_series_runs")
            display(Markdown(f"Saved normalized run table to `{export_path}`."))
            """
        ),
        md(
            """
            ## Notes

            - 현재 결과는 single-seed wrap-up이라 표준편차 해석은 하지 않습니다.
            - `heterogeneous`는 `Transformer + DLinear`, `homogeneous`는 동일 아키텍처 peer 조합입니다.
            - 메인 메시지는 final metric이지만, 마지막 figure에서 imitation weight와 active ratio 변화도 함께 남깁니다.
            """
        ),
    ]
    return notebook(cells, "Time-Series Wrap-Up")


def build_classification_notebook():
    cells = [
        md(
            """
            # Classification Wrap-Up

            PKS가 정리해 둔 `mnist`, `cifar10`, `cifar100` 결과를 바탕으로 single baseline과 SSML pair 구성을 같은 표 체계로 비교합니다.
            """
        ),
        code(SETUP_CELL),
        md("## Load"),
        code(
            """
            PKS_ROOT = REPO_ROOT / "notebook" / "results_from_pks" / "result_zip"
            classification = load_pks_results(PKS_ROOT)
            classification = classification[classification["task"] == "classification"].copy()
            classification_file_count = classification["source_path"].nunique()
            if classification_file_count != 15:
                raise AssertionError(f"Expected 15 classification PKS CSV files, found {classification_file_count}")

            display(classification.head())
            classification[["dataset", "source_path"]].drop_duplicates().sort_values(["dataset", "source_path"])
            """
        ),
        md("## Normalize"),
        code(
            CURVE_HELPERS_CELL
            + textwrap.dedent(
                """

                classification["dataset_label"] = classification["dataset"].map(pretty_dataset)
                classification["model_label"] = classification["model"].map(pretty_model)
                classification["peer_label"] = classification["peer_model"].map(pretty_model)
                classification["metric_percent"] = classification["metric_value"] * 100.0

                def config_label(row):
                    if row["mode"] == "single_baseline":
                        return f"{pretty_model(row['model'])} (single)"
                    return f"{pretty_model(row['model'])} | peer={pretty_model(row['peer_model'])}"

                classification["config_label"] = classification.apply(config_label, axis=1)

                classification_summary = (
                    classification.groupby(["dataset", "dataset_label", "config_label"], dropna=False)
                    .agg(
                        mean_accuracy=("metric_percent", "mean"),
                        std_accuracy=("metric_percent", "std"),
                        n_seeds=("seed", "nunique"),
                    )
                    .reset_index()
                    .sort_values(["dataset", "config_label"])
                )

                display(classification_summary)
                """
            )
        ),
        md("## Summary Table"),
        code(
            """
            summary_export_path = save_table(classification_summary, "classification", "classification_summary")
            display(Markdown(f"Saved summary table to `{summary_export_path}`."))
            classification_summary
            """
        ),
        md("## Main Figures"),
        code(
            """
            datasets = sorted(classification_summary["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))
            fig, axes = plt.subplots(1, len(datasets), figsize=(20, 5), sharey=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, datasets):
                ds = classification_summary[classification_summary["dataset"] == dataset].copy().reset_index(drop=True)
                x = np.arange(len(ds))
                ax.bar(x, ds["mean_accuracy"], color="#5e81ac")
                std_values = ds["std_accuracy"].fillna(0.0).to_numpy()
                ax.errorbar(x, ds["mean_accuracy"], yerr=std_values, fmt="none", ecolor="#2e3440", capsize=4)
                ax.set_title(pretty_dataset(dataset))
                ax.set_ylabel("Best accuracy (%)")
                ax.set_xticks(x)
                ax.set_xticklabels(ds["config_label"], rotation=45, ha="right")

            fig.suptitle("Dataset-wise Best Accuracy", fontsize=16)
            fig.tight_layout()
            main_path = save_figure(fig, "classification", "classification_best_accuracy")
            display(Markdown(f"Saved main figure to `{main_path}`."))
            plt.show()
            """
        ),
        md("## Secondary Figures"),
        code(
            """
            datasets = sorted(classification["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))
            fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 12), sharex=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, datasets):
                ds = classification[classification["dataset"] == dataset].copy()
                for config_label, group in ds.groupby("config_label"):
                    mean_curve, std_curve = mean_and_std(group["test_curve"])
                    epochs = np.arange(1, len(mean_curve) + 1)
                    ax.plot(epochs, mean_curve, linewidth=2.0, label=config_label)
                    ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

                ax.set_title(pretty_dataset(dataset))
                ax.set_ylabel("Test metric")
                ax.legend(fontsize=8, ncol=2)

            axes[-1].set_xlabel("Epoch")
            fig.suptitle("Representative Mean Test Curves", fontsize=16)
            fig.tight_layout()
            curve_path = save_figure(fig, "classification", "classification_mean_test_curves")
            display(Markdown(f"Saved test-curve figure to `{curve_path}`."))
            plt.show()
            """
        ),
        code(
            """
            datasets = sorted(classification["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))
            fig, axes = plt.subplots(1, len(datasets), figsize=(20, 5), sharey=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, datasets):
                ds = classification[classification["dataset"] == dataset].copy()
                labels = sorted(ds["config_label"].unique())
                positions = np.arange(len(labels))
                for pos, label in enumerate(labels):
                    values = ds.loc[ds["config_label"] == label, "metric_percent"].to_numpy()
                    ax.scatter(np.full_like(values, pos, dtype=float), values, s=45, alpha=0.8)

                ax.set_title(pretty_dataset(dataset))
                ax.set_ylabel("Best accuracy (%)")
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha="right")

            fig.suptitle("Seed-Level Dispersion", fontsize=16)
            fig.tight_layout()
            dispersion_path = save_figure(fig, "classification", "classification_seed_dispersion")
            display(Markdown(f"Saved dispersion figure to `{dispersion_path}`."))
            plt.show()
            """
        ),
        md("## Export"),
        code(
            """
            export_frame = classification.drop(columns=["train_curve", "test_curve"]).copy()
            export_path = save_table(export_frame, "classification", "classification_runs")
            display(Markdown(f"Saved normalized run table to `{export_path}`."))
            """
        ),
        md(
            """
            ## Notes

            - PKS classification CSV는 single baseline 2개와 SSML pair 3개 구성을 dataset마다 제공합니다.
            - pair 결과는 각 model의 관점으로 행을 분리해 `model | peer=...` 형식으로 비교합니다.
            - metric은 higher-is-better accuracy이므로 summary chart는 백분율로 표시합니다.
            """
        ),
    ]
    return notebook(cells, "Classification Wrap-Up")


def build_operator_notebook():
    cells = [
        md(
            """
            # Operator Wrap-Up

            PKS의 `burgers1d`, `darcy2d` 결과를 정리하고, 비어 있는 `navierstokes` 폴더는 unavailable 상태로 명시합니다.
            """
        ),
        code(SETUP_CELL),
        md("## Load"),
        code(
            """
            PKS_ROOT = REPO_ROOT / "notebook" / "results_from_pks" / "result_zip"
            operator = load_pks_results(PKS_ROOT)
            operator = operator[operator["task"] == "operator"].copy()
            operator_file_count = operator["source_path"].nunique()
            if operator_file_count != 18:
                raise AssertionError(f"Expected 18 operator PKS CSV files, found {operator_file_count}")

            navierstokes_dir = PKS_ROOT / "navierstokes"
            navierstokes_csv = sorted(navierstokes_dir.glob("*.csv")) if navierstokes_dir.exists() else []
            if navierstokes_csv:
                raise AssertionError("Expected the navierstokes PKS folder to be empty for this wrap-up.")

            display(Markdown("**Data unavailable:** `navierstokes` currently has no PKS CSV artifacts, so this notebook reports only Burgers-1D and Darcy-2D."))
            display(operator.head())
            """
        ),
        md("## Normalize"),
        code(
            CURVE_HELPERS_CELL
            + textwrap.dedent(
                """

                operator["dataset_label"] = operator["dataset"].map(pretty_dataset)
                operator["model_label"] = operator["model"].map(pretty_model)
                operator["peer_label"] = operator["peer_model"].map(pretty_model)

                def config_label(row):
                    if row["mode"] == "single_baseline":
                        return f"{pretty_model(row['model'])} (single)"
                    return f"{pretty_model(row['model'])} | peer={pretty_model(row['peer_model'])}"

                operator["config_label"] = operator.apply(config_label, axis=1)

                operator_summary = (
                    operator.groupby(["dataset", "dataset_label", "config_label"], dropna=False)
                    .agg(
                        mean_error=("metric_value", "mean"),
                        std_error=("metric_value", "std"),
                        n_seeds=("seed", "nunique"),
                    )
                    .reset_index()
                    .sort_values(["dataset", "config_label"])
                )

                display(operator_summary)
                """
            )
        ),
        md("## Summary Table"),
        code(
            """
            summary_export_path = save_table(operator_summary, "operator", "operator_summary")
            display(Markdown(f"Saved summary table to `{summary_export_path}`."))
            operator_summary
            """
        ),
        md("## Main Figures"),
        code(
            """
            datasets = sorted(operator_summary["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))
            fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5), sharey=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, datasets):
                ds = operator_summary[operator_summary["dataset"] == dataset].copy().reset_index(drop=True)
                x = np.arange(len(ds))
                ax.bar(x, ds["mean_error"], color="#bf616a")
                std_values = ds["std_error"].fillna(0.0).to_numpy()
                ax.errorbar(x, ds["mean_error"], yerr=std_values, fmt="none", ecolor="#2e3440", capsize=4)
                ax.set_title(pretty_dataset(dataset))
                ax.set_ylabel("Best error")
                ax.set_xticks(x)
                ax.set_xticklabels(ds["config_label"], rotation=45, ha="right")

            fig.suptitle("Dataset-wise Best Error", fontsize=16)
            fig.tight_layout()
            main_path = save_figure(fig, "operator", "operator_best_error")
            display(Markdown(f"Saved main figure to `{main_path}`."))
            plt.show()
            """
        ),
        md("## Secondary Figures"),
        code(
            """
            datasets = sorted(operator["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))
            fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 10), sharex=False)
            axes = np.atleast_2d(axes)

            for row_idx, dataset in enumerate(datasets):
                ds = operator[operator["dataset"] == dataset].copy()
                train_ax = axes[row_idx, 0]
                test_ax = axes[row_idx, 1]

                for config_label, group in ds.groupby("config_label"):
                    train_mean, train_std = mean_and_std(group["train_curve"])
                    test_mean, test_std = mean_and_std(group["test_curve"])
                    epochs = np.arange(1, len(train_mean) + 1)

                    train_ax.plot(epochs, train_mean, linewidth=2.0, label=config_label)
                    train_ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.15)

                    test_ax.plot(epochs, test_mean, linewidth=2.0, label=config_label)
                    test_ax.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.15)

                train_ax.set_title(f"{pretty_dataset(dataset)} | Train")
                train_ax.set_ylabel("Train error")
                test_ax.set_title(f"{pretty_dataset(dataset)} | Test")
                test_ax.set_ylabel("Test error")

            axes[-1, 0].set_xlabel("Epoch")
            axes[-1, 1].set_xlabel("Epoch")
            axes[0, 1].legend(fontsize=8, ncol=2)
            fig.suptitle("Representative Train/Test Curves", fontsize=16)
            fig.tight_layout()
            curve_path = save_figure(fig, "operator", "operator_train_test_curves")
            display(Markdown(f"Saved curve figure to `{curve_path}`."))
            plt.show()
            """
        ),
        code(
            """
            datasets = sorted(operator["dataset"].unique(), key=lambda x: DATASET_ORDER.get(x, 99))
            fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5), sharey=False)
            axes = np.atleast_1d(axes)

            for ax, dataset in zip(axes, datasets):
                ds = operator[operator["dataset"] == dataset].copy()
                labels = sorted(ds["config_label"].unique())
                positions = np.arange(len(labels))
                for pos, label in enumerate(labels):
                    values = ds.loc[ds["config_label"] == label, "metric_value"].to_numpy()
                    ax.scatter(np.full_like(values, pos, dtype=float), values, s=45, alpha=0.8)

                ax.set_title(pretty_dataset(dataset))
                ax.set_ylabel("Best error")
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha="right")

            fig.suptitle("Seed-Level Dispersion", fontsize=16)
            fig.tight_layout()
            dispersion_path = save_figure(fig, "operator", "operator_seed_dispersion")
            display(Markdown(f"Saved dispersion figure to `{dispersion_path}`."))
            plt.show()
            """
        ),
        md("## Export"),
        code(
            """
            export_frame = operator.drop(columns=["train_curve", "test_curve"]).copy()
            export_path = save_table(export_frame, "operator", "operator_runs")
            display(Markdown(f"Saved normalized run table to `{export_path}`."))
            """
        ),
        md(
            """
            ## Notes

            - operator metric은 lower-is-better error로 해석합니다.
            - `navierstokes`는 현재 PKS 결과가 없으므로 unavailable로 명시하고 억지로 채우지 않습니다.
            - pair 결과는 각 model 관점을 분리해 같은 테이블에서 비교합니다.
            """
        ),
    ]
    return notebook(cells, "Operator Wrap-Up")

def main():
    outputs = {
        "00_index.ipynb": build_index_notebook(),
        "01_time_series_wrapup.ipynb": build_time_series_notebook(),
        "02_classification_wrapup.ipynb": build_classification_notebook(),
        "03_operator_wrapup.ipynb": build_operator_notebook(),
    }

    for relative_name, nb in outputs.items():
        out_path = ROOT / relative_name
        nbf.write(nb, out_path)
        print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
