from __future__ import annotations

import textwrap
import os
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


DEFAULT_RESULT_TAG = os.environ.get("PAPER_RERUN_TAG", "paper_rerun_canonical")


SETUP_CELL = f"""
from pathlib import Path
import math
import os
import sys

def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "notebook").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current working directory.")

REPO_ROOT = find_repo_root(Path.cwd().resolve())
NOTEBOOK_ROOT = REPO_ROOT / "notebook" / "FINAL_WRAPUP"
for path in (REPO_ROOT, NOTEBOOK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from _shared.io import load_curve_file, load_run_tree
from _shared.plotting import (
    METHOD_COLORS,
    METHOD_ORDER,
    apply_report_style,
    pretty_dataset,
    pretty_method,
    pretty_model,
    save_figure,
    save_table,
)
from src.utils.pair_visualization import (
    aggregate_rows,
    best_methods,
    discover_rows,
)

apply_report_style()
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 200)

PAPER_RERUN_TAG = os.environ.get("PAPER_RERUN_TAG", "{DEFAULT_RESULT_TAG}")
RESULT_ROOT = REPO_ROOT / "results" / PAPER_RERUN_TAG
REPORT_TASK = "paper_rerun_progress"
RESULT_ROOT
"""


HELPER_CELL = """
METHOD_SEQUENCE = ["independent", "dml", "ssml"]
EXPECTED_SUMMARY_COUNTS = {
    "classification": 24,
    "time_series": 36,
    "operator": 12,
}
TASK_CONFIGS = {
    "classification": {
        "primary_model": "resnet18",
        "peer_model": "vit_b16",
        "datasets": ["cifar10", "cifar100"],
        "methods": ["independent", "dml", "ssml"],
        "metric_key": "acc",
        "direction": "maximize",
        "title": "Classification: ResNet-18 with ViT-B/16 peer",
    },
    "time_series": {
        "primary_model": "transformer",
        "peer_model": "dlinear",
        "datasets": ["etth1", "electricity", "weather"],
        "methods": ["independent", "dml", "ssml"],
        "metric_key": "mse",
        "direction": "minimize",
        "title": "Time-Series: Transformer with DLinear peer",
    },
    "operator": {
        "primary_model": "fno",
        "peer_model": "deeponet",
        "datasets": ["darcy"],
        "methods": ["independent", "dml", "ssml"],
        "metric_key": "mse",
        "direction": "minimize",
        "title": "Operator: FNO with DeepONet peer",
    },
}

def method_sequence_for(task):
    return TASK_CONFIGS[task].get("methods", METHOD_SEQUENCE)

def coerce_peer_series(series):
    cleaned = series.fillna("").astype(str).replace({"None": "", "nan": "", "<NA>": ""})
    return cleaned

def progress_frame(summary_runs):
    rows = []
    for task, expected in EXPECTED_SUMMARY_COUNTS.items():
        completed = int((summary_runs["task"] == task).sum())
        rows.append(
            {
                "task": task,
                "completed_runs": completed,
                "expected_runs": expected,
                "completion_ratio": completed / expected if expected else np.nan,
                "status": "complete" if completed >= expected else "in_progress",
            }
        )
    frame = pd.DataFrame(rows)
    frame["task_label"] = frame["task"].str.replace("_", " ").str.title()
    return frame

def primary_comparison_frame(agg, task):
    cfg = TASK_CONFIGS[task]
    methods = method_sequence_for(task)
    rows = []
    peer_series = coerce_peer_series(agg["peer_model"])
    for dataset in cfg["datasets"]:
        for method in methods:
            if method == "independent":
                view = agg[
                    (agg["task"] == task)
                    & (agg["dataset"] == dataset)
                    & (agg["model"] == cfg["primary_model"])
                    & (agg["method"] == method)
                    & (peer_series == "")
                ]
            else:
                view = agg[
                    (agg["task"] == task)
                    & (agg["dataset"] == dataset)
                    & (agg["model"] == cfg["primary_model"])
                    & (agg["method"] == method)
                    & (peer_series == cfg["peer_model"])
                ]
            if view.empty:
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "method": method,
                        "mean_metric": np.nan,
                        "std_metric": np.nan,
                        "n_runs": 0,
                        "is_complete": False,
                    }
                )
            else:
                row = view.sort_values(["n_runs", "mean_metric"], ascending=[False, True]).iloc[0]
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "method": method,
                        "mean_metric": float(row["mean_metric"]),
                        "std_metric": float(row["std_metric"]),
                        "n_runs": int(row["n_runs"]),
                        "is_complete": int(row["n_runs"]) >= 3,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["task_label"] = frame["task"].str.replace("_", " ").str.title()
    frame["dataset_label"] = frame["dataset"].map(pretty_dataset)
    frame["method_label"] = frame["method"].map(pretty_method)
    frame["method_order"] = frame["method"].map(METHOD_ORDER).fillna(99)
    return frame.sort_values(["task", "dataset", "method_order"])

def build_master_comparison(agg):
    frames = [primary_comparison_frame(agg, task) for task in TASK_CONFIGS]
    return pd.concat(frames, ignore_index=True)

def pair_model_comparison_frame(agg, task):
    cfg = TASK_CONFIGS[task]
    methods = method_sequence_for(task)
    rows = []
    peer_series = coerce_peer_series(agg["peer_model"])
    model_specs = [
        (cfg["primary_model"], cfg["peer_model"], 0),
        (cfg["peer_model"], cfg["primary_model"], 1),
    ]
    for dataset in cfg["datasets"]:
        for model_name, other_model, model_order in model_specs:
            for method in methods:
                if method == "independent":
                    view = agg[
                        (agg["task"] == task)
                        & (agg["dataset"] == dataset)
                        & (agg["model"] == model_name)
                        & (agg["method"] == method)
                        & (peer_series == "")
                    ]
                else:
                    view = agg[
                        (agg["task"] == task)
                        & (agg["dataset"] == dataset)
                        & (agg["model"] == model_name)
                        & (agg["method"] == method)
                        & (peer_series == other_model)
                    ]
                if view.empty:
                    rows.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "model_name": model_name,
                            "peer_name": other_model,
                            "method": method,
                            "mean_metric": np.nan,
                            "std_metric": np.nan,
                            "n_runs": 0,
                            "is_complete": False,
                            "model_order": model_order,
                        }
                    )
                else:
                    row = view.sort_values(["n_runs", "mean_metric"], ascending=[False, True]).iloc[0]
                    rows.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "model_name": model_name,
                            "peer_name": other_model,
                            "method": method,
                            "mean_metric": float(row["mean_metric"]),
                            "std_metric": float(row["std_metric"]),
                            "n_runs": int(row["n_runs"]),
                            "is_complete": int(row["n_runs"]) >= 3,
                            "model_order": model_order,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["task_label"] = frame["task"].str.replace("_", " ").str.title()
    frame["dataset_label"] = frame["dataset"].map(pretty_dataset)
    frame["model_label"] = frame["model_name"].map(pretty_model)
    frame["peer_label"] = frame["peer_name"].map(pretty_model)
    frame["method_label"] = frame["method"].map(pretty_method)
    frame["method_order"] = frame["method"].map(METHOD_ORDER).fillna(99)
    return frame.sort_values(["task", "dataset", "model_order", "method_order"])

def build_pair_model_comparison(agg):
    frames = [pair_model_comparison_frame(agg, task) for task in TASK_CONFIGS]
    return pd.concat(frames, ignore_index=True)

def comparison_pivot(frame, task):
    task_frame = frame[frame["task"] == task].copy()
    return task_frame.pivot(index="dataset_label", columns="method_label", values="mean_metric")

def curve_key_candidates(task, model_idx):
    if task == "classification":
        return [f"val_acc{model_idx}", "val_acc"]
    if task in {"time_series", "operator"}:
        return [f"val_mse{model_idx}", "val_mse"]
    return [f"val_loss{model_idx}", "val_loss"]

def curve_ylabel(task):
    return {
        "classification": "Validation accuracy",
        "time_series": "Validation MSE",
        "operator": "Validation MSE",
    }[task]

def select_run_row(runs, task, dataset, method, model, peer_model):
    frame = runs[
        (runs["task"] == task)
        & (runs["dataset"] == dataset)
        & (runs["method"] == method)
        & (runs["model"] == model)
    ].copy()
    if peer_model is None:
        frame = frame[coerce_peer_series(frame["peer_model"]) == ""]
    else:
        frame = frame[coerce_peer_series(frame["peer_model"]) == peer_model]
    if frame.empty:
        return None
    return frame.sort_values(["best_metric", "seed"], ascending=[task != "classification", True]).iloc[0]

def plot_progress(progress_df):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(progress_df))
    bars = ax.bar(
        x,
        progress_df["completion_ratio"].astype(float),
        color=["#5e81ac" if status == "complete" else "#d08770" for status in progress_df["status"]],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(progress_df["task_label"], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Completion ratio")
    ax.set_title("Paper Rerun Progress")
    ax.grid(axis="y", alpha=0.3)
    for bar, completed, expected in zip(bars, progress_df["completed_runs"], progress_df["expected_runs"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{completed}/{expected}", ha="center", va="bottom", fontsize=9)
    return fig

def plot_task_comparison(comparison_df, task):
    cfg = TASK_CONFIGS[task]
    methods = method_sequence_for(task)
    task_df = comparison_df[comparison_df["task"] == task].copy()
    datasets = cfg["datasets"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.5), squeeze=False)
    axes = axes[0]
    for ax, dataset in zip(axes, datasets):
        data = task_df[task_df["dataset"] == dataset].copy().sort_values("method_order")
        x = np.arange(len(methods))
        means = [float(v) if pd.notna(v) else np.nan for v in data["mean_metric"]]
        stds = [float(v) if pd.notna(v) else 0.0 for v in data["std_metric"].fillna(0.0)]
        colors = [METHOD_COLORS.get(method, "#888888") for method in data["method"]]
        bars = ax.bar(x, np.nan_to_num(means, nan=0.0), yerr=stds, capsize=3, color=colors, alpha=0.92)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_method(method) for method in data["method"]], rotation=20, ha="right")
        ax.set_title(pretty_dataset(dataset))
        ax.set_ylabel(cfg["metric_key"])
        ax.grid(axis="y", alpha=0.3)
        for bar, mean_value, n_runs in zip(bars, means, data["n_runs"]):
            if math.isnan(mean_value):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "pending", rotation=90, ha="center", va="bottom", fontsize=8, color="#bf616a")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n_runs}", ha="center", va="bottom", fontsize=8)
        if cfg["direction"] == "maximize":
            ymax = np.nanmax(np.asarray(means, dtype=float)) if np.isfinite(np.nanmax(np.asarray(means, dtype=float))) else 1.0
            ax.set_ylim(0, max(0.05, ymax * 1.25))
        else:
            positive = [m for m in means if not math.isnan(m)]
            ymax = max(positive) if positive else 1.0
            ax.set_ylim(0, ymax * 1.25)
    fig.suptitle(cfg["title"], fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

def plot_dual_model_comparison(pair_df, task):
    cfg = TASK_CONFIGS[task]
    methods = method_sequence_for(task)
    models = [cfg["primary_model"], cfg["peer_model"]]
    datasets = cfg["datasets"]
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(5.2 * len(models), 4.0 * len(datasets)), squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx][col_idx]
            data = pair_df[
                (pair_df["task"] == task)
                & (pair_df["dataset"] == dataset)
                & (pair_df["model_name"] == model_name)
            ].copy().sort_values("method_order")
            x = np.arange(len(methods))
            means = [float(v) if pd.notna(v) else np.nan for v in data["mean_metric"]]
            stds = [float(v) if pd.notna(v) else 0.0 for v in data["std_metric"].fillna(0.0)]
            colors = [METHOD_COLORS.get(method, "#888888") for method in data["method"]]
            bars = ax.bar(x, np.nan_to_num(means, nan=0.0), yerr=stds, capsize=3, color=colors, alpha=0.92)
            ax.set_xticks(x)
            ax.set_xticklabels([pretty_method(method) for method in data["method"]], rotation=20, ha="right")
            ax.set_title(f"{pretty_dataset(dataset)} | {pretty_model(model_name)}")
            if col_idx == 0:
                ax.set_ylabel(cfg["metric_key"])
            ax.grid(axis="y", alpha=0.3)
            for bar, mean_value, n_runs in zip(bars, means, data["n_runs"]):
                if math.isnan(mean_value):
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "pending", rotation=90, ha="center", va="bottom", fontsize=8, color="#bf616a")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n_runs}", ha="center", va="bottom", fontsize=8)
            if cfg["direction"] == "maximize":
                finite_means = [m for m in means if not math.isnan(m)]
                ymax = max(finite_means) if finite_means else 1.0
                ax.set_ylim(0, max(0.05, ymax * 1.25))
            else:
                finite_means = [m for m in means if not math.isnan(m)]
                ymax = max(finite_means) if finite_means else 1.0
                ax.set_ylim(0, ymax * 1.25)
    fig.suptitle(f"{cfg['title']} | Both Models", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_task_curves(runs, task):
    cfg = TASK_CONFIGS[task]
    methods = method_sequence_for(task)
    selected_dataset = cfg["datasets"][0]
    if task == "time_series":
        selected_dataset = "weather"
    if task == "classification":
        selected_dataset = "cifar10"
    task_runs = []
    for method in methods:
        peer_model = None if method == "independent" else cfg["peer_model"]
        row = select_run_row(runs, task, selected_dataset, method, cfg["primary_model"], peer_model)
        if row is None:
            continue
        curves = load_curve_file(Path(row["run_dir"]) / "curves.npz")
        candidates = curve_key_candidates(task, int(row.get("model_idx", 1)))
        curve = None
        for key in candidates:
            if key in curves:
                curve = np.asarray(curves[key], dtype=float).reshape(-1)
                break
        if curve is None or curve.size == 0:
            continue
        task_runs.append((method, curve))
    if not task_runs:
        raise ValueError(f"No usable curves for task={task}")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for method, curve in task_runs:
        epochs = np.arange(1, len(curve) + 1)
        ax.plot(
            epochs,
            curve,
            label=pretty_method(method),
            color=METHOD_COLORS.get(method, "#888888"),
            linewidth=2.0,
        )
    ax.set_title(f"{TASK_CONFIGS[task]['title']} | {pretty_dataset(selected_dataset)}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(curve_ylabel(task))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_joint_pair_curve(runs, task):
    cfg = TASK_CONFIGS[task]
    selected_dataset = cfg["datasets"][0]
    if task == "time_series":
        selected_dataset = "weather"
    if task == "classification":
        selected_dataset = "cifar10"
    candidate_methods = [method for method in method_sequence_for(task) if method != "independent"]
    row = None
    chosen_method = None
    for method in candidate_methods:
        row = select_run_row(runs, task, selected_dataset, method, cfg["primary_model"], cfg["peer_model"])
        if row is not None:
            chosen_method = method
            break
    if row is None:
        raise ValueError(f"No joint run found for task={task}")
    curves = load_curve_file(Path(row["run_dir"]) / "curves.npz")
    curve1 = None
    curve2 = None
    for key in curve_key_candidates(task, 1):
        if key in curves:
            curve1 = np.asarray(curves[key], dtype=float).reshape(-1)
            break
    for key in curve_key_candidates(task, 2):
        if key in curves:
            curve2 = np.asarray(curves[key], dtype=float).reshape(-1)
            break
    if curve1 is None or curve2 is None or curve1.size == 0 or curve2.size == 0:
        raise ValueError(f"Joint curves missing for task={task}")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    epochs = np.arange(1, len(curve1) + 1)
    ax.plot(epochs, curve1, label=f"{pretty_model(cfg['primary_model'])} ({pretty_method(chosen_method)})", color="#5e81ac", linewidth=2.0)
    ax.plot(epochs, curve2, label=f"{pretty_model(cfg['peer_model'])} ({pretty_method(chosen_method)})", color="#d08770", linewidth=2.0)
    ax.set_title(f"{TASK_CONFIGS[task]['title']} | {pretty_dataset(selected_dataset)} | Joint Pair")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(curve_ylabel(task))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
"""


def build_progress_notebook():
    cells = [
        md(
            f"""
            # Paper Rerun Progress

            현재 `results/{DEFAULT_RESULT_TAG}` 기준으로 **진행률**, **메인 hetero pair 성능**, **두 모델 비교**, **대표 validation curve**를 정리합니다.

            목적:
            - 지금 어디까지 끝났는지 확인
            - hetero pair (`ResNet18+ViT`, `Transformer+DLinear`, `FNO+DeepONet`)가 independent 대비 어떤지 빠르게 본다
            - co-training run에서 `model1/model2`가 모두 실제로 학습됐는지 시각적으로 확인한다
            """
        ),
        code(SETUP_CELL),
        md("## Load"),
        code(
            """
            if not RESULT_ROOT.exists():
                raise FileNotFoundError(f"Paper rerun result root not found: {RESULT_ROOT}")

            summary_runs = load_run_tree(RESULT_ROOT)
            run_rows = discover_rows(RESULT_ROOT)
            agg_rows = aggregate_rows(run_rows)
            best_rows = best_methods(agg_rows)

            runs = pd.DataFrame(run_rows)
            agg = pd.DataFrame(agg_rows)
            best = pd.DataFrame(best_rows)
            paper_task_names = ["classification", "time_series", "operator"]

            display(summary_runs[summary_runs["task"].isin(paper_task_names)].groupby("task").size().rename("completed_summary_runs").to_frame())
            display(agg[agg["task"].isin(paper_task_names)].head())
            """
        ),
        md("## Normalize"),
        code(HELPER_CELL),
        md("## Summary Table"),
        code(
            """
            progress = progress_frame(summary_runs)
            paper_tasks = list(TASK_CONFIGS.keys())
            runs = runs[runs["task"].isin(paper_tasks)].copy()
            agg = agg[agg["task"].isin(paper_tasks)].copy()
            comparison = build_master_comparison(agg)
            pair_comparison = build_pair_model_comparison(agg)
            best_overview = best[best["task"].isin(paper_tasks)].copy()

            display(progress)
            display(comparison)
            display(pair_comparison)
            display(best_overview)
            """
        ),
        md("## Main Figures"),
        code(
            """
            progress_fig = plot_progress(progress)
            progress_fig_path = save_figure(progress_fig, REPORT_TASK, "progress_overview")
            display(progress_fig)
            display(Markdown(f"Saved: `{progress_fig_path}`"))

            comparison_fig_paths = []
            for task_name in TASK_CONFIGS:
                fig = plot_task_comparison(comparison, task_name)
                path = save_figure(fig, REPORT_TASK, f"{task_name}_primary_comparison")
                comparison_fig_paths.append(path)
                display(fig)
                display(Markdown(f"Saved: `{path}`"))

            dual_model_fig_paths = []
            for task_name in TASK_CONFIGS:
                fig = plot_dual_model_comparison(pair_comparison, task_name)
                path = save_figure(fig, REPORT_TASK, f"{task_name}_both_models_comparison")
                dual_model_fig_paths.append(path)
                display(fig)
                display(Markdown(f"Saved: `{path}`"))
            """
        ),
        md("## Secondary Figures"),
        code(
            """
            curve_fig_paths = []
            for task_name in TASK_CONFIGS:
                try:
                    fig = plot_task_curves(runs, task_name)
                except Exception as exc:
                    display(Markdown(f"- Skipped `{task_name}` curves: {exc}"))
                    continue
                path = save_figure(fig, REPORT_TASK, f"{task_name}_representative_curve")
                curve_fig_paths.append(path)
                display(fig)
                display(Markdown(f"Saved: `{path}`"))

            joint_curve_fig_paths = []
            for task_name in TASK_CONFIGS:
                try:
                    fig = plot_joint_pair_curve(runs, task_name)
                except Exception as exc:
                    display(Markdown(f"- Skipped `{task_name}` joint pair curves: {exc}"))
                    continue
                path = save_figure(fig, REPORT_TASK, f"{task_name}_joint_pair_curve")
                joint_curve_fig_paths.append(path)
                display(fig)
                display(Markdown(f"Saved: `{path}`"))
            """
        ),
        md("## Export"),
        code(
            """
            progress_path = save_table(progress, REPORT_TASK, "progress_status")
            comparison_path = save_table(comparison, REPORT_TASK, "primary_model_comparison")
            pair_comparison_path = save_table(pair_comparison, REPORT_TASK, "both_models_comparison")
            best_path = save_table(best_overview, REPORT_TASK, "best_methods_current")

            export_manifest = pd.DataFrame(
                [
                    {"artifact": "progress table", "path": str(progress_path)},
                    {"artifact": "comparison table", "path": str(comparison_path)},
                    {"artifact": "both-models comparison table", "path": str(pair_comparison_path)},
                    {"artifact": "best-method table", "path": str(best_path)},
                    {"artifact": "scope", "path": "classification,time_series,operator"},
                ]
            )
            display(export_manifest)
            export_manifest_path = save_table(export_manifest, REPORT_TASK, "export_manifest")
            display(Markdown(f"Saved export manifest to `{export_manifest_path}`"))
            """
        ),
        md("## Notes"),
        code(
            """
            pending = progress[progress["status"] != "complete"].copy()
            if pending.empty:
                display(Markdown("All configured paper rerun groups are complete."))
            else:
                display(Markdown("Incomplete groups remain:"))
                display(pending[["task_label", "completed_runs", "expected_runs", "completion_ratio"]])

            display(Markdown("Current figures are validation-based. Test metrics are not yet exported by the rerun summaries."))

            for task_name in TASK_CONFIGS:
                pivot = comparison_pivot(comparison, task_name)
                display(Markdown(f"### {TASK_CONFIGS[task_name]['title']}"))
                display(pivot)
            """
        ),
    ]
    return notebook(cells, "Paper Rerun Progress")


def main():
    out_path = ROOT / "05_paper_rerun_progress.ipynb"
    nb = build_progress_notebook()
    nbf.write(nb, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
