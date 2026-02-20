from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

METHOD_ORDER = {"independent": 0, "naive": 1, "dml": 2, "studygroup": 3}
MAXIMIZE_METRICS = {"acc", "miou", "pixel_acc"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate and visualize method-comparison experiment results.")
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _to_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _slug(s: str) -> str:
    allowed = []
    for c in str(s):
        if c.isalnum() or c in "._-":
            allowed.append(c)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")


def _infer_metric_key(task: str, data: dict[str, Any]) -> str:
    if data.get("best_metric_key"):
        return str(data["best_metric_key"])
    if task == "classification":
        return "acc"
    if task in {"operator", "time_series"}:
        return "mse"
    if task == "segmentation":
        return "miou"
    return "metric"


def _infer_best_metric(task: str, data: dict[str, Any], suffix: str = "") -> float:
    if suffix and f"best_metric{suffix}" in data:
        return _to_float(data.get(f"best_metric{suffix}"))
    if "best_metric" in data:
        return _to_float(data.get("best_metric"))
    if task == "classification":
        return _to_float(data.get("best_val_acc"))
    if task in {"operator", "time_series"}:
        return _to_float(data.get("best_val_mse"))
    if task == "segmentation":
        return _to_float(data.get("best_val_miou"))
    return float("nan")


def _infer_final_val(task: str, data: dict[str, Any], suffix: str = "") -> float:
    if suffix and f"final_val{suffix}" in data:
        return _to_float(data.get(f"final_val{suffix}"))
    if "final_val" in data:
        return _to_float(data.get("final_val"))
    if "final_val_loss" in data:
        return _to_float(data.get("final_val_loss"))
    if task in {"operator", "time_series"}:
        return _to_float(data.get("final_val_mse"))
    return float("nan")


def _make_row(
    *,
    summary_path: Path,
    task: str,
    dataset: str,
    method: str,
    model: str,
    peer_model: str,
    seed: int,
    metric_key: str,
    best_metric: float,
    final_val: float,
    epochs: int,
    warmup_epochs: int,
    lambda_imitation: float,
    margin: float,
    curve_mode: str,
    model_idx: int,
) -> dict[str, Any]:
    return {
        "task": task,
        "dataset": dataset,
        "method": method,
        "model": model,
        "peer_model": peer_model,
        "seed": seed,
        "metric_key": metric_key,
        "best_metric": best_metric,
        "final_val": final_val,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "lambda_imitation": lambda_imitation,
        "margin": margin,
        "curve_mode": curve_mode,
        "model_idx": model_idx,
        "run_dir": str(summary_path.parent),
        "summary_path": str(summary_path),
    }


def discover_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(input_dir.rglob("summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[VIS][warn] failed to read {summary_path}: {e}")
            continue

        task = str(data.get("task", "")).strip()
        method = str(data.get("method", "")).strip()
        if not task or not method:
            continue

        dataset = str(data.get("dataset", "")).strip()
        if not dataset:
            train_ds = str(data.get("train_dataset", "")).strip()
            val_ds = str(data.get("val_dataset", "")).strip()
            if task == "segmentation" and train_ds and val_ds:
                dataset = f"{train_ds}_to_{val_ds}"
        if not dataset:
            continue

        seed = _to_int(data.get("seed", -1))
        metric_key = _infer_metric_key(task, data)
        epochs = _to_int(data.get("epochs", -1))
        warmup_epochs = _to_int(data.get("warmup_epochs", -1))
        lambda_imitation = _to_float(data.get("lambda_imitation"))
        margin = _to_float(data.get("margin"))

        # Preferred model-centric schema.
        if data.get("model"):
            rows.append(
                _make_row(
                    summary_path=summary_path,
                    task=task,
                    dataset=dataset,
                    method=method,
                    model=str(data.get("model", "")),
                    peer_model=str(data.get("peer_model", "")),
                    seed=seed,
                    metric_key=metric_key,
                    best_metric=_infer_best_metric(task, data),
                    final_val=_infer_final_val(task, data),
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    lambda_imitation=lambda_imitation,
                    margin=margin,
                    curve_mode=str(data.get("curve_mode", "single")),
                    model_idx=_to_int(data.get("model_idx", 1), 1),
                )
            )
            continue

        # Backward compatibility for old pair schema.
        model1 = str(data.get("model1", "")).strip()
        model2 = str(data.get("model2", "")).strip()
        if not model1 or not model2:
            continue

        rows.append(
            _make_row(
                summary_path=summary_path,
                task=task,
                dataset=dataset,
                method=method,
                model=model1,
                peer_model=model2,
                seed=seed,
                metric_key=metric_key,
                best_metric=_infer_best_metric(task, data, suffix="1"),
                final_val=_infer_final_val(task, data, suffix="1"),
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                lambda_imitation=lambda_imitation,
                margin=margin,
                curve_mode="pair",
                model_idx=1,
            )
        )
        rows.append(
            _make_row(
                summary_path=summary_path,
                task=task,
                dataset=dataset,
                method=method,
                model=model2,
                peer_model=model1,
                seed=seed,
                metric_key=metric_key,
                best_metric=_infer_best_metric(task, data, suffix="2"),
                final_val=_infer_final_val(task, data, suffix="2"),
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                lambda_imitation=lambda_imitation,
                margin=margin,
                curve_mode="pair",
                model_idx=2,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return float(statistics.mean(clean)), float(statistics.pstdev(clean))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["task"], r["dataset"], r["model"], r["method"], r["metric_key"])].append(r)

    out: list[dict[str, Any]] = []
    for key, items in grouped.items():
        task, dataset, model, method, metric_key = key
        vals = [float(i["best_metric"]) for i in items]
        mean_val, std_val = _mean_std(vals)
        out.append(
            {
                "task": task,
                "dataset": dataset,
                "model": model,
                "method": method,
                "metric_key": metric_key,
                "n_runs": len(items),
                "mean_metric": mean_val,
                "std_metric": std_val,
            }
        )

    out.sort(
        key=lambda r: (
            r["task"],
            r["dataset"],
            r["model"],
            METHOD_ORDER.get(r["method"], 99),
            r["method"],
        )
    )
    return out


def best_methods(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in agg_rows:
        grouped[(r["task"], r["dataset"], r["model"], r["metric_key"])].append(r)

    out: list[dict[str, Any]] = []
    for key, items in grouped.items():
        task, dataset, model, metric_key = key
        maximize = metric_key in MAXIMIZE_METRICS

        def metric_value(x: dict[str, Any]) -> float:
            v = float(x["mean_metric"])
            if math.isnan(v):
                return float("-inf") if maximize else float("inf")
            return v

        best_item = max(items, key=metric_value) if maximize else min(items, key=metric_value)
        out.append(
            {
                "task": task,
                "dataset": dataset,
                "model": model,
                "metric_key": metric_key,
                "direction": "maximize" if maximize else "minimize",
                "best_method": best_item["method"],
                "best_mean_metric": best_item["mean_metric"],
            }
        )

    out.sort(key=lambda r: (r["task"], r["dataset"], r["model"]))
    return out


def _model_root(output_dir: Path, task: str, dataset: str, model: str) -> Path:
    return output_dir / _slug(task) / _slug(dataset) / _slug(model)


def save_metric_bar_plots(agg_rows: list[dict[str, Any]], output_dir: Path) -> int:
    if not HAS_MATPLOTLIB:
        print("[VIS][warn] matplotlib unavailable; CSV only.")
        return 0

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in agg_rows:
        grouped[(r["task"], r["dataset"], r["model"], r["metric_key"])].append(r)

    count = 0
    for key, items in grouped.items():
        task, dataset, model, metric_key = key
        model_root = _model_root(output_dir, task, dataset, model)
        model_root.mkdir(parents=True, exist_ok=True)

        items.sort(key=lambda r: (METHOD_ORDER.get(r["method"], 99), r["method"]))
        methods = [r["method"] for r in items]
        means = [float(r["mean_metric"]) for r in items]
        stds = [float(r["std_metric"]) for r in items]

        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(methods)), 4.6))
        ax.bar(x, means, 0.62, yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel(metric_key)
        ax.set_title(f"{task} | {dataset} | {model}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        fig.savefig(model_root / f"{_slug(metric_key)}_by_method.png", dpi=160)
        plt.close(fig)
        count += 1

    return count


def _latest_rows_per_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    out: list[dict[str, Any]] = []
    for items in grouped.values():
        out.append(max(items, key=lambda x: str(x["run_dir"])))
    return out


def _load_curve(path: Path, keys: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            for key in keys:
                if key in data:
                    values = np.asarray(data[key], dtype=np.float64).reshape(-1)
                    return values
    except Exception as e:
        print(f"[VIS][warn] failed to load curve from {path}: {e}")
        return None
    return None


def _curve_keys(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    mode = str(row.get("curve_mode", "single"))
    if mode == "pair":
        idx = int(row.get("model_idx", 1))
        return [f"train_total{idx}", f"train_loss{idx}", f"train_sup{idx}", "train_total", "train_loss"], [
            f"val_loss{idx}",
            "val_loss",
            f"val_mse{idx}",
        ]
    return ["train_total", "train_loss", "train_sup", "train_mse"], ["val_loss", "val_mse"]


def save_loss_curve_plots(rows: list[dict[str, Any]], output_dir: Path) -> int:
    if not HAS_MATPLOTLIB:
        return 0

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["dataset"], row["model"], row["seed"])].append(row)

    count = 0
    for key, items in grouped.items():
        task, dataset, model, seed = key
        model_root = _model_root(output_dir, task, dataset, model)
        model_root.mkdir(parents=True, exist_ok=True)

        latest = _latest_rows_per_method(items)
        latest.sort(key=lambda r: (METHOD_ORDER.get(r["method"], 99), r["method"]))
        if not latest:
            continue

        cmap = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), squeeze=False)
        ax_train = axes[0][0]
        ax_test = axes[0][1]

        has_any_line = False
        linestyle_cycle = ["-", "--", ":", "-."]
        marker_cycle = ["o", "s", "^", "v"]
        for i, row in enumerate(latest):
            method = row["method"]
            color = cmap(i % 10)
            style = linestyle_cycle[i % len(linestyle_cycle)]
            marker = marker_cycle[i % len(marker_cycle)]
            curve_path = Path(row["run_dir"]) / "curves.npz"
            train_keys, val_keys = _curve_keys(row)
            train_curve = _load_curve(curve_path, train_keys)
            val_curve = _load_curve(curve_path, val_keys)

            if train_curve is not None and train_curve.size > 0:
                epochs = np.arange(1, train_curve.size + 1)
                ax_train.plot(
                    epochs,
                    train_curve,
                    label=method,
                    color=color,
                    linewidth=2.0,
                    linestyle=style,
                    marker=marker,
                    markevery=max(1, train_curve.size // 8),
                    markersize=3.5,
                )
                has_any_line = True
            if val_curve is not None and val_curve.size > 0:
                epochs = np.arange(1, val_curve.size + 1)
                ax_test.plot(
                    epochs,
                    val_curve,
                    label=method,
                    color=color,
                    linewidth=2.0,
                    linestyle=style,
                    marker=marker,
                    markevery=max(1, val_curve.size // 8),
                    markersize=3.5,
                )
                has_any_line = True

        if not has_any_line:
            plt.close(fig)
            continue

        ax_train.set_title("Train")
        ax_test.set_title("Test")
        ax_train.set_xlabel("epoch")
        ax_test.set_xlabel("epoch")
        ax_train.set_ylabel("loss")
        ax_test.set_ylabel("loss")
        ax_train.grid(alpha=0.3)
        ax_test.grid(alpha=0.3)
        ax_train.legend(fontsize=8)
        ax_test.legend(fontsize=8)

        fig.suptitle(f"{task} | {dataset} | {model} | seed={seed}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fig.savefig(model_root / f"seed{seed}__train_test_loss_by_method.png", dpi=160)
        plt.close(fig)
        count += 1

    return count


def clear_plot_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    model_dirs: set[Path] = set()
    legacy_dataset_dirs: set[Path] = set()
    for row in rows:
        task = _slug(row["task"])
        dataset = _slug(row["dataset"])
        model = _slug(row["model"])
        model_dirs.add(output_dir / task / dataset / model)
        legacy_dataset_dirs.add(output_dir / f"{task}_pair" / dataset)

    legacy_plots_dir = output_dir / "plots"
    if legacy_plots_dir.exists():
        shutil.rmtree(legacy_plots_dir)

    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        for pat in ("*_by_method.png", "*train_test_loss_by_method.png"):
            for png in model_dir.glob(pat):
                if png.is_file():
                    png.unlink()

    for dataset_dir in legacy_dataset_dirs:
        if not dataset_dir.exists():
            continue
        for png in dataset_dir.rglob("*.png"):
            if png.is_file() and (
                png.name.endswith("_by_method.png")
                or png.name.endswith("train_test_loss_by_method.png")
            ):
                png.unlink()


def visualize_pair_results(input_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = discover_rows(input_path)
    if not rows:
        print(f"[VIS] no summary.json under {input_path}")
        return {"runs": 0, "groups": 0, "plots": 0, "loss_curve_plots": 0}

    clear_plot_outputs(rows, output_path)

    run_fields = [
        "task",
        "dataset",
        "model",
        "peer_model",
        "method",
        "seed",
        "metric_key",
        "best_metric",
        "final_val",
        "epochs",
        "warmup_epochs",
        "lambda_imitation",
        "margin",
        "curve_mode",
        "model_idx",
        "run_dir",
        "summary_path",
    ]
    write_csv(output_path / "runs.csv", rows, run_fields)

    agg_rows = aggregate_rows(rows)
    agg_fields = [
        "task",
        "dataset",
        "model",
        "method",
        "metric_key",
        "n_runs",
        "mean_metric",
        "std_metric",
    ]
    write_csv(output_path / "aggregate.csv", agg_rows, agg_fields)

    best_rows = best_methods(agg_rows)
    best_fields = [
        "task",
        "dataset",
        "model",
        "metric_key",
        "direction",
        "best_method",
        "best_mean_metric",
    ]
    write_csv(output_path / "best_methods.csv", best_rows, best_fields)

    n_metric_plots = save_metric_bar_plots(agg_rows, output_path)
    n_curve_plots = save_loss_curve_plots(rows, output_path)

    print(
        f"[VIS] runs={len(rows)} groups={len(agg_rows)} "
        f"metric_plots={n_metric_plots} loss_curve_plots={n_curve_plots}"
    )
    print(f"[VIS] wrote: {output_path}")
    return {
        "runs": len(rows),
        "groups": len(agg_rows),
        "plots": n_metric_plots,
        "loss_curve_plots": n_curve_plots,
    }


def main() -> int:
    args = parse_args()
    visualize_pair_results(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
