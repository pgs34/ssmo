from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PKS_TASK_BY_DATASET = {
    "mnist": "classification",
    "cifar10": "classification",
    "cifar100": "classification",
    "burgers1d": "operator",
    "darcy2d": "operator",
    "navierstokes": "operator",
}

PKS_DIRECTION_BY_TASK = {
    "classification": "maximize",
    "operator": "minimize",
}


def canonicalize_method(summary: dict[str, Any]) -> str:
    raw_method = str(summary.get("method", "")).strip().lower()
    if raw_method in {"independent", "dml", "ssml"}:
        return raw_method

    peer_model = summary.get("peer_model")
    if peer_model in {None, "", "None"}:
        return "independent"

    mean_weight = pd.to_numeric(summary.get("mean_imitation_weight"), errors="coerce")
    active_ratio = pd.to_numeric(summary.get("active_imitation_ratio"), errors="coerce")
    if not pd.isna(mean_weight) and not pd.isna(active_ratio):
        if float(mean_weight) >= 0.999 and float(active_ratio) >= 0.999:
            return "dml"
        return "ssml"
    return raw_method


def parse_curve_cell(text: Any) -> np.ndarray:
    if isinstance(text, np.ndarray):
        return text.astype(float).reshape(-1)
    if isinstance(text, (list, tuple)):
        return np.asarray(text, dtype=float).reshape(-1)
    if text is None:
        return np.asarray([], dtype=float)
    if isinstance(text, float) and np.isnan(text):
        return np.asarray([], dtype=float)

    raw = str(text).strip()
    if not raw or raw.lower() == "nan":
        return np.asarray([], dtype=float)

    try:
        value = ast.literal_eval(raw)
    except Exception:
        cleaned = raw.replace("\n", " ").strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        cleaned = cleaned.replace(",", " ")
        arr = np.fromstring(cleaned, sep=" ")
        if arr.size == 0 and cleaned:
            raise ValueError(f"Could not parse curve cell: {raw[:120]}...")
        return arr.astype(float)

    return np.asarray(value, dtype=float).reshape(-1)


def load_pks_results(root: str | Path) -> pd.DataFrame:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"PKS result root not found: {base}")

    csv_paths = sorted(
        path
        for path in base.rglob("experiment_results_*.csv")
        if ".ipynb_checkpoints" not in path.parts
    )
    if not csv_paths:
        raise FileNotFoundError(f"No PKS CSV files found under: {base}")

    rows: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        dataset = csv_path.parent.name.lower()
        task = PKS_TASK_BY_DATASET.get(dataset)
        if task is None:
            raise ValueError(f"Unsupported PKS dataset folder: {dataset}")

        frame = pd.read_csv(csv_path)
        suffix = csv_path.stem.replace("experiment_results_", "", 1)

        if {"best", "train_curve", "test_curve"}.issubset(frame.columns):
            model = suffix.lower()
            for seed, record in frame.iterrows():
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "source": "pks",
                        "mode": "single_baseline",
                        "model": model,
                        "peer_model": None,
                        "seed": int(seed),
                        "metric_value": float(record["best"]),
                        "metric_direction": PKS_DIRECTION_BY_TASK[task],
                        "train_curve": parse_curve_cell(record["train_curve"]),
                        "test_curve": parse_curve_cell(record["test_curve"]),
                        "source_path": str(csv_path),
                        "raw_config": suffix.lower(),
                    }
                )
            continue

        pair_columns = {
            "best1",
            "train_curve1",
            "test_curve1",
            "best2",
            "train_curve2",
            "test_curve2",
        }
        if pair_columns.issubset(frame.columns):
            parts = suffix.lower().split("_")
            if len(parts) != 2:
                raise ValueError(f"Expected pair PKS file name '<model1>_<model2>', got: {suffix}")
            model1, model2 = parts
            for seed, record in frame.iterrows():
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "source": "pks",
                        "mode": "pair_ssml",
                        "model": model1,
                        "peer_model": model2,
                        "seed": int(seed),
                        "metric_value": float(record["best1"]),
                        "metric_direction": PKS_DIRECTION_BY_TASK[task],
                        "train_curve": parse_curve_cell(record["train_curve1"]),
                        "test_curve": parse_curve_cell(record["test_curve1"]),
                        "source_path": str(csv_path),
                        "raw_config": suffix.lower(),
                    }
                )
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "source": "pks",
                        "mode": "pair_ssml",
                        "model": model2,
                        "peer_model": model1,
                        "seed": int(seed),
                        "metric_value": float(record["best2"]),
                        "metric_direction": PKS_DIRECTION_BY_TASK[task],
                        "train_curve": parse_curve_cell(record["train_curve2"]),
                        "test_curve": parse_curve_cell(record["test_curve2"]),
                        "source_path": str(csv_path),
                        "raw_config": suffix.lower(),
                    }
                )
            continue

        raise ValueError(f"Unsupported PKS CSV schema: {csv_path}")

    return pd.DataFrame(rows)


def load_run_tree(root: str | Path) -> pd.DataFrame:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Run-tree root not found: {base}")

    summary_paths = sorted(base.rglob("summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary.json files found under: {base}")

    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        curve_path = summary_path.with_name("curves.npz")
        epoch_metrics_path = summary_path.with_name("epoch_metrics.jsonl")

        peer_model = summary.get("peer_model")
        if peer_model in {"", "None"}:
            peer_model = None

        loss_name = (
            summary.get("regression_imitation_loss")
            or summary.get("classification_imitation_loss")
        )

        row = {
            **summary,
            "task": summary.get("task"),
            "dataset": summary.get("dataset"),
            "model": summary.get("model"),
            "peer_model": peer_model,
            "method": canonicalize_method(summary),
            "loss_name": loss_name,
            "best_metric": summary.get("best_metric"),
            "final_metric": summary.get("final_metric"),
            "curve_path": str(curve_path) if curve_path.exists() else None,
            "epoch_metrics_path": str(epoch_metrics_path) if epoch_metrics_path.exists() else None,
            "summary_path": str(summary_path),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def load_curve_file(path: str | Path) -> dict[str, np.ndarray]:
    curve_path = Path(path)
    if not curve_path.exists():
        raise FileNotFoundError(f"Curve file not found: {curve_path}")

    with np.load(curve_path) as data:
        return {name: np.asarray(data[name]).reshape(-1) for name in data.files}


def load_epoch_metrics(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()

    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Epoch metrics file not found: {metrics_path}")

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)
