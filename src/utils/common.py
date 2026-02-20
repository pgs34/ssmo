from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_dir(
    base_dir: str | Path,
    task: str,
    dataset: str,
    model: str,
) -> Path:
    run_dir = Path(base_dir) / task / dataset / model
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_curves(path: str | Path, **arrays) -> None:
    np_arrays = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez(path, **np_arrays)


def save_live_loss_plot(
    run_dir: str | Path,
    task: str,
    seed: int,
    methods: list[str] | None = None,
) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_root = Path(run_dir)
    if not run_root.exists():
        print(f"[viz] run_dir not found: {run_root}")
        return False

    task = str(task).strip()
    method_pattern = re.compile(
        r"^(?P<model>.+)_(?P<method>independent|naive|dml|studygroup)(?:_(?P<loss>[^/]+))?_seed(?P<seed>\d+)$"
    )
    method_curve_paths: dict[str, Path] = {}
    model_title = ""
    output_path: Path

    # Legacy layout: <dataset>/<model>/<method>/seed<seed>
    if run_root.name == f"seed{seed}":
        model_root = run_root.parent.parent
        if not model_root.exists():
            print(f"[viz] model root not found: {model_root}")
            return False
        candidate_methods = [p.name for p in model_root.iterdir() if p.is_dir()]
        for method in candidate_methods:
            method_curve_paths[method] = model_root / method / f"seed{seed}" / "curves.npz"
        model_title = model_root.name
        output_path = model_root / f"seed{seed}__train_test_loss_by_method.png"

    # Flat layout: <dataset>/<model>_<method>_seed<seed>
    else:
        m = method_pattern.match(run_root.name)
        if m is None:
            print(f"[viz] unsupported run_dir layout: {run_root}")
            return False
        model_name = m.group("model")
        loss_name = m.group("loss")
        dataset_root = run_root.parent
        if not dataset_root.exists():
            print(f"[viz] dataset root not found: {dataset_root}")
            return False
        candidate_methods = []
        for p in dataset_root.iterdir():
            if not p.is_dir():
                continue
            mm = method_pattern.match(p.name)
            if mm is None:
                continue
            if mm.group("model") != model_name:
                continue
            if int(mm.group("seed")) != seed:
                continue
            if mm.group("loss") != loss_name:
                continue
            method = mm.group("method")
            candidate_methods.append(method)
            method_curve_paths[method] = p / "curves.npz"
        model_title = model_name
        if loss_name:
            output_path = dataset_root / f"{model_name}_seed{seed}_{loss_name}_train_test_loss_by_method.png"
        else:
            output_path = dataset_root / f"{model_name}_seed{seed}__train_test_loss_by_method.png"

    if methods is None:
        methods = candidate_methods
    else:
        methods = [m for m in methods if m in candidate_methods]

    method_order = {"independent": 0, "naive": 1, "dml": 2, "studygroup": 3}
    methods.sort(key=lambda m: (method_order.get(m, 99), m))

    if not methods:
        print(f"[viz] no method directories found for {run_root}")
        return False

    task_train_keys = {
        "classification": ["train_loss", "train_total", "train_loss1", "train_total1"],
        "segmentation": ["train_loss", "train_total", "train_loss1", "train_total1"],
        "operator": ["train_mse", "train_total", "train_loss", "train_mse1"],
        "time_series": ["train_mse", "train_total", "train_loss", "train_mse1"],
    }
    task_val_keys = {
        "classification": ["val_loss", "val_mse", "val_mae"],
        "segmentation": ["val_loss", "val_mse", "val_mae"],
        "operator": ["val_mse", "val_loss", "val_mae"],
        "time_series": ["val_mse", "val_loss", "val_mae"],
    }

    t = task.lower()
    train_keys = task_train_keys.get(t, ["train_loss", "train_total", "train_mse", "train_total1"])
    val_keys = task_val_keys.get(t, ["val_loss", "val_mse", "val_mae"])

    def _load_curve(path: Path, keys: list[str]) -> np.ndarray | None:
        if not path.exists():
            return None
        with np.load(path) as data:
            for key in keys:
                if key in data:
                    arr = np.asarray(data[key]).reshape(-1)
                    if arr.size > 0:
                        return arr
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), squeeze=False)
    ax_train = axes[0][0]
    ax_val = axes[0][1]

    cmap = plt.get_cmap("tab10")
    has_curve = False
    linestyle_cycle = ["-", "--", ":", "-."]
    marker_cycle = ["o", "s", "^", "v"]

    for i, method in enumerate(methods):
        curve_path = method_curve_paths.get(method)
        if curve_path is None:
            continue
        if not curve_path.exists():
            continue

        train_curve = _load_curve(curve_path, train_keys)
        val_curve = _load_curve(curve_path, val_keys)
        if train_curve is None and val_curve is None:
            continue

        color = cmap(i % 10)
        line_style = linestyle_cycle[i % len(linestyle_cycle)]
        marker = marker_cycle[i % len(marker_cycle)]
        if train_curve is not None and train_curve.size > 0:
            ax_train.plot(
                np.arange(1, train_curve.size + 1),
                train_curve,
                label=method,
                color=color,
                linestyle=line_style,
                marker=marker,
                markevery=max(1, train_curve.size // 8),
                linewidth=1.8,
                markersize=3.2,
            )
            has_curve = True

        if val_curve is not None and val_curve.size > 0:
            ax_val.plot(
                np.arange(1, val_curve.size + 1),
                val_curve,
                label=method,
                color=color,
                linestyle=line_style,
                marker=marker,
                markevery=max(1, val_curve.size // 8),
                linewidth=1.8,
                markersize=3.2,
            )
            has_curve = True

    if not has_curve:
        plt.close(fig)
        print(f"[viz] no usable curves found for {run_root}")
        return False

    ax_train.set_title("Train")
    ax_val.set_title("Test")
    ax_train.set_xlabel("epoch")
    ax_val.set_xlabel("epoch")
    ax_train.set_ylabel("loss")
    ax_val.set_ylabel("loss")
    ax_train.grid(alpha=0.3)
    ax_val.grid(alpha=0.3)
    ax_train.legend(fontsize=8)
    ax_val.legend(fontsize=8)

    if run_root.name != f"seed{seed}":
        mm = method_pattern.match(run_root.name)
        loss_name = mm.group("loss") if mm is not None else None
    else:
        loss_name = None

    if loss_name:
        fig.suptitle(f"{model_title} | seed{seed} | {task} | {loss_name}", fontsize=11)
    else:
        fig.suptitle(f"{model_title} | seed{seed} | {task}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
