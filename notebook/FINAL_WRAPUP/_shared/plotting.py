from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = {
    "independent": 0,
    "dml": 2,
    "ssml": 3,
}

DATASET_ORDER = {
    "mnist": 0,
    "cifar10": 1,
    "cifar100": 2,
    "burgers1d": 3,
    "darcy2d": 4,
    "navierstokes": 5,
    "etth1": 6,
    "electricity": 7,
    "weather": 8,
}

METHOD_COLORS = {
    "independent": "#4c566a",
    "dml": "#5e81ac",
    "ssml": "#a3be8c",
}

MODEL_LABELS = {
    "cnn": "CNN",
    "mlp": "MLP",
    "fno": "FNO",
    "deeponet": "DeepONet",
    "gnot": "GNOT",
    "dlinear": "DLinear",
    "transformer": "Transformer",
    "vit_b16": "ViT-B/16",
    "resnet18": "ResNet-18",
}

DATASET_LABELS = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "burgers1d": "Burgers-1D",
    "darcy2d": "Darcy-2D",
    "navierstokes": "Navier-Stokes",
    "etth1": "ETTh1",
    "electricity": "Electricity",
    "weather": "Weather",
}


def apply_report_style() -> None:
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.figsize": (10, 5),
            "figure.dpi": 120,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def pretty_model(name: str | None) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "None"
    return MODEL_LABELS.get(str(name), str(name))


def pretty_method(name: str | None) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "None"
    key = str(name)
    if key == "dml":
        return "DML"
    if key == "ssml":
        return "SSML"
    return str(key).capitalize()


def pretty_dataset(name: str | None) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "None"
    return DATASET_LABELS.get(str(name), str(name))


def pretty_pair(model: str | None, peer_model: str | None) -> str:
    if peer_model is None or (isinstance(peer_model, float) and pd.isna(peer_model)):
        return pretty_model(model)
    return f"{pretty_model(model)} + {pretty_model(peer_model)}"


def slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")
    return normalized or "artifact"


def _artifact_root() -> Path:
    return Path(__file__).resolve().parent.parent


def save_figure(fig: plt.Figure, task: str, name: str, dpi: int = 180) -> Path:
    out_dir = _artifact_root() / "figures" / slugify(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(name)}.png"
    fig.savefig(out_path, dpi=dpi)
    return out_path


def save_table(df: pd.DataFrame, task: str, name: str) -> Path:
    out_dir = _artifact_root() / "tables" / slugify(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(name)}.csv"
    df.to_csv(out_path, index=False)
    return out_path
