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

def find_final_code_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "config" / "final_wrapup_manifest.yaml").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("Could not locate final_code root from the current working directory.")

FINAL_CODE_ROOT = find_final_code_root(Path.cwd().resolve())
NOTEBOOK_ROOT = FINAL_CODE_ROOT / "notebooks"
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

from _shared.final_wrapup import (
    build_classification_table,
    build_main_results_table,
    build_operator_table,
    build_time_series_table,
    export_results_summary,
    final_code_root,
    inventory_frame,
    load_manifest,
    plot_final_classification,
    plot_final_operator,
    plot_final_time_series,
)

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)

FINAL_CODE_ROOT
"""


def build_index_notebook():
    cells = [
        md(
            """
            # final_code Index

            이 notebook들은 `final_code/results/` 아래에서 새로 생성한 run만 읽습니다.
            외부 `results/`는 보지 않습니다.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            manifest = load_manifest()
            inventory = inventory_frame()
            missing = inventory[inventory["missing"] > 0].reset_index(drop=True)

            display(Markdown(f"**Root**: `{final_code_root()}`"))
            display(inventory)
            display(Markdown("**Missing only**"))
            missing
            """
        ),
        md(
            """
            ## Notebook order

            - `01_time_series_wrapup.ipynb`
            - `02_operator_wrapup.ipynb`
            - `03_classification_wrapup.ipynb`
            - `04_final_wrapup.ipynb`
            """
        ),
    ]
    return notebook(cells, "final_code Index")


def build_time_series_notebook():
    cells = [
        md(
            """
            # Time-series Wrap-up

            `Weather`, `Electricity`, `ETTh1`를 정리합니다.
            `SEED_OVERRIDES`, `WINDOWS`, `LEGEND_LABELS`, export toggle을 직접 바꿔가며 확인할 수 있습니다.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            SEED_OVERRIDES = {
                # "etth1.independent": [0, 1, 2, 3, 4, 5],
                # "etth1.ssml": [0, 1, 2],
            }
            WINDOWS = {
                "etth1": (1, 20),
                "weather": (1, 20),
                "electricity": (1, 20),
            }
            LEGEND_LABELS = {
                "independent": "Independent",
                "dml": "DML",
                "ssml": "SSML",
            }
            SHOW_BANDS = False
            EXPORT_TABLE = False
            EXPORT_FIGURE = False
            """
        ),
        code(
            """
            time_series_table = build_time_series_table(seed_overrides=SEED_OVERRIDES, export=EXPORT_TABLE)
            display(time_series_table)

            fig = plot_final_time_series(
                seed_overrides=SEED_OVERRIDES,
                windows=WINDOWS,
                legend_labels=LEGEND_LABELS,
                show_bands=SHOW_BANDS,
                export=EXPORT_FIGURE,
            )
            plt.show()
            """
        ),
    ]
    return notebook(cells, "Time-series Wrap-up")


def build_operator_notebook():
    cells = [
        md(
            """
            # Operator Wrap-up

            `Burgers`, `Darcy`를 정리합니다.
            외부 plot은 전체 흐름, inset은 zoom 구간입니다.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            SEED_OVERRIDES = {}
            LATE_WINDOWS = {
                "burgers": (120, 180),
                "darcy": (110, 150),
            }
            FULL_WINDOWS = {
                "burgers": (1, 180),
                "darcy": (1, 150),
            }
            LEGEND_LABELS = {
                "independent": "Independent",
                "dml": "DML",
                "ssml": "SSML",
            }
            SHOW_BANDS = False
            EXPORT_TABLE = False
            EXPORT_FIGURE = False
            """
        ),
        code(
            """
            operator_table = build_operator_table(seed_overrides=SEED_OVERRIDES, export=EXPORT_TABLE)
            display(operator_table)

            fig = plot_final_operator(
                seed_overrides=SEED_OVERRIDES,
                windows=LATE_WINDOWS,
                full_windows=FULL_WINDOWS,
                legend_labels=LEGEND_LABELS,
                show_bands=SHOW_BANDS,
                export=EXPORT_FIGURE,
            )
            plt.show()
            """
        ),
    ]
    return notebook(cells, "Operator Wrap-up")


def build_classification_notebook():
    cells = [
        md(
            """
            # Classification Wrap-up

            `CIFAR-10`, `CIFAR-100 cifarstem follow-up` 결과를 정리합니다.
            `CIFAR-100`은 기본으로 epoch 50까지만 보도록 되어 있습니다.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            SEED_OVERRIDES = {}
            WINDOWS = {
                "cifar10": (1, 100),
                "cifar100": (1, 50),
            }
            TAIL_WINDOWS = {
                "cifar10": (84, 100),
                "cifar100": (40, 50),
            }
            LEGEND_LABELS = {
                "independent": "Independent",
                "dml": "DML",
                "ssml": "SSML",
            }
            SHOW_BANDS = False
            EXPORT_TABLE = False
            EXPORT_FIGURE = False
            """
        ),
        code(
            """
            classification_table = build_classification_table(seed_overrides=SEED_OVERRIDES, export=EXPORT_TABLE)
            display(classification_table)

            fig = plot_final_classification(
                seed_overrides=SEED_OVERRIDES,
                windows=WINDOWS,
                tail_windows=TAIL_WINDOWS,
                legend_labels=LEGEND_LABELS,
                show_bands=SHOW_BANDS,
                export=EXPORT_FIGURE,
            )
            plt.show()
            """
        ),
    ]
    return notebook(cells, "Classification Wrap-up")


def build_final_notebook():
    cells = [
        md(
            """
            # Final Wrap-up

            최종 summary table과 세 개의 wrap-up figure를 한 번에 정리하는 notebook입니다.
            필요하면 window, seed subset, legend를 이 notebook 안에서 바로 바꾸면 됩니다.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            SEED_OVERRIDES = {}

            TIME_WINDOWS = {
                "etth1": (1, 20),
                "weather": (1, 20),
                "electricity": (1, 20),
            }
            OPERATOR_WINDOWS = {
                "burgers": (120, 180),
                "darcy": (110, 150),
            }
            OPERATOR_FULL_WINDOWS = {
                "burgers": (1, 180),
                "darcy": (1, 150),
            }
            CLASS_WINDOWS = {
                "cifar10": (1, 100),
                "cifar100": (1, 50),
            }
            CLASS_TAIL_WINDOWS = {
                "cifar10": (84, 100),
                "cifar100": (40, 50),
            }
            LEGEND_LABELS = {
                "independent": "Independent",
                "dml": "DML",
                "ssml": "SSML",
            }

            SHOW_BANDS = False
            EXPORT_TABLE = True
            EXPORT_FIGURES = True
            EXPORT_RESULTS_SUMMARY = True
            """
        ),
        code(
            """
            final_table = build_main_results_table(seed_overrides=SEED_OVERRIDES, export=EXPORT_TABLE)
            display(final_table)
            """
        ),
        code(
            """
            fig_time = plot_final_time_series(
                seed_overrides=SEED_OVERRIDES,
                windows=TIME_WINDOWS,
                legend_labels=LEGEND_LABELS,
                show_bands=SHOW_BANDS,
                export=EXPORT_FIGURES,
            )
            plt.show()

            fig_operator = plot_final_operator(
                seed_overrides=SEED_OVERRIDES,
                windows=OPERATOR_WINDOWS,
                full_windows=OPERATOR_FULL_WINDOWS,
                legend_labels=LEGEND_LABELS,
                show_bands=SHOW_BANDS,
                export=EXPORT_FIGURES,
            )
            plt.show()

            fig_classification = plot_final_classification(
                seed_overrides=SEED_OVERRIDES,
                windows=CLASS_WINDOWS,
                tail_windows=CLASS_TAIL_WINDOWS,
                legend_labels=LEGEND_LABELS,
                show_bands=SHOW_BANDS,
                export=EXPORT_FIGURES,
            )
            plt.show()
            """
        ),
        code(
            """
            if EXPORT_RESULTS_SUMMARY:
                summary_path = export_results_summary()
                display(Markdown(f"Results summary exported: `{summary_path.relative_to(FINAL_CODE_ROOT)}`"))
            """
        ),
    ]
    return notebook(cells, "Final Wrap-up")


def main():
    outputs = {
        "00_index.ipynb": build_index_notebook(),
        "01_time_series_wrapup.ipynb": build_time_series_notebook(),
        "02_operator_wrapup.ipynb": build_operator_notebook(),
        "03_classification_wrapup.ipynb": build_classification_notebook(),
        "04_final_wrapup.ipynb": build_final_notebook(),
    }
    for relative_name, nb in outputs.items():
        out_path = ROOT / relative_name
        nbf.write(nb, out_path)
        print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
