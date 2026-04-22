from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .io import load_curve_file, load_epoch_metrics
from .plotting import METHOD_COLORS, apply_report_style, pretty_dataset, pretty_method

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "config" / "final_wrapup_manifest.yaml"
METHOD_ORDER = ("independent", "dml", "ssml")
TASK_LABELS = {
    "time_series": "Time-series",
    "operator": "Operator",
    "classification": "Classification",
}

TASK_VARIATION_NOTES = r"""
## 4. Task별 변형 포인트

이 절은 논문 본문에서는 세 task를 하나의 관점으로 묶어 설명하고, dataset별 세부 해석은 appendix 성격의 상세 메모로 분리하는 방식이 가장 적절하다. 핵심은 세 task가 서로 다른 문제처럼 보이더라도, 실제로는 **동일한 SSML 원리를 서로 다른 selection unit 위에 올린 경우**라는 점이다. 두 모델을 `f_i`, `f_j`라고 하고, selection의 기본 단위를 `z`라고 두면 SSML의 공통 형태는 다음과 같다.

$$
\ell_i^{\mathrm{sup}}(z)=\mathcal L_{\mathrm{task}}(f_i(z), y(z)),
\qquad
g_{i\leftarrow j}(z)=\mathbf 1\!\left[\ell_j^{\mathrm{sup}}(z)+m<\ell_i^{\mathrm{sup}}(z)\right],
$$

$$
\mathcal L_i
=
\mathbb E_z\!\left[\ell_i^{\mathrm{sup}}(z)\right]
+\lambda\,\mathbb E_z\!\left[g_{i\leftarrow j}(z)\,\ell^{\mathrm{imit}}\!\left(f_i(z),\operatorname{sg}(f_j(z))\right)\right].
$$

여기서 supervised term이 항상 유지된다는 점이 중요하다. 즉, SSML은 peer output으로 ground truth를 대체하는 방법이 아니라, **peer가 더 잘하는 단위에서만 추가적인 imitation signal을 더하는 방법**이다. 이 관점에서 보면 time-series forecasting, operator learning, image classification은 서로 다른 task라기보다, 동일한 식 위에서 `z`만 다르게 정의한 경우라고 정리할 수 있다.

$$
z=
\begin{cases}
(n,h), & \text{time-series forecasting}\\
(n,q), & \text{operator learning}\\
n, & \text{image classification}
\end{cases}
$$

여기서 `n`은 sample index, `h`는 forecast horizon, `q \in \Omega`는 spatial location을 뜻한다. 이 정의에 따라 supervised loss는 각 task에 맞게 바뀌지만 구조는 동일하다. Forecasting에서는 $\ell_i^{\mathrm{sup}}(n,h)=\left\|\hat y_i^{(n)}(h)-y^{(n)}(h)\right\|_2^2$, operator learning에서는 $\ell_i^{\mathrm{sup}}(n,q)=\left\|\hat u_i^{(n)}(q)-u^{(n)}(q)\right\|_2^2$, classification에서는 $\ell_i^{\mathrm{sup}}(n)=\mathrm{CE}\!\left(p_i(x_n), y_n\right)$로 쓸 수 있다. 따라서 세 domain을 하나로 묶는 가장 좋은 설명은 다음과 같다. **SSML은 prediction 구조가 sequence이든, field이든, class probability이든 상관없이, target을 구성하는 더 작은 단위에서 peer superiority를 판정하고 그 위치에만 imitation을 여는 일반 원리**라는 것이다.

이때 task별 차이는 원리 자체가 아니라 selection unit의 성격에서 생긴다. Forecasting에서는 horizon마다 우위가 달라지므로 gate가 시간축을 따라 희소하게 열려야 하고, operator learning에서는 공간 해상도에 따라 gate의 granularity가 달라진다. Classification에서는 selection unit이 sample로 가장 단순하지만, homogeneous setting에서는 gate가 지나치게 조밀해지거나 특정 class에 편중될 수 있으므로 filtering과 balancing이 더 중요해진다. 결국 세 task는 모두 $\text{same supervised backbone}+\text{peer-better mask}+\text{task-specific control of mask density}$라는 공통 구조로 읽을 수 있다. 논문 본문에서는 이 unified view를 먼저 제시한 뒤, task별 디테일은 appendix 성격의 상세 메모로 분리하는 편이 가장 자연스럽다.

### 4.1 Appendix-style Detailed Notes

아래 내용은 본문보다는 appendix나 supplementary note 쪽에 가까운 상세 메모로 두는 편이 좋다. 본문에서는 unified formulation과 대표 결과만 남기고, 여기서는 dataset별로 어떤 세부 변형이 실제로 중요했는지 bullet point로 정리한다.

#### Time-series forecasting

- 공통 구조:
  - selection unit은 `z=(n,h)`이며, 각 horizon `h`에서 peer가 더 작은 error를 보일 때만 imitation을 연다.
  - forecasting에서는 gate density가 높아지면 후반 drift가 커지므로, `\lambda_t` schedule과 activation sparsity가 사실상 방법의 일부처럼 작동한다.
- `Weather`:
  - 가장 canonical한 forecasting win이다.
  - 비교적 단순한 horizon-wise selective gate만으로 `Independent`와 `DML`을 모두 넘는다.
  - task-specific stabilization을 많이 추가하지 않아도 hard selective imitation이 충분히 작동한 경우로 해석할 수 있다.
- `Electricity`:
  - best checkpoint 기준으로는 매우 강한 개선이 나오지만, 후반 epoch에서 validation curve가 다시 상승하는 late drift가 반복된다.
  - strongest row인 `corrective_v1 / corr_gate64_l15_sp5e4`는 "peer imitation"이라기보다 취약 horizon만 보정하는 corrective guidance에 가깝다.
  - 후속 `late handoff`, `budget cap`, `sparser guidance`는 모두 활성 imitation 비율 $\rho_t=\frac{1}{NH}\sum_{n,h} g_{i\leftarrow j}^{(t)}(n,h)$를 낮추고 drift를 줄이기 위한 변형으로 읽을 수 있다.
- `ETTh1`:
  - 기본 horizon-wise gate만으로는 충분하지 않아 `teacher fine-tuning`, `reweight`, `handoff`, `router`, `pair-deployed reevaluation`이 추가되었다.
  - 최종 best 값은 pure student improvement라기보다 stronger branch selection의 영향이 크다.
  - 따라서 ETTh1는 clean win이 아니라, selective teaching과 selective deployment를 구분해야 함을 보여주는 diagnostic case로 두는 것이 적절하다.

#### Operator learning

- 공통 구조:
  - selection unit은 `z=(n,q)`이며, spatial location `q` 혹은 spatial block에서 peer superiority를 판정한다.
  - `sample`, `coarse`, `element` 변형은 서로 다른 방법이라기보다, 공간 해상도를 다르게 잡은 동일한 SSML의 구현으로 보는 편이 맞다.
- `Burgers`:
  - 메인 스토리는 from-scratch co-training보다 strong baseline 위에서 selective peer polishing을 수행했다는 점에 있다.
  - `operator_burgers_polish_aggressive_v4`에서 cosine decay 기반 polish regime이 도입되며 성능이 크게 개선되었다.
  - `relay`, `coarse`, `sample`, `element`를 비교한 결과 best는 `cos_relay_full_l0012_s20_70_40_sample_lr4e4`였다.
  - 해석상 핵심은 "peer superiority를 무조건 더 미세하게 보는 것"보다, 충분히 informative한 공간 단위에서 안정적으로 여는 것이 더 중요했다는 점이다.
- `Darcy`:
  - canonical `operator_ssml_tuned_v1` setting 자체에서 이미 clean한 ordering이 나온다.
  - 별도의 복잡한 polish narrative 없이도 field-level selective correction이 자연스럽게 작동한 사례로 설명할 수 있다.
  - 이후 `student lift`, `relay` 계열은 메인 claim이라기보다 weak-model rescue에 가까운 follow-up으로 두는 편이 적절하다.

#### Image classification

- 공통 구조:
  - selection unit은 `z=n`인 sample-wise gate다.
  - 다만 homogeneous image classification에서는 gate가 너무 쉽게 조밀해지거나 특정 class에 편중될 수 있어, filtering과 balancing이 중요해진다.
  - 개념적으로는
    $$
    \tilde g_{i\leftarrow j}(n)
    =
    g_{i\leftarrow j}(n)\cdot
    \mathbf 1[n\in \mathcal T_k]\cdot
    \mathbf 1[n\in \mathcal D]\cdot
    w_c(n)
    $$
    와 같이 top-k, disagreement filter, class balancing을 곱해준 형태로 이해할 수 있다.
- `CIFAR-10`:
  - classification domain에서 가장 직관적인 positive example이다.
  - 복잡한 filtering 없이도 sample-wise hard selective imitation만으로 `Independent`와 `DML`을 모두 넘는다.
  - long DML rerun 이후에도 ordering이 유지되므로 main paper의 clean classification row로 쓰기 좋다.
- `CIFAR-100`:
  - sample-wise gate만으로는 부족한 hard homogeneous setting이다.
  - strict128 원판과 후속에서 `augfilter`, `top-k`, `disagreement-aware filtering`, `scheduled complement`, `best checkpoint pool` 등이 모두 gate density와 class imbalance를 제어하기 위한 장치로 들어갔다.
  - aggressive follow-up 일부는 `lr / weight_decay` 전달 버그가 확인되었으므로 diagnostic으로만 유지하는 것이 맞다.
  - `cifarstem_followup_v1`은 strict128 원판과 동일한 SSML 논리를 backbone bottleneck이 덜한 setting에서 다시 검증한 구조적 follow-up으로 설명하는 것이 가장 자연스럽다.
"""


def final_code_root() -> Path:
    return ROOT


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _artifact_dir(kind: str) -> Path:
    out_dir = ROOT / "artifacts" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def export_figure(fig: plt.Figure, filename: str, dpi: int = 180) -> Path:
    out_path = _artifact_dir("figures") / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def export_table(df: pd.DataFrame, filename: str) -> Path:
    out_path = _artifact_dir("tables") / filename
    df.to_csv(out_path, index=False)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_dataframe_to_markdown(df), encoding="utf-8")
    return out_path


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    rows = [[_markdown_escape(value) for value in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def _markdown_escape(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _latest_dispatch_run_id() -> str:
    latest_path = ROOT / "results" / "_dispatch" / "latest_run_id.txt"
    if latest_path.exists():
        run_id = latest_path.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id
    dispatch_root = ROOT / "results" / "_dispatch"
    runs = sorted(path.name for path in dispatch_root.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No dispatch runs found under {dispatch_root}")
    return runs[-1]


def export_results_summary(run_id: str | None = None, filename: str = "Results_Summary.md") -> Path:
    """Write the final Markdown summary from final_code-local notebook artifacts."""
    run_id = run_id or _latest_dispatch_run_id()
    dispatch_dir = ROOT / "results" / "_dispatch" / run_id
    plan_path = dispatch_dir / "plan.tsv"
    events_path = dispatch_dir / "events.tsv"
    table_md_path = ROOT / "artifacts" / "tables" / "final_wrapup_summary.md"
    table_csv_path = ROOT / "artifacts" / "tables" / "final_wrapup_summary.csv"
    figure_paths = {
        "Time-Series": ROOT / "artifacts" / "figures" / "final_wrapup_time_series.png",
        "Operator": ROOT / "artifacts" / "figures" / "final_wrapup_operator.png",
        "Classification": ROOT / "artifacts" / "figures" / "final_wrapup_classification.png",
    }

    required_paths = [plan_path, events_path, table_md_path, table_csv_path, *figure_paths.values()]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(_relative_path(path) for path in missing)
        raise FileNotFoundError(f"Missing final_code-local summary inputs:\n{missing_text}")

    plan = pd.read_csv(plan_path, sep="\t")
    events = pd.read_csv(events_path, sep="\t")
    table_md = table_md_path.read_text(encoding="utf-8").strip()

    complete_rows = events[events["event"] == "run_complete"]
    complete_status = "unknown"
    complete_time = "unknown"
    if not complete_rows.empty:
        last_complete = complete_rows.iloc[-1]
        complete_status = str(last_complete.get("status", "unknown"))
        complete_time = str(last_complete.get("ts", "unknown"))

    plan_rows = [
        "| Target | Host | GPU | Experiments |",
        "| --- | --- | --- | --- |",
    ]
    for target, group in plan.groupby("target", sort=False):
        host = str(group.iloc[0]["host"])
        gpu = str(group.iloc[0]["gpu"])
        experiments = ", ".join(f"`{name}`" for name in group["experiment"].astype(str).tolist())
        plan_rows.append(f"| `{target}` | `{host}` | `{gpu}` | {experiments} |")

    figure_lines: list[str] = []
    for label, path in figure_paths.items():
        rel_path = _relative_path(path)
        figure_lines.extend(
            [
                f"### {label}",
                "",
                f"Notebook artifact: `{rel_path}`",
                "",
                f"![{label}]({rel_path})",
                "",
            ]
        )

    artifact_rows = [
        "| Artifact | Path |",
        "| --- | --- |",
        f"| Summary table CSV | `{_relative_path(table_csv_path)}` |",
        f"| Summary table Markdown | `{_relative_path(table_md_path)}` |",
    ]
    for label, path in figure_paths.items():
        artifact_rows.append(f"| {label} figure | `{_relative_path(path)}` |")

    text = "\n".join(
        [
            "# Results Summary",
            "",
            "## Scope",
            "",
            "This document is generated from files inside this `final_code/` directory only.",
            "",
            f"- Source run directory: `{_relative_path(ROOT / 'results')}/`",
            f"- Source notebook: `{_relative_path(ROOT / 'notebooks' / '04_final_wrapup.ipynb')}`",
            f"- Notebook table artifact: `{_relative_path(table_md_path)}`",
            f"- Notebook figure artifacts: `{_relative_path(ROOT / 'artifacts' / 'figures')}/`",
            f"- Dispatch record: `{_relative_path(dispatch_dir)}/`",
            "",
            "The table and figures below are notebook-exported artifacts from `final_code/`.",
            "",
            "## Run Record",
            "",
            f"Dispatch run: `{run_id}`",
            "",
            *plan_rows,
            "",
            f"Run-complete status from `{_relative_path(events_path)}`: `{complete_status}` at `{complete_time}`.",
            "",
            "## Final Table",
            "",
            f"Generated by `notebooks/04_final_wrapup.ipynb`; artifact: `{_relative_path(table_md_path)}`",
            "",
            table_md,
            "",
            f"CSV artifact: `{_relative_path(table_csv_path)}`",
            "",
            TASK_VARIATION_NOTES.strip(),
            "",
            "## Figures",
            "",
            *figure_lines,
            "## Artifact Inventory",
            "",
            *artifact_rows,
            "",
            "## Regeneration",
            "",
            "Open `notebooks/04_final_wrapup.ipynb`, set:",
            "",
            "```python",
            "EXPORT_TABLE = True",
            "EXPORT_FIGURES = True",
            "EXPORT_RESULTS_SUMMARY = True",
            "```",
            "",
            "Then run the notebook from the first cell.",
            "",
        ]
    )

    out_path = ROOT / filename
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _run_dir_from_template(template: str, seed: int) -> Path:
    return ROOT / template.format(seed=seed)


@lru_cache(maxsize=None)
def _load_run_bundle_cached(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    curve_path = run_dir / "curves.npz"
    epoch_metrics_path = run_dir / "epoch_metrics.jsonl"
    curves = load_curve_file(curve_path) if curve_path.exists() else {}
    epoch_metrics = load_epoch_metrics(epoch_metrics_path) if epoch_metrics_path.exists() else pd.DataFrame()
    return {
      "run_dir": run_dir,
      "summary": summary,
      "curves": curves,
      "epoch_metrics": epoch_metrics,
    }


def _load_run_bundle(run_dir: Path) -> dict[str, Any]:
    return dict(_load_run_bundle_cached(str(run_dir.resolve())))


def _to_numeric_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values.astype(float).reshape(-1)
    else:
        arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return arr.reshape(-1)


def _curve_from_run(run: dict[str, Any], keys: list[str]) -> np.ndarray:
    curves = run["curves"]
    for key in keys:
        if key in curves:
            arr = _to_numeric_array(curves[key])
            if arr.size:
                return arr
    epoch_metrics = run["epoch_metrics"]
    for key in keys:
        if not epoch_metrics.empty and key in epoch_metrics.columns:
            arr = _to_numeric_array(epoch_metrics[key])
            if arr.size:
                return arr
    return np.asarray([], dtype=float)


def _best_branch_curve(run: dict[str, Any]) -> np.ndarray:
    epoch_metrics = run["epoch_metrics"]
    if not epoch_metrics.empty and "val_mse_reported" in epoch_metrics.columns:
        reported = _to_numeric_array(epoch_metrics["val_mse_reported"])
        if reported.size:
            return reported

    primary = _curve_from_run(run, ["val_mse1", "val_mse_guided", "val_mse"])
    peer = _curve_from_run(run, ["val_mse2", "peer_val_mse"])
    if primary.size and peer.size:
        min_len = min(len(primary), len(peer))
        return np.minimum(primary[:min_len], peer[:min_len])
    if primary.size:
        return primary
    if peer.size:
        return peer
    return _curve_from_run(run, ["val_mse"])


def _extract_curve(run: dict[str, Any], curve_mode: str) -> np.ndarray:
    if curve_mode == "val_mse":
        return _curve_from_run(run, ["val_mse", "val_mse1"])
    if curve_mode == "best_branch_mse":
        return _best_branch_curve(run)
    if curve_mode == "val_acc_percent":
        acc = _curve_from_run(run, ["val_acc", "val_acc1"])
        if acc.size:
            return acc * 100.0
        return np.asarray([], dtype=float)
    if curve_mode == "val_error_percent":
        acc = _curve_from_run(run, ["val_acc", "val_acc1"])
        if acc.size:
            return (1.0 - acc) * 100.0
        return np.asarray([], dtype=float)
    raise ValueError(f"Unsupported curve mode: {curve_mode}")


def _metric_from_run(run: dict[str, Any], dataset_spec: dict[str, Any], entry: dict[str, Any]) -> float:
    scale = float(dataset_spec.get("metric_scale", 1.0))
    metric_mode = entry.get("metric_mode", "summary_best")
    if metric_mode == "summary_best":
        value = pd.to_numeric(run["summary"].get("best_metric"), errors="coerce")
        return float(value) * scale if not pd.isna(value) else float("nan")
    if metric_mode == "curve_best":
        curve_mode = entry.get("metric_curve_mode", entry.get("curve_mode"))
        curve = _extract_curve(run, curve_mode)
        if curve.size == 0:
            return float("nan")
        if dataset_spec["metric_direction"] == "maximize":
            return float(np.nanmax(curve)) * scale
        return float(np.nanmin(curve)) * scale
    raise ValueError(f"Unsupported metric mode: {metric_mode}")


def _override_seeds(
    domain: str,
    dataset: str,
    method: str,
    default_seeds: list[int],
    seed_overrides: dict[Any, Any] | None,
) -> list[int]:
    if not seed_overrides:
        return [int(seed) for seed in default_seeds]
    for key in (
        (domain, dataset, method),
        f"{domain}.{dataset}.{method}",
        f"{dataset}.{method}",
        method,
    ):
        if key in seed_overrides:
            return [int(seed) for seed in seed_overrides[key]]
    return [int(seed) for seed in default_seeds]


def _entry_runs(
    domain: str,
    dataset: str,
    method: str,
    entry: dict[str, Any],
    seed_overrides: dict[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seeds = _override_seeds(domain, dataset, method, entry.get("seeds", []), seed_overrides)
    for seed in seeds:
        run_dir = _run_dir_from_template(entry["run_template"], seed)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        bundle = _load_run_bundle(run_dir)
        bundle["seed"] = seed
        runs.append(bundle)
    return runs


def _is_better(value: float, best_value: float, direction: str) -> bool:
    if direction == "maximize":
        return value > best_value
    return value < best_value


@lru_cache(maxsize=None)
def _selected_candidate_name(domain: str, dataset: str, method: str) -> str | None:
    manifest = load_manifest()
    dataset_spec = manifest["groups"][domain][dataset]
    method_spec = dataset_spec["methods"][method]
    candidates = method_spec.get("candidates", [])
    best_name = None
    best_missing = None
    best_metric = None
    for candidate in candidates:
        expected = len(candidate.get("seeds", []))
        runs = _entry_runs(domain, dataset, method, candidate, seed_overrides=None)
        metrics = [
            _metric_from_run(run, dataset_spec, {**method_spec, **candidate})
            for run in runs
        ]
        metrics = [value for value in metrics if np.isfinite(value)]
        if not metrics:
            continue
        missing = expected - len(metrics)
        mean_value = float(np.mean(metrics))
        if (
            best_name is None
            or missing < int(best_missing)
            or (missing == best_missing and _is_better(mean_value, float(best_metric), dataset_spec["metric_direction"]))
        ):
            best_name = candidate["name"]
            best_missing = missing
            best_metric = mean_value
    return best_name


def _resolved_entry(domain: str, dataset: str, method: str) -> dict[str, Any]:
    manifest = load_manifest()
    method_spec = dict(manifest["groups"][domain][dataset]["methods"][method])
    if method_spec.get("selection") != "best_baseline":
        return method_spec
    selected_name = _selected_candidate_name(domain, dataset, method)
    candidates = method_spec.pop("candidates", [])
    for candidate in candidates:
        if candidate["name"] == selected_name:
            resolved = dict(method_spec)
            resolved.update(candidate)
            resolved["selected_candidate"] = selected_name
            return resolved
    fallback = dict(method_spec)
    if candidates:
        fallback.update(candidates[0])
    fallback["selected_candidate"] = selected_name
    return fallback


def _stack_curves(curves: list[np.ndarray]) -> np.ndarray:
    prepared = [np.asarray(curve, dtype=float).reshape(-1) for curve in curves if curve is not None and len(curve)]
    if not prepared:
        return np.empty((0, 0), dtype=float)
    min_len = min(len(arr) for arr in prepared)
    return np.vstack([arr[:min_len] for arr in prepared])


def _mean_std_curve(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = _stack_curves(curves)
    if stacked.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def _normalize_window(window_value: Any, fallback: list[int]) -> tuple[int, int]:
    source = fallback if window_value is None else window_value
    if isinstance(source, int):
        return 1, int(source)
    if isinstance(source, (list, tuple)) and len(source) == 2:
        return int(source[0]), int(source[1])
    raise ValueError(f"Unsupported window value: {source}")


def _clip_positive_band(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.clip(mean - std, a_min=1e-12, a_max=None)
    upper = np.clip(mean + std, a_min=1e-12, a_max=None)
    return lower, upper


def _format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "missing"
    if not np.isfinite(std):
        std = 0.0
    scale = max(abs(mean), abs(std))
    if scale and scale < 1e-4:
        return f"{mean:.2e} ± {std:.2e}"
    if scale < 0.01:
        return f"{mean:.6f} ± {std:.6f}"
    if scale < 1.0:
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.2f} ± {std:.2f}"


def inventory_frame() -> pd.DataFrame:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for domain, datasets in manifest["groups"].items():
        for dataset, dataset_spec in datasets.items():
            selected_independent = _selected_candidate_name(domain, dataset, "independent") if "independent" in dataset_spec["methods"] else None
            for dependency in dataset_spec.get("dependencies", []):
                expected_paths = [_run_dir_from_template(dependency["run_template"], seed) for seed in dependency.get("seeds", [])]
                found = sum((path / "summary.json").exists() for path in expected_paths)
                rows.append(
                    {
                        "domain": domain,
                        "dataset": dataset,
                        "kind": "dependency",
                        "method": dependency["name"],
                        "variant": dependency.get("display_name", dependency["name"]),
                        "selected": False,
                        "expected": len(expected_paths),
                        "found": found,
                        "missing": len(expected_paths) - found,
                        "run_template": dependency["run_template"],
                    }
                )
            for method, method_spec in dataset_spec["methods"].items():
                if method_spec.get("selection") == "best_baseline":
                    for candidate in method_spec.get("candidates", []):
                        expected_paths = [_run_dir_from_template(candidate["run_template"], seed) for seed in candidate.get("seeds", [])]
                        found = sum((path / "summary.json").exists() for path in expected_paths)
                        rows.append(
                            {
                                "domain": domain,
                                "dataset": dataset,
                                "kind": "candidate",
                                "method": method,
                                "variant": candidate.get("display_name", candidate["name"]),
                                "selected": candidate["name"] == selected_independent,
                                "expected": len(expected_paths),
                                "found": found,
                                "missing": len(expected_paths) - found,
                                "run_template": candidate["run_template"],
                            }
                        )
                else:
                    expected_paths = [_run_dir_from_template(method_spec["run_template"], seed) for seed in method_spec.get("seeds", [])]
                    found = sum((path / "summary.json").exists() for path in expected_paths)
                    rows.append(
                        {
                            "domain": domain,
                            "dataset": dataset,
                            "kind": "method",
                            "method": method,
                            "variant": method_spec.get("display_name", pretty_method(method)),
                            "selected": True,
                            "expected": len(expected_paths),
                            "found": found,
                            "missing": len(expected_paths) - found,
                            "run_template": method_spec["run_template"],
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["domain", "dataset", "kind", "method", "variant"]).reset_index(drop=True)


def _build_domain_frame(domain: str, seed_overrides: dict[Any, Any] | None = None) -> pd.DataFrame:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for dataset in manifest["figure_order"][domain]:
        dataset_spec = manifest["groups"][domain][dataset]
        for method in METHOD_ORDER:
            entry = _resolved_entry(domain, dataset, method)
            runs = _entry_runs(domain, dataset, method, entry, seed_overrides=seed_overrides)
            metrics = [_metric_from_run(run, dataset_spec, entry) for run in runs]
            metrics = [value for value in metrics if np.isfinite(value)]
            mean_value = float(np.mean(metrics)) if metrics else float("nan")
            std_value = float(np.std(metrics, ddof=1)) if len(metrics) >= 2 else 0.0 if metrics else float("nan")
            rows.append(
                {
                    "domain": domain,
                    "task_label": TASK_LABELS[domain],
                    "dataset": dataset,
                    "dataset_label": dataset_spec.get("display_name", pretty_dataset(dataset)),
                    "method": method,
                    "method_label": pretty_method(method),
                    "mean": mean_value,
                    "std": std_value,
                    "n": len(metrics),
                    "display": _format_mean_std(mean_value, std_value),
                    "selected_variant": entry.get("display_name", pretty_method(method)),
                    "note": dataset_spec.get("note", ""),
                    "metric_label": dataset_spec.get("metric_label", ""),
                    "metric_direction": dataset_spec.get("metric_direction", "minimize"),
                }
            )
    return pd.DataFrame(rows)


def build_time_series_table(seed_overrides: dict[Any, Any] | None = None, export: bool = False) -> pd.DataFrame:
    frame = _build_domain_frame("time_series", seed_overrides=seed_overrides)
    if export:
        export_table(frame, "time_series_wrapup_summary.csv")
    return frame


def build_operator_table(seed_overrides: dict[Any, Any] | None = None, export: bool = False) -> pd.DataFrame:
    frame = _build_domain_frame("operator", seed_overrides=seed_overrides)
    if export:
        export_table(frame, "operator_wrapup_summary.csv")
    return frame


def build_classification_table(seed_overrides: dict[Any, Any] | None = None, export: bool = False) -> pd.DataFrame:
    frame = _build_domain_frame("classification", seed_overrides=seed_overrides)
    if export:
        export_table(frame, "classification_wrapup_summary.csv")
    return frame


def build_main_results_table(seed_overrides: dict[Any, Any] | None = None, export: bool = False) -> pd.DataFrame:
    frames = [
        build_time_series_table(seed_overrides=seed_overrides, export=False),
        build_operator_table(seed_overrides=seed_overrides, export=False),
        build_classification_table(seed_overrides=seed_overrides, export=False),
    ]
    combined = pd.concat(frames, ignore_index=True)
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for domain in ("time_series", "operator", "classification"):
        for dataset in manifest["figure_order"][domain]:
            subset = combined[(combined["domain"] == domain) & (combined["dataset"] == dataset)]
            row: dict[str, Any] = {
                "Task": TASK_LABELS[domain],
                "Dataset": manifest["groups"][domain][dataset].get("display_name", pretty_dataset(dataset)),
                "Note": manifest["groups"][domain][dataset].get("note", ""),
            }
            for method in METHOD_ORDER:
                method_row = subset[subset["method"] == method]
                if method_row.empty:
                    row[pretty_method(method)] = "missing"
                    continue
                row[pretty_method(method)] = method_row.iloc[0]["display"]
            rows.append(row)
    frame = pd.DataFrame(rows)
    if export:
        export_table(frame, "final_wrapup_summary.csv")
    return frame


def plot_final_time_series(
    seed_overrides: dict[Any, Any] | None = None,
    windows: dict[str, Any] | None = None,
    legend_labels: dict[str, str] | None = None,
    show_bands: bool = False,
    export: bool = False,
) -> plt.Figure:
    apply_report_style()
    manifest = load_manifest()
    legend_labels = legend_labels or {}
    datasets = manifest["figure_order"]["time_series"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(17, 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    legend_map: dict[str, Any] = {}

    for ax, dataset in zip(axes, datasets):
        dataset_spec = manifest["groups"]["time_series"][dataset]
        window = _normalize_window((windows or {}).get(dataset), dataset_spec["figure_window"])
        start_epoch, end_epoch = window
        for method in METHOD_ORDER:
            entry = _resolved_entry("time_series", dataset, method)
            runs = _entry_runs("time_series", dataset, method, entry, seed_overrides=seed_overrides)
            curves = [_extract_curve(run, entry["curve_mode"]) for run in runs]
            mean_curve, std_curve = _mean_std_curve(curves)
            if mean_curve.size == 0:
                continue
            epochs = np.arange(1, len(mean_curve) + 1)
            mask = (epochs >= start_epoch) & (epochs <= end_epoch)
            label = legend_labels.get(method, pretty_method(method))
            line = ax.plot(
                epochs[mask],
                mean_curve[mask],
                color=METHOD_COLORS.get(method, "#4c566a"),
                linewidth=2.2,
                label=label,
            )[0]
            if show_bands:
                ax.fill_between(
                    epochs[mask],
                    (mean_curve - std_curve)[mask],
                    (mean_curve + std_curve)[mask],
                    color=METHOD_COLORS.get(method, "#4c566a"),
                    alpha=0.15,
                )
            legend_map.setdefault(label, line)
        ax.set_title(dataset_spec["display_name"])
        ax.set_xlim(start_epoch, end_epoch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.grid(True, alpha=0.25)

    fig.legend(list(legend_map.values()), list(legend_map.keys()), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    if export:
        export_figure(fig, "final_wrapup_time_series.png")
    return fig


def plot_final_operator(
    seed_overrides: dict[Any, Any] | None = None,
    windows: dict[str, Any] | None = None,
    full_windows: dict[str, Any] | None = None,
    legend_labels: dict[str, str] | None = None,
    show_bands: bool = False,
    export: bool = False,
) -> plt.Figure:
    apply_report_style()
    manifest = load_manifest()
    legend_labels = legend_labels or {}
    datasets = manifest["figure_order"]["operator"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4.8), sharey=False)
    axes = np.atleast_1d(axes)
    legend_map: dict[str, Any] = {}

    for ax, dataset in zip(axes, datasets):
        dataset_spec = manifest["groups"]["operator"][dataset]
        zoom_window = _normalize_window((windows or {}).get(dataset), dataset_spec["figure_window"])
        full_window = _normalize_window((full_windows or {}).get(dataset), dataset_spec.get("full_window", [1, zoom_window[1]]))
        main_methods = dataset_spec.get("main_methods", list(METHOD_ORDER))
        inset_methods = dataset_spec.get("inset_methods", main_methods)

        for method in main_methods:
            entry = _resolved_entry("operator", dataset, method)
            runs = _entry_runs("operator", dataset, method, entry, seed_overrides=seed_overrides)
            curves = [_extract_curve(run, entry["curve_mode"]) for run in runs]
            mean_curve, std_curve = _mean_std_curve(curves)
            if mean_curve.size == 0:
                continue
            epochs = np.arange(1, len(mean_curve) + 1)
            mask = (epochs >= full_window[0]) & (epochs <= full_window[1])
            label = legend_labels.get(method, pretty_method(method))
            line = ax.plot(
                epochs[mask],
                mean_curve[mask],
                color=METHOD_COLORS.get(method, "#4c566a"),
                linewidth=2.2,
                label=label,
            )[0]
            if show_bands:
                lower, upper = _clip_positive_band(mean_curve, std_curve)
                ax.fill_between(
                    epochs[mask],
                    lower[mask],
                    upper[mask],
                    color=METHOD_COLORS.get(method, "#4c566a"),
                    alpha=0.12,
                )
            legend_map.setdefault(label, line)

        ax.set_title(dataset_spec["display_name"])
        ax.set_xlim(full_window[0], full_window[1])
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.grid(True, alpha=0.25, which="both")

        inset = inset_axes(ax, width="45%", height="45%", loc="upper right")
        zoom_values: list[np.ndarray] = []
        for method in inset_methods:
            entry = _resolved_entry("operator", dataset, method)
            runs = _entry_runs("operator", dataset, method, entry, seed_overrides=seed_overrides)
            curves = [_extract_curve(run, entry["curve_mode"]) for run in runs]
            mean_curve, _ = _mean_std_curve(curves)
            if mean_curve.size == 0:
                continue
            epochs = np.arange(1, len(mean_curve) + 1)
            mask = (epochs >= zoom_window[0]) & (epochs <= zoom_window[1])
            inset.plot(
                epochs[mask],
                mean_curve[mask],
                color=METHOD_COLORS.get(method, "#4c566a"),
                linewidth=1.5,
            )
            if np.any(mask):
                zoom_values.append(mean_curve[mask])
        inset.set_xlim(zoom_window[0], zoom_window[1])
        inset.set_yscale("log")
        if zoom_values:
            zoom_concat = np.concatenate([values for values in zoom_values if values.size])
            if zoom_concat.size:
                y_min = float(np.nanmin(zoom_concat))
                y_max = float(np.nanmax(zoom_concat))
                if y_min > 0 and y_max > 0:
                    log_min = np.log10(y_min)
                    log_max = np.log10(y_max)
                    pad = max((log_max - log_min) * 0.12, 0.05)
                    inset.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))
        inset.tick_params(labelsize=7)
        inset.grid(True, alpha=0.18, which="both")
        inset.set_title("Zoom", fontsize=8)

    fig.legend(list(legend_map.values()), list(legend_map.keys()), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(top=0.84, wspace=0.28)
    if export:
        export_figure(fig, "final_wrapup_operator.png")
    return fig


def plot_final_classification(
    seed_overrides: dict[Any, Any] | None = None,
    windows: dict[str, Any] | None = None,
    tail_windows: dict[str, Any] | None = None,
    legend_labels: dict[str, str] | None = None,
    show_bands: bool = False,
    export: bool = False,
) -> plt.Figure:
    apply_report_style()
    manifest = load_manifest()
    legend_labels = legend_labels or {}
    datasets = manifest["figure_order"]["classification"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    legend_map: dict[str, Any] = {}

    for ax, dataset in zip(axes, datasets):
        dataset_spec = manifest["groups"]["classification"][dataset]
        window = _normalize_window((windows or {}).get(dataset), dataset_spec["figure_window"])
        tail_window = _normalize_window((tail_windows or {}).get(dataset), dataset_spec.get("tail_window", [max(window[0], window[1] - 19), window[1]]))
        start_epoch, end_epoch = window
        inset = inset_axes(ax, width="45%", height="45%", loc="upper right")
        tail_values: list[np.ndarray] = []
        for method in METHOD_ORDER:
            entry = _resolved_entry("classification", dataset, method)
            runs = _entry_runs("classification", dataset, method, entry, seed_overrides=seed_overrides)
            curves = [_extract_curve(run, entry["curve_mode"]) for run in runs]
            mean_curve, std_curve = _mean_std_curve(curves)
            if mean_curve.size == 0:
                continue
            epochs = np.arange(1, len(mean_curve) + 1)
            mask = (epochs >= start_epoch) & (epochs <= end_epoch)
            label = legend_labels.get(method, pretty_method(method))
            line = ax.plot(
                epochs[mask],
                mean_curve[mask],
                color=METHOD_COLORS.get(method, "#4c566a"),
                linewidth=2.2,
                label=label,
            )[0]
            if show_bands:
                ax.fill_between(
                    epochs[mask],
                    np.clip((mean_curve - std_curve)[mask], a_min=0.0, a_max=None),
                    np.clip((mean_curve + std_curve)[mask], a_min=0.0, a_max=None),
                    color=METHOD_COLORS.get(method, "#4c566a"),
                    alpha=0.15,
                )
            tail_mask = (epochs >= tail_window[0]) & (epochs <= tail_window[1])
            if np.any(tail_mask):
                inset.plot(
                    epochs[tail_mask],
                    mean_curve[tail_mask],
                    color=METHOD_COLORS.get(method, "#4c566a"),
                    linewidth=1.5,
                )
                tail_values.append(mean_curve[tail_mask])
            legend_map.setdefault(label, line)
        ax.set_title(dataset_spec["display_name"])
        ax.set_xlim(start_epoch, end_epoch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(dataset_spec.get("metric_label", "Validation Accuracy (%)"))
        ax.grid(True, alpha=0.25)

        inset.set_xlim(tail_window[0], tail_window[1])
        if tail_values:
            tail_concat = np.concatenate([values for values in tail_values if values.size])
            if tail_concat.size:
                y_min = float(np.nanmin(tail_concat))
                y_max = float(np.nanmax(tail_concat))
                span = y_max - y_min
                pad = max(span * 0.12, 0.15)
                inset.set_ylim(max(0.0, y_min - pad), y_max + pad)
        inset.tick_params(labelsize=7)
        inset.grid(True, alpha=0.18)
        inset.set_title("Zoom", fontsize=8)

    fig.legend(list(legend_map.values()), list(legend_map.keys()), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    if export:
        export_figure(fig, "final_wrapup_classification.png")
    return fig
