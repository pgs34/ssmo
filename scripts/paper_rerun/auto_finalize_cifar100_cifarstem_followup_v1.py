from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_OUTPUT_ROOT = ROOT / os.environ.get(
    "FOLLOWUP_OUTPUT_ROOT", "results/classification_cifar100_cifarstem_followup_v1"
)
POOL_ROOT = ROOT / os.environ.get(
    "POOL_ROOT", "results/classification_cifar100_bestckpt_pool_cifarstem_v1"
)
REPORT_PATH = ROOT / os.environ.get(
    "REPORT_PATH",
    "results/logs/classification_cifar100_cifarstem_followup_v1/node0/cifarstem_followup_report.json",
)
SUMMARY_PATH = ROOT / os.environ.get("FOLLOWUP_SUMMARY_MD", "Results_Summary.md")
PLOT_REFRESH_SCRIPT = ROOT / os.environ.get(
    "PLOT_REFRESH_SCRIPT", "scripts/paper_rerun/refresh_top_level_best_plots.sh"
)
QUEUE_SCRIPT = ROOT / os.environ.get(
    "QUEUE_SCRIPT",
    "scripts/paper_rerun/cluster/run_worker_cifar100_cifarstem_followup_queue_v1.sh",
)
WATCH_LOG_ROOT = ROOT / os.environ.get(
    "WATCH_LOG_ROOT", "results/logs/classification_cifar100_cifarstem_followup_v1/autofinish"
)
LOCK_PATH = ROOT / os.environ.get(
    "AUTOFINISH_LOCK_PATH",
    "results/.locks/auto_finalize_cifar100_cifarstem_followup_v1.lock",
)

CLASSIFICATION_IMITATION_LOSS = os.environ.get("CLASSIFICATION_IMITATION_LOSS", "kl")
FOLLOWUP_PROTOCOL_ID = os.environ.get("FOLLOWUP_PROTOCOL_ID", "cifarstem_followup_v1")
CIFARSTEM_INDEPENDENT_LABEL = os.environ.get(
    "CIFARSTEM_INDEPENDENT_LABEL", "cifarstem_independent_v1"
)
CIFARSTEM_DML_LABEL = os.environ.get("CIFARSTEM_DML_LABEL", "cifarstem_dml_v1")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "120"))

DEFAULT_ALL_PROBE_CASE_SPECS = (
    "pcu_cifarstem_sched_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 "
    "pcu_cifarstem_sched_l10_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.010:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 "
    "pcu_cifarstem_dense_v1:peer_confident_student_uncertain:0.28:0.12:0.05:0.012:0.000:0.35:0.40:0.01:0.03:6:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.15:0.90:1.00:25:70:0.00 "
    "oxtra42_cifarstem_v1:useful_hard_sample_confident:0.42:0.42:0.020:0.018:0.000:0.42:0.42:0.01:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:0.00:1.00:0.00:12:18:36:0.25:0.00:0.00:0.00:-1:-1:0.00 "
    "pcu_cifarstem_sched_l09_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.009:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 "
    "pcu_cifarstem_sched_l08_t7_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.008:0.000:0.35:0.40:0.01:0.03:5:7.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 "
    "oxtra35_cifarstem_relax_v1:useful_hard_sample_confident:0.35:0.35:0.020:0.015:0.000:0.40:0.40:0.01:0.01:12:6.0:0.0002:0.90:1.10:3:0.50:0.04:0.00:1.00:0.00:8:18:45:0.25:0.00:0.00:0.00:-1:-1:0.00"
)
ALL_PROBE_CASE_SPECS = os.environ.get("ALL_PROBE_CASE_SPECS", DEFAULT_ALL_PROBE_CASE_SPECS)
CANDIDATE_LABELS = [spec.split(":", 1)[0] for spec in ALL_PROBE_CASE_SPECS.split() if spec]
FAMILY_MAP = {label: ("pcu" if label.startswith("pcu_") else "oxtra") for label in CANDIDATE_LABELS}

SEED2_TARGET_LABELS = [
    label
    for label in os.environ.get(
        "SEED2_TARGET_LABELS", "oxtra42_cifarstem_v1 pcu_cifarstem_dense_v1"
    ).split()
    if label
]

BACKFILL_HOSTS = {0: os.environ.get("BACKFILL_HOST_SEED0", "worker2"), 1: os.environ.get("BACKFILL_HOST_SEED1", "worker3")}
BACKFILL_GPU = os.environ.get("BACKFILL_GPU", "0")
BACKFILL_NUM_WORKERS = os.environ.get("BACKFILL_NUM_WORKERS", "4")
BACKFILL_INDEPENDENT_BATCH = os.environ.get("BACKFILL_INDEPENDENT_BATCH", "1536")
BACKFILL_DUAL_BATCH = os.environ.get("BACKFILL_DUAL_BATCH", "768")
BACKFILL_INDEPENDENT_LR = os.environ.get("BACKFILL_INDEPENDENT_LR", "0.05")
BACKFILL_INDEPENDENT_WARMUP = os.environ.get("BACKFILL_INDEPENDENT_WARMUP", "5")
BACKFILL_INDEPENDENT_MIN_SCALE = os.environ.get("BACKFILL_INDEPENDENT_MIN_SCALE", "0.10")
BACKFILL_DML_LR = os.environ.get("BACKFILL_DML_LR", "0.025")
BACKFILL_DML_WARMUP = os.environ.get("BACKFILL_DML_WARMUP", "8")
BACKFILL_DML_MIN_SCALE = os.environ.get("BACKFILL_DML_MIN_SCALE", "0.20")
BACKFILL_SSML_LR = os.environ.get("BACKFILL_SSML_LR", "0.025")
BACKFILL_SSML_WARMUP = os.environ.get("BACKFILL_SSML_WARMUP", "8")
BACKFILL_SSML_MIN_SCALE = os.environ.get("BACKFILL_SSML_MIN_SCALE", "0.20")
BACKFILL_HARDWARE_PROFILE = os.environ.get("BACKFILL_HARDWARE_PROFILE", "rtx3090ti")

START_MARKER = "<!-- CIFAR100_CIFARSTEM_FOLLOWUP_V1_START -->"
END_MARKER = "<!-- CIFAR100_CIFARSTEM_FOLLOWUP_V1_END -->"


def log(message: str) -> None:
    print(f"[cifarstem_autofinish_v1] {message}", flush=True)


def summary_path(label: str, seed: int) -> Path:
    run_dir = FOLLOWUP_OUTPUT_ROOT / label / "classification/cifar100"
    if label == CIFARSTEM_INDEPENDENT_LABEL:
        run_name = f"resnet34_cifar_gelu_independent_{CLASSIFICATION_IMITATION_LOSS}_seed{seed}"
    elif label == CIFARSTEM_DML_LABEL:
        run_name = f"resnet34_cifar_gelu_dml_{CLASSIFICATION_IMITATION_LOSS}_seed{seed}"
    else:
        run_name = f"resnet34_cifar_gelu_ssml_{CLASSIFICATION_IMITATION_LOSS}_seed{seed}"
    return run_dir / run_name / "summary.json"


def curve_path(label: str, seed: int) -> Path:
    run_dir = FOLLOWUP_OUTPUT_ROOT / label / "classification/cifar100"
    if label == CIFARSTEM_INDEPENDENT_LABEL:
        run_name = f"resnet34_cifar_gelu_independent_{CLASSIFICATION_IMITATION_LOSS}_seed{seed}"
    elif label == CIFARSTEM_DML_LABEL:
        run_name = f"resnet34_cifar_gelu_dml_{CLASSIFICATION_IMITATION_LOSS}_seed{seed}"
    else:
        run_name = f"resnet34_cifar_gelu_ssml_{CLASSIFICATION_IMITATION_LOSS}_seed{seed}"
    return run_dir / run_name / "curves.npz"


def load_summary_score(label: str, seed: int) -> float | None:
    path = summary_path(label, seed)
    if not path.exists():
        return None
    with path.open() as f:
        return float(json.load(f)["best_val_acc"])


def available_seeds(label: str) -> list[int]:
    return [seed for seed in (0, 1, 2) if summary_path(label, seed).exists()]


def mean_score(label: str, seeds: tuple[int, ...] | list[int]) -> float | None:
    scores = [load_summary_score(label, seed) for seed in seeds]
    if any(score is None for score in scores):
        return None
    return float(sum(scores) / len(scores))


def build_report(promoted_cases: list[str]) -> dict:
    seed2_scores: dict[str, float] = {}
    all_labels = [CIFARSTEM_INDEPENDENT_LABEL, CIFARSTEM_DML_LABEL, *CANDIDATE_LABELS]
    for label in all_labels:
        score = load_summary_score(label, 2)
        if score is not None:
            seed2_scores[label] = score

    three_seed_cases = [
        label for label in CANDIDATE_LABELS if available_seeds(label) == [0, 1, 2]
    ]
    preview_case = None
    preview_seeds: list[int] = []
    preview_mode = "pending"

    controls_3seed = (
        available_seeds(CIFARSTEM_INDEPENDENT_LABEL) == [0, 1, 2]
        and available_seeds(CIFARSTEM_DML_LABEL) == [0, 1, 2]
    )
    if controls_3seed and three_seed_cases:
        preview_case = max(
            three_seed_cases,
            key=lambda label: mean_score(label, (0, 1, 2)) or float("-inf"),
        )
        preview_seeds = [0, 1, 2]
        preview_mode = "matched_3seed"
    elif (
        CIFARSTEM_INDEPENDENT_LABEL in seed2_scores
        and CIFARSTEM_DML_LABEL in seed2_scores
    ):
        seed2_cases = [label for label in CANDIDATE_LABELS if label in seed2_scores]
        if seed2_cases:
            preview_case = max(seed2_cases, key=lambda label: seed2_scores[label])
            preview_seeds = [2]
            preview_mode = "seed2_probe_only"

    latest_row = None
    if preview_case is not None and preview_seeds:
        indep_score = mean_score(CIFARSTEM_INDEPENDENT_LABEL, preview_seeds)
        dml_score = mean_score(CIFARSTEM_DML_LABEL, preview_seeds)
        ssml_score = mean_score(preview_case, preview_seeds)
        if indep_score is not None and dml_score is not None and ssml_score is not None:
            latest_row = {
                "track": "CIFAR-100 cifarstem_followup_v1",
                "backbone": "resnet34_cifar_gelu x resnet34_cifar_gelu",
                "protocol": "matched 3-seed" if preview_mode == "matched_3seed" else "seed2 probe only",
                "independent": indep_score,
                "dml": dml_score,
                "ssml": ssml_score,
                "ssml_case": preview_case,
                "ssml_family": FAMILY_MAP.get(preview_case, "unknown"),
                "verdict": (
                    "SSML > independent and DML"
                    if ssml_score > indep_score and ssml_score > dml_score
                    else "SSML > independent only"
                    if ssml_score > indep_score
                    else "SSML <= independent or DML"
                ),
            }

    pool_complete = {
        seed: (
            POOL_ROOT
            / "classification/classification/cifar100"
            / f"resnet34_cifar_gelu_independent_{CLASSIFICATION_IMITATION_LOSS}_seed{seed}"
            / "best_model.pt"
        ).exists()
        for seed in (0, 1, 2)
    }

    return {
        "track": "classification_cifar100_cifarstem_followup_v1",
        "pool_root": str(POOL_ROOT.relative_to(ROOT)),
        "candidate_labels": CANDIDATE_LABELS,
        "candidate_families": FAMILY_MAP,
        "promoted_cases": promoted_cases,
        "pool_complete": pool_complete,
        "seed2_scores": seed2_scores,
        "preview_case": preview_case,
        "preview_mode": preview_mode,
        "preview_seeds": preview_seeds,
        "latest_matched_row": latest_row,
        "controls_complete_3seed": {
            CIFARSTEM_INDEPENDENT_LABEL: available_seeds(CIFARSTEM_INDEPENDENT_LABEL) == [0, 1, 2],
            CIFARSTEM_DML_LABEL: available_seeds(CIFARSTEM_DML_LABEL) == [0, 1, 2],
        },
        "cases_complete_3seed": {
            label: available_seeds(label) == [0, 1, 2] for label in CANDIDATE_LABELS
        },
        "available_seeds": {label: available_seeds(label) for label in [CIFARSTEM_INDEPENDENT_LABEL, CIFARSTEM_DML_LABEL, *CANDIDATE_LABELS]},
        "curve_paths": {
            label: {str(seed): str(curve_path(label, seed).relative_to(ROOT)) for seed in available_seeds(label)}
            for label in [CIFARSTEM_INDEPENDENT_LABEL, CIFARSTEM_DML_LABEL, *CANDIDATE_LABELS]
        },
        "seed2_target_labels": SEED2_TARGET_LABELS,
    }


def determine_promoted_cases(seed2_target_complete: bool) -> list[str]:
    if not seed2_target_complete:
        return []

    independent_seed2 = load_summary_score(CIFARSTEM_INDEPENDENT_LABEL, 2)
    if independent_seed2 is None:
        return []

    eligible: list[tuple[str, float, str]] = []
    for label in CANDIDATE_LABELS:
        score = load_summary_score(label, 2)
        if score is not None and score > independent_seed2:
            eligible.append((label, score, FAMILY_MAP[label]))

    eligible.sort(key=lambda item: item[1], reverse=True)
    promoted: list[str] = []
    if eligible:
        first_label, _, first_family = eligible[0]
        promoted.append(first_label)
        for label, _, family in eligible[1:]:
            if family != first_family:
                promoted.append(label)
                break
    return promoted


def upsert_results_summary_appendix(report: dict) -> None:
    latest_row = report.get("latest_matched_row")
    preview_mode = report.get("preview_mode", "pending")
    promoted_cases = report.get("promoted_cases") or []

    if latest_row:
        row = (
            f"| {latest_row['track']} | `{latest_row['backbone']}` | `{latest_row['protocol']}` | "
            f"`{latest_row['independent']:.6f}` | `{latest_row['dml']:.6f}` | `{latest_row['ssml']:.6f}` | "
            f"`{latest_row['verdict']}` | preview SSML case = `{latest_row['ssml_case']}` ({latest_row['ssml_family']}); "
            f"current preview mode = `{preview_mode}` |"
        )
    else:
        row = (
            "| CIFAR-100 cifarstem_followup_v1 | `resnet34_cifar_gelu x resnet34_cifar_gelu` | `pending launch / awaiting matched controls` | "
            "pending | pending | pending | pending | seed0 pool bootstrap + seed2 control/probe sweep is the next gate |"
        )

    promoted_line = ", ".join(f"`{label}`" for label in promoted_cases) if promoted_cases else "`none yet`"
    section = f"""
{START_MARKER}

## CIFAR-100 cifarstem_followup_v1 Appendix

### Why backbone pivot

SSML 자체가 전반적으로 망가진 것은 아니다. 다른 domain과 일부 classification setting에서는 이미 개선 신호와 승리가 있었고, 현재 CIFAR-100 clean homogeneous는 `resnet34_gelu` strict track의 backbone/stem 병목이 더 크게 보인다. 그래서 이번 pivot은 방법론 포기가 아니라, capacity와 inductive bias를 바꿔 같은 SSML logic를 다시 검증하는 CIFAR-100 병목 분리 실험이다.

### Latest matched result

| Track | Backbone | Protocol | Independent | DML | SSML | Current verdict | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
{row}

Promoted backfill targets: {promoted_line}

![CIFAR-100 only validation error](./test_error_cifar100_only.png)
{END_MARKER}
"""

    existing = SUMMARY_PATH.read_text()
    if START_MARKER in existing and END_MARKER in existing:
        prefix = existing.split(START_MARKER, 1)[0].rstrip()
        suffix = existing.split(END_MARKER, 1)[1].lstrip()
        updated = prefix + "\n\n" + section.strip() + ("\n\n" + suffix if suffix else "\n")
    else:
        updated = existing.rstrip() + "\n\n" + section.strip() + "\n"
    SUMMARY_PATH.write_text(updated)


def refresh_plots() -> None:
    if not PLOT_REFRESH_SCRIPT.exists():
        log(f"skip plot refresh, missing {PLOT_REFRESH_SCRIPT}")
        return
    subprocess.run(["bash", str(PLOT_REFRESH_SCRIPT)], cwd=ROOT, check=False)


def seed2_targets_complete() -> bool:
    if not SEED2_TARGET_LABELS:
        return False
    return all(summary_path(label, 2).exists() for label in SEED2_TARGET_LABELS)


def launch_backfill_queue(host: str, seed: int, promoted_cases: list[str]) -> None:
    job_items: list[str] = []
    required = [
        ("independent", CIFARSTEM_INDEPENDENT_LABEL),
        ("dml", CIFARSTEM_DML_LABEL),
        *[("ssml", label) for label in promoted_cases],
    ]
    for run_group, label in required:
        if not summary_path(label, seed).exists():
            job_items.append(f"{run_group}:{label}")
    if not job_items:
        return

    WATCH_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    remote_log_dir = WATCH_LOG_ROOT / host / f"seed{seed}_queue"
    remote_launcher_log = remote_log_dir / "launcher.out"
    env_map = {
        "QUEUE_NAME": f"cifarstem_followup_backfill_seed{seed}_v1",
        "TARGET_GPU": BACKFILL_GPU,
        "TARGET_SEED": str(seed),
        "JOB_ITEMS": " ".join(job_items),
        "LOG_ROOT": str(WATCH_LOG_ROOT / host / f"seed{seed}_jobs"),
        "OUTPUT_ROOT": str(FOLLOWUP_OUTPUT_ROOT),
        "CLASSIFICATION_IMITATION_LOSS": CLASSIFICATION_IMITATION_LOSS,
        "PROTOCOL_ID": FOLLOWUP_PROTOCOL_ID,
        "HARDWARE_PROFILE": BACKFILL_HARDWARE_PROFILE,
        "FOLLOWUP_EPOCHS": "100",
        "NUM_WORKERS": BACKFILL_NUM_WORKERS,
        "INDEPENDENT_BATCH_SIZE": BACKFILL_INDEPENDENT_BATCH,
        "DUAL_BATCH_SIZE": BACKFILL_DUAL_BATCH,
        "INDEPENDENT_LR": BACKFILL_INDEPENDENT_LR,
        "INDEPENDENT_WARMUP": BACKFILL_INDEPENDENT_WARMUP,
        "INDEPENDENT_MIN_SCALE": BACKFILL_INDEPENDENT_MIN_SCALE,
        "DML_LR": BACKFILL_DML_LR,
        "DML_WARMUP": BACKFILL_DML_WARMUP,
        "DML_MIN_SCALE": BACKFILL_DML_MIN_SCALE,
        "SSML_LR": BACKFILL_SSML_LR,
        "SSML_WARMUP": BACKFILL_SSML_WARMUP,
        "SSML_MIN_SCALE": BACKFILL_SSML_MIN_SCALE,
        "ALL_PROBE_CASE_SPECS": ALL_PROBE_CASE_SPECS,
        "CIFARSTEM_INDEPENDENT_LABEL": CIFARSTEM_INDEPENDENT_LABEL,
        "CIFARSTEM_DML_LABEL": CIFARSTEM_DML_LABEL,
    }
    env_str = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_map.items())
    remote_cmd = (
        f"cd {shlex.quote(str(ROOT))} && "
        f"mkdir -p {shlex.quote(str(remote_log_dir))} && "
        f"nohup env {env_str} bash {shlex.quote(str(QUEUE_SCRIPT))} "
        f"> {shlex.quote(str(remote_launcher_log))} 2>&1 < /dev/null & echo $!"
    )
    result = subprocess.run(["ssh", host, remote_cmd], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        pid = result.stdout.strip() or "unknown"
        log(f"backfill queue launch host={host} seed={seed} pid={pid} items={' '.join(job_items)}")
    else:
        log(
            "backfill queue launch failed "
            f"host={host} seed={seed} rc={result.returncode} stderr={result.stderr.strip()}"
        )


def maybe_launch_backfills(promoted_cases: list[str]) -> None:
    if not promoted_cases or not seed2_targets_complete():
        return
    for seed, host in BACKFILL_HOSTS.items():
        launch_backfill_queue(host, seed, promoted_cases)


def report_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2, sort_keys=True).encode("utf-8")


def write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_bytes(report_bytes(report) + b"\n")


def is_finished(report: dict) -> bool:
    promoted_cases = report.get("promoted_cases") or []
    controls_complete = bool(
        report.get("controls_complete_3seed", {}).get(CIFARSTEM_INDEPENDENT_LABEL)
        and report.get("controls_complete_3seed", {}).get(CIFARSTEM_DML_LABEL)
    )
    cases_complete = report.get("cases_complete_3seed", {})

    if promoted_cases:
        return controls_complete and all(cases_complete.get(label, False) for label in promoted_cases)
    return seed2_targets_complete() and bool(report.get("preview_mode") == "seed2_probe_only")


def main() -> int:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fp = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"watcher already running lock={LOCK_PATH}")
        return 0

    log(f"root={ROOT}")
    log(f"followup_output_root={FOLLOWUP_OUTPUT_ROOT}")
    log(f"report_path={REPORT_PATH}")
    log(f"summary_path={SUMMARY_PATH}")
    log(f"seed2_targets={SEED2_TARGET_LABELS}")

    last_report_blob: bytes | None = None
    last_plot_refresh_key: bytes | None = None

    while True:
        try:
            targets_complete = seed2_targets_complete()
            promoted_cases = determine_promoted_cases(targets_complete)
            report = build_report(promoted_cases)
            report_blob = report_bytes(report)

            if report_blob != last_report_blob:
                write_report(report)
                upsert_results_summary_appendix(report)
                last_report_blob = report_blob
                log(
                    "report updated "
                    f"preview_mode={report.get('preview_mode')} preview_case={report.get('preview_case')} "
                    f"promoted={report.get('promoted_cases')}"
                )

            maybe_launch_backfills(promoted_cases)

            plot_refresh_key = report_blob
            if plot_refresh_key != last_plot_refresh_key:
                refresh_plots()
                last_plot_refresh_key = plot_refresh_key
                log("plots refreshed")

            if is_finished(report):
                log("finalization complete")
                return 0

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            log("interrupted")
            return 130
        except Exception as exc:  # pragma: no cover - safety loop
            log(f"loop error: {exc}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
