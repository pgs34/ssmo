#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--metric-key", required=True)
    parser.add_argument("--expected-seeds", default="0,1,2")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--current-best", type=float, required=True)
    parser.add_argument("--strongest-baseline", type=float, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    expected_seeds = tuple(int(x) for x in args.expected_seeds.split(",") if x.strip())
    grouped: dict[str, dict[int, float]] = {}
    for summary_path in sorted(args.run_root.glob("*/**/summary.json")):
        rel = summary_path.relative_to(args.run_root)
        if len(rel.parts) < 2:
            continue
        case_name = rel.parts[0]
        summary = load_json(summary_path)
        seed = int(summary.get("seed", -1))
        metric = summary.get(args.metric_key)
        if seed < 0 or metric is None:
            continue
        grouped.setdefault(case_name, {})[seed] = float(metric)

    rows = []
    for case_name, seed_map in sorted(grouped.items()):
        missing = [seed for seed in expected_seeds if seed not in seed_map]
        values = [seed_map[seed] for seed in expected_seeds if seed in seed_map]
        row = {
            "case": case_name,
            "completed": not missing,
            "missing_seeds": missing,
            "count": len(values),
        }
        if values:
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            row["mean"] = mean
            row["std"] = std
            if args.higher_is_better:
                row["gap_to_current_best"] = mean - args.current_best
                row["gap_to_strongest_baseline"] = mean - args.strongest_baseline
            else:
                row["gap_to_current_best"] = args.current_best - mean
                row["gap_to_strongest_baseline"] = args.strongest_baseline - mean
        rows.append(row)

    if args.higher_is_better:
        rows.sort(key=lambda row: row.get("mean", float("-inf")), reverse=True)
    else:
        rows.sort(key=lambda row: row.get("mean", float("inf")))

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
