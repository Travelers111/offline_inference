#!/usr/bin/env python3
"""Compare easy-mirro offline inference summary metrics across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_KEYS = [
    "avg_first_step_delta_mse",
    "avg_first_step_delta_mae",
    "avg_valid_chunk_delta_mse",
    "avg_valid_chunk_delta_mae",
]


def load_metrics(path: Path) -> dict:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
    return json.loads(metrics_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare easy-mirro checkpoint eval outputs")
    parser.add_argument("--outputs", nargs="+", required=True, help="Output directories containing metrics.json")
    parser.add_argument("--names", nargs="+", default=None, help="Optional display names")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    outputs = [Path(p).expanduser().resolve() for p in args.outputs]
    names = args.names or [p.name for p in outputs]
    if len(names) != len(outputs):
        raise ValueError("--names length must match --outputs length")

    rows = []
    for name, output_dir in zip(names, outputs):
        metrics = load_metrics(output_dir)
        row = {"name": name, "output_dir": str(output_dir)}
        for key in PRIMARY_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)

    best_key = "avg_valid_chunk_delta_mse"
    best = min(rows, key=lambda row: row[best_key])
    result = {
        "primary_metric": best_key,
        "best": best["name"],
        "rows": rows,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
