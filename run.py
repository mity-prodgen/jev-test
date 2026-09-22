#!/usr/bin/env python3
"""CLI entry point: run a batch and print its summary.

    python run.py testcases/off_menu.yaml
    python run.py testcases/calibration.yaml --model jev-1.13.0
"""

from __future__ import annotations

import argparse
import json

from jev_harness.runner import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Jev test batch and export results.")
    parser.add_argument("batch", help="Path to a batch YAML/JSON file")
    parser.add_argument("--model", default=None, help="Override the model, e.g. jev-1.13.0 (pin this for reproducibility)")
    parser.add_argument("--out-dir", default="results", help="Output directory root (default: results/)")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    run_dir = run_batch(
        args.batch,
        out_dir=args.out_dir,
        model=args.model,
        max_retries=args.max_retries,
    )

    summary = json.loads((run_dir / "summary.json").read_text())
    print(f"Results written to {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
