"""Single-command reproducible pipeline runner."""

from __future__ import annotations

import argparse
import json

from secom_causal.config import load_config
from secom_causal.phase1 import run_phase1
from secom_causal.phase2 import run_phase2
from secom_causal.phase3 import run_phase3


def main() -> None:
    """Parse CLI arguments and run requested pipeline phase(s)."""
    parser = argparse.ArgumentParser(description="SECOM Causal Root-Cause Pipeline")
    parser.add_argument(
        "--phase",
        type=str,
        default="1",
        help="Phase to run ('1', '2', '3', or 'all').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.phase == "1":
        summary = run_phase1(config)
        print(json.dumps(summary, indent=2, default=str))
    elif args.phase == "2":
        summary = run_phase2(config)
        print(json.dumps(summary, indent=2, default=str))
    elif args.phase == "3":
        summary = run_phase3(config)
        print(json.dumps(summary, indent=2, default=str))
    elif args.phase == "all":
        s1 = run_phase1(config)
        s2 = run_phase2(config)
        s3 = run_phase3(config)
        print(json.dumps({"phase1": s1, "phase2": s2, "phase3": s3}, indent=2, default=str))
    else:
        raise ValueError(f"Phase '{args.phase}' is not implemented yet.")


if __name__ == "__main__":
    main()
