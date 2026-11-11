"""Single-command reproducible pipeline runner."""

from __future__ import annotations

import argparse
import json

from secom_causal.config import load_config
from secom_causal.phase1 import run_phase1


def main() -> None:
    """Parse CLI arguments and run requested pipeline phase(s)."""
    parser = argparse.ArgumentParser(description="SECOM Causal Root-Cause Pipeline")
    parser.add_argument(
        "--phase",
        type=str,
        default="1",
        help="Phase to run (currently supported: '1').",
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
    else:
        raise ValueError(f"Phase '{args.phase}' is not implemented yet.")


if __name__ == "__main__":
    main()
