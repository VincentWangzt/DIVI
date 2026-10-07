"""Time native score estimators on shared, frozen DIVI checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finalization.score_approximation import select_checkpoints
from finalization.score_estimator_timing import load_timing_config, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-only", action="store_true", help="List source checkpoints without fitting or timing.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides.")
    args = parser.parse_args()
    cfg = load_timing_config(args.config, args.overrides)
    if args.check_only:
        for checkpoint in select_checkpoints(cfg):
            print(f"{checkpoint.target} seed={checkpoint.seed} epoch={checkpoint.epoch}: {checkpoint.checkpoint_dir}")
    else:
        run_benchmark(cfg)


if __name__ == "__main__":
    main()
