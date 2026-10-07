"""Evaluate score estimators on shared, frozen DIVI checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finalization.score_approximation import load_score_config, run_analysis, select_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-only", action="store_true", help="List matching source checkpoints without fitting or evaluation.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides.")
    args = parser.parse_args()
    cfg = load_score_config(args.config, args.overrides)
    if args.check_only:
        for checkpoint in select_checkpoints(cfg):
            print(f"{checkpoint.target} seed={checkpoint.seed} epoch={checkpoint.epoch}: {checkpoint.checkpoint_dir}")
    else:
        run_analysis(cfg)


if __name__ == "__main__":
    main()
