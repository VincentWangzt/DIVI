"""Post-hoc evaluation of VI output fourth moments at each checkpoint.

Loads DIVI checkpoints from the campaign manifest and computes
E_ε[‖μ_φ(ε)‖₂⁴] and E_ε[‖σ_φ(ε)‖₂⁴] at each saved checkpoint,
outputting results as CSV.

Usage::

    # Evaluate x_shaped target, all seeds, all checkpoints
    python -m finalization.eval_vi_fourth_moment \\
        --targets x_shaped --n-samples 1024 --device cuda \\
        --output-dir results/vi_fourth_moment/x_shaped

    # All targets, every 5th checkpoint
    python -m finalization.eval_vi_fourth_moment \\
        --checkpoint-stride 5 --n-samples 1024 \\
        --output-dir results/vi_fourth_moment/full
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from utils.logging import get_logger

from .artifacts import (
    RunRecord,
    completed_runs,
    find_all_checkpoints,
    load_manifest,
    select_runs,
)
from .config import REPO_ROOT, repo_path
from .eval_assumption_jacobian import (
    build_runner_for_eval,
    prepare_config,
    remove_file_handlers,
    set_seed,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "target",
    "seed",
    "epoch",
    "z_dim",
    "n_samples",
    "mu_fourth_moment",
    "std_fourth_moment",
    "vi_fourth_moment",
    "mu_second_moment",
    "std_second_moment",
    "run_id",
    "duration_sec",
]


def evaluate_run(
    rec: RunRecord,
    *,
    device: str,
    n_samples: int,
    checkpoint_stride: int,
) -> list[dict[str, Any]]:
    """Evaluate VI output fourth moments at all checkpoints for one run."""
    checkpoints = find_all_checkpoints(rec.result_path)
    if not checkpoints:
        logger.warning(f"No checkpoints found for {rec.run_id} at {rec.result_path}")
        return []

    if checkpoint_stride > 1:
        checkpoints = checkpoints[::checkpoint_stride]

    cfg = prepare_config(rec, device=device)
    runner = build_runner_for_eval(rec, cfg)
    vi_model = runner.vi_model

    results = []
    for epoch, ckpt_path in checkpoints:
        t0 = time.perf_counter()

        state = torch.load(ckpt_path, map_location=runner.device)
        vi_model.load_state_dict(state)
        vi_model.eval()

        with torch.no_grad():
            epsilon = vi_model.sample_epsilon(num=n_samples)
            mu = vi_model.getmu(epsilon)        # [N, D_z]
            std = vi_model.getstd(epsilon)      # [N, D_z]

            norm_mu = torch.norm(mu, p=2, dim=-1)    # [N]
            norm_std = torch.norm(std, p=2, dim=-1)  # [N]

            mu_fourth = torch.mean(norm_mu ** 4).item()
            std_fourth = torch.mean(norm_std ** 4).item()
            vi_fourth_moment = max(mu_fourth, std_fourth)

            mu_second = torch.mean(norm_mu ** 2).item()
            std_second = torch.mean(norm_std ** 2).item()

        duration = time.perf_counter() - t0

        results.append({
            "target": rec.target,
            "seed": rec.seed,
            "epoch": epoch,
            "z_dim": int(vi_model.z_dim),
            "n_samples": n_samples,
            "mu_fourth_moment": mu_fourth,
            "std_fourth_moment": std_fourth,
            "vi_fourth_moment": vi_fourth_moment,
            "mu_second_moment": mu_second,
            "std_second_moment": std_second,
            "run_id": rec.run_id,
            "duration_sec": round(duration, 3),
        })

    return results


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write results to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = "campaigns/default_config_grid/manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VI output fourth moments at DIVI checkpoints.",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST,
        help="Path to campaign manifest JSON (default: %(default)s).",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Target names to evaluate (default: all DIVI targets in manifest).",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds to evaluate (default: all seeds in manifest).",
    )
    p.add_argument(
        "--checkpoint-stride",
        type=int,
        default=1,
        help="Evaluate every N-th checkpoint (default: 1 = all).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=1024,
        help="Number of epsilon samples for Monte Carlo estimate (default: 1024).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/vi_fourth_moment",
        help="Output directory for CSV results (default: %(default)s).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device (default: auto).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load manifest and filter to DIVI runs
    manifest_path = repo_path(args.manifest)
    if manifest_path is None or not manifest_path.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)
    manifest = load_manifest(args.manifest)
    records = completed_runs(manifest)

    # Filter to DIVI only
    divi_records = [r for r in records if r.runner_type.upper() == "DIVI"]
    if not divi_records:
        print("No completed DIVI runs found in manifest.", file=sys.stderr)
        sys.exit(1)

    # Filter by targets
    if args.targets:
        target_set = set(args.targets)
        divi_records = [r for r in divi_records if r.target in target_set]

    # Filter by seeds
    if args.seeds:
        seed_set = set(args.seeds)
        divi_records = [r for r in divi_records if r.seed in seed_set]

    if not divi_records:
        print("No matching DIVI runs after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(divi_records)} DIVI runs...")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    all_results: list[dict[str, Any]] = []
    for rec in tqdm(divi_records, desc="Runs"):
        try:
            run_results = evaluate_run(
                rec,
                device=args.device,
                n_samples=args.n_samples,
                checkpoint_stride=args.checkpoint_stride,
            )
            all_results.extend(run_results)
            # Incremental save
            write_csv(all_results, output_dir / "results_full.csv")
        except FileNotFoundError as e:
            logger.warning(f"Skipping {rec.run_id}: {e}")
        except Exception as e:
            logger.error(f"Error evaluating {rec.run_id}: {e}", exc_info=True)

    if all_results:
        write_csv(all_results, output_dir / "results_full.csv")
        print(f"\nResults written to {output_dir}/")
        print(f"  results_full.csv: {len(all_results)} rows")
    else:
        print("No results produced (checkpoints may be missing locally).",
              file=sys.stderr)


if __name__ == "__main__":
    main()
