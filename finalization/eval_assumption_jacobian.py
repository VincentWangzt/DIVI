"""Post-hoc evaluation of the bounded-gradient assumption (Assumption 1).

Loads DIVI checkpoints from the campaign manifest and computes
E_ε[‖∇_φ μ_φ(ε)‖₂²] and E_ε[‖∇_φ σ_φ(ε)‖₂²] at each saved checkpoint,
outputting results as CSV.

Usage::

    # Evaluate x_shaped target, all seeds, all checkpoints
    python -m finalization.eval_assumption_jacobian \\
        --targets x_shaped --n-samples 256 --device cuda \\
        --output-dir results/assumption_validation/x_shaped

    # Quick sanity check only
    python -m finalization.eval_assumption_jacobian --sanity-check

    # All targets, every 5th checkpoint
    python -m finalization.eval_assumption_jacobian \\
        --checkpoint-stride 5 --n-samples 256 \\
        --output-dir results/assumption_validation/full
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from runner.runners import Runners
from utils.jacobian_spectral import evaluate_assumption_bound, sanity_check
from utils.logging import get_logger

from .artifacts import (
    RunRecord,
    completed_runs,
    find_all_checkpoints,
    load_manifest,
    select_runs,
)
from .config import REPO_ROOT, repo_path

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers (adapted from runner_eval.py)
# ---------------------------------------------------------------------------

def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_file_handlers() -> None:
    root_logger = get_logger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)


def prepare_config(rec: RunRecord, *, device: str) -> Any:
    """Load and prepare config for a run, overriding device and output dirs."""
    cfg = OmegaConf.load(rec.config_path)
    cfg.config_path = rec.config_path.as_posix()
    if device == "cpu":
        resolved_device = "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        resolved_device = "cuda"
    else:
        resolved_device = (
            "cuda"
            if cfg.get("use_cuda", False) and torch.cuda.is_available()
            else "cpu"
        )
    cfg.device = resolved_device
    cfg.use_cuda = resolved_device == "cuda"
    # Scratch directories so the runner doesn't write to real paths
    scratch = tempfile.mkdtemp(prefix="assumption_eval_")
    cfg.output = OmegaConf.merge(
        cfg.get("output", {}),
        {
            "results_dir": os.path.join(scratch, "results", rec.run_id),
            "tb_dir": os.path.join(scratch, "tb", rec.run_id),
        },
    )
    return cfg


def build_runner_for_eval(rec: RunRecord, cfg: Any):
    """Instantiate a runner (without loading any checkpoint yet)."""
    set_seed(rec.seed, cfg.device == "cuda")
    runner = Runners[rec.runner_type](config=cfg)
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    return runner


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "target",
    "seed",
    "epoch",
    "d_z",
    "d_phi",
    "n_samples",
    "mean_sq_spectral_mu",
    "mean_sq_spectral_std",
    "M_eps",
    "std_err_mu",
    "std_err_std",
    "max_spectral_mu",
    "max_spectral_std",
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
    """Evaluate assumption quantities at all checkpoints for one run."""
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

        epsilon = vi_model.sample_epsilon(num=n_samples)
        bound = evaluate_assumption_bound(vi_model, epsilon)

        duration = time.perf_counter() - t0

        results.append({
            "target": rec.target,
            "seed": rec.seed,
            "epoch": epoch,
            "d_z": bound.d_z,
            "d_phi": bound.d_phi,
            "n_samples": bound.n_samples,
            "mean_sq_spectral_mu": bound.mean_sq_spectral_mu,
            "mean_sq_spectral_std": bound.mean_sq_spectral_std,
            "M_eps": bound.M_eps,
            "std_err_mu": bound.std_err_mu,
            "std_err_std": bound.std_err_std,
            "max_spectral_mu": bound.max_spectral_mu,
            "max_spectral_std": bound.max_spectral_std,
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


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    """Aggregate results across seeds at the final epoch per target, write CSV."""
    import collections

    # Group by (target, epoch)
    groups: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["target"], row["epoch"])].append(row)

    # For each target, pick the max epoch
    target_final: dict[str, int] = {}
    for (target, epoch) in groups:
        if target not in target_final or epoch > target_final[target]:
            target_final[target] = epoch

    summary_rows = []
    for target, final_epoch in sorted(target_final.items()):
        group = groups[(target, final_epoch)]
        M_eps_values = [r["M_eps"] for r in group]
        n_seeds = len(M_eps_values)
        mean_M = np.mean(M_eps_values)
        se_M = np.std(M_eps_values, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
        summary_rows.append({
            "target": target,
            "final_epoch": final_epoch,
            "d_z": group[0]["d_z"],
            "d_phi": group[0]["d_phi"],
            "n_seeds": n_seeds,
            "M_eps_mean": round(float(mean_M), 6),
            "M_eps_se": round(float(se_M), 6),
            "M_eps_max": round(float(max(M_eps_values)), 6),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["target", "final_epoch", "d_z", "d_phi", "n_seeds",
              "M_eps_mean", "M_eps_se", "M_eps_max"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = "campaigns/default_config_grid/manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate bounded-gradient assumption (Assumption 1) on DIVI checkpoints.",
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
        default=256,
        help="Number of epsilon samples for Monte Carlo estimate (default: 256).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/assumption_validation",
        help="Output directory for CSV results (default: %(default)s).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device (default: auto).",
    )
    p.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run sanity check only, then exit.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.sanity_check:
        print("Running sanity check...")
        sanity_check(device="cpu")
        print("Sanity check PASSED.")
        return

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

    # Run sanity check first
    print("Running sanity check...", end=" ")
    sanity_check(device="cpu")
    print("PASSED.")

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
        write_summary(all_results, output_dir / "summary_table.csv")
        print(f"\nResults written to {output_dir}/")
        print(f"  results_full.csv: {len(all_results)} rows")
        print(f"  summary_table.csv: aggregated by target")
    else:
        print("No results produced (checkpoints may be missing locally).",
              file=sys.stderr)


if __name__ == "__main__":
    main()
