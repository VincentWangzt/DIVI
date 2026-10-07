"""Post-hoc evaluation of score fourth moments at each checkpoint.

Loads DIVI checkpoints from the campaign manifest and computes
E_z[||score_p(z)||^4] and E_z[||score_q(z)||^4] at each saved checkpoint,
where z ~ q_phi (the VI model), score_p is the target score, and score_q
is the proxy (reverse/denoising model) score.

Usage::

    # Evaluate x_shaped target, all seeds, all checkpoints
    python -m finalization.eval_score_fourth_moment \\
        --targets x_shaped --n-samples 10240 --device cuda \\
        --output-dir results/score_4th_moment/x_shaped

    # All targets, every 5th checkpoint
    python -m finalization.eval_score_fourth_moment \\
        --checkpoint-stride 5 --n-samples 10240 \\
        --output-dir results/score_4th_moment/full
"""

from __future__ import annotations

import argparse
import csv
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
    scratch = tempfile.mkdtemp(prefix="score_4th_eval_")
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
    "z_dim",
    "n_samples",
    "score_p_4th_moment",
    "score_q_4th_moment",
    "score_diff_l2_fourth",
    "score_diff_l2_second",
    "score_p_mean_norm",
    "score_q_mean_norm",
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
    """Evaluate score fourth moments at all checkpoints for one run."""
    checkpoints = find_all_checkpoints(rec.result_path)
    if not checkpoints:
        logger.warning(f"No checkpoints found for {rec.run_id} at {rec.result_path}")
        return []

    if checkpoint_stride > 1:
        checkpoints = checkpoints[::checkpoint_stride]

    cfg = prepare_config(rec, device=device)
    runner = build_runner_for_eval(rec, cfg)
    vi_model = runner.vi_model
    reverse_model = runner.reverse_model
    target_model = runner.target_model

    results = []
    for epoch, ckpt_path in checkpoints:
        t0 = time.perf_counter()

        # Load VI model
        state = torch.load(ckpt_path, map_location=runner.device)
        vi_model.load_state_dict(state)
        vi_model.eval()

        # Load reverse model
        ckpt_dir = ckpt_path.parent
        rev_ckpt_path = ckpt_dir / "reverse_model.pt"
        if not rev_ckpt_path.is_file():
            logger.warning(
                f"No reverse_model.pt at epoch {epoch} for {rec.run_id}; skipping."
            )
            continue
        rev_state = torch.load(rev_ckpt_path, map_location=runner.device)
        reverse_model.load_state_dict(rev_state)
        reverse_model.eval()

        # Compute score fourth moments
        with torch.no_grad():
            _, z = vi_model.sampling(num=n_samples)
            score_p = target_model.score(z)         # [N, z_dim]
            score_q = reverse_model.score(z)        # [N, z_dim]
            norm_p = torch.norm(score_p, p=2, dim=-1)  # [N]
            norm_q = torch.norm(score_q, p=2, dim=-1)  # [N]
            score_p_4th = torch.mean(norm_p ** 4).item()
            score_q_4th = torch.mean(norm_q ** 4).item()
            score_p_mean_norm = torch.mean(norm_p).item()
            score_q_mean_norm = torch.mean(norm_q).item()
            # E[||score_p - score_q||_2^4]
            diff_sq = torch.sum((score_p - score_q) ** 2, dim=-1)  # ||.||_2^2
            score_diff_l2_fourth = torch.mean(diff_sq ** 2).item()  # E[||.||_2^4]
            score_diff_l2_second = torch.mean(diff_sq).item()  # E[||.||_2^2]

        duration = time.perf_counter() - t0

        results.append({
            "target": rec.target,
            "seed": rec.seed,
            "epoch": epoch,
            "z_dim": int(vi_model.z_dim),
            "n_samples": n_samples,
            "score_p_4th_moment": score_p_4th,
            "score_q_4th_moment": score_q_4th,
            "score_diff_l2_fourth": score_diff_l2_fourth,
            "score_diff_l2_second": score_diff_l2_second,
            "score_p_mean_norm": score_p_mean_norm,
            "score_q_mean_norm": score_q_mean_norm,
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
        description="Evaluate score fourth moments at DIVI checkpoints.",
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
        default=10240,
        help="Number of z samples for Monte Carlo estimate (default: 10240).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/score_4th_moment",
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
