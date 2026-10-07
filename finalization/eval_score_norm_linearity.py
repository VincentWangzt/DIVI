"""Post-hoc evaluation of score-norm linearity.

Loads DIVI checkpoints from the campaign manifest and computes

    log max(‖∇_z log p(z)‖₂, ‖s_ψ(z)‖₂) − log(‖z‖₂ + 1)

at selected training checkpoints, outputting results as CSV.

If both the target score and the proxy score have at most linear growth
in ‖z‖, then the above quantity should remain bounded above uniformly
over z.

Usage (standalone)::

    python -m finalization.eval_score_norm_linearity \\
        --targets x_shaped Langevin_post --checkpoints 2000 5000 10000 \\
        --n-samples 2048 --device auto \\
        --output-dir campaigns/default_config_grid/generated_reports/finalization

Usage (via finalization pipeline)::

    python scripts/run_finalization.py --only score_norm_linearity_grid
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

from runner.runners import Runners
from utils.logging import get_logger

from .artifacts import (
    RunRecord,
    completed_runs,
    find_all_checkpoints,
    load_manifest,
    normalize_target,
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
# CSV schema
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "target",
    "seed",
    "epoch",
    "sample_idx",
    "z_norm",
    "score_gt_norm",
    "score_proxy_norm",
    "max_score_norm",
    "log_ratio",
    "run_id",
]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write results to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _select_checkpoints(
    all_checkpoints: list[tuple[int, Path]],
    target_epochs: list[int],
) -> list[tuple[int, Path]]:
    """Filter checkpoints to only the requested epochs."""
    epoch_to_path = {epoch: path for epoch, path in all_checkpoints}
    selected = []
    for ep in target_epochs:
        if ep in epoch_to_path:
            selected.append((ep, epoch_to_path[ep]))
        else:
            logger.warning(
                "Checkpoint epoch %d not found; skipping.", ep,
            )
    return selected


@torch.no_grad()
def _evaluate_single_checkpoint(
    runner: Any,
    vi_state: dict,
    reverse_state: dict,
    n_samples: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Evaluate score-norm linearity at a single checkpoint.

    Returns dict with keys: z_norm, score_gt_norm, score_proxy_norm,
    max_score_norm, log_ratio.
    Each is a 1-D numpy array of shape [n_samples].
    """
    runner.vi_model.load_state_dict(vi_state)
    runner.reverse_model.load_state_dict(reverse_state)
    runner.vi_model.eval()
    runner.reverse_model.eval()

    # Sample z from current VI model
    _eps, z = runner.vi_model.sampling(num=n_samples)
    z = z.to(device)

    # Ground truth score: ∇_z log p(z)
    score_gt = runner.target_model.score(z)
    # Proxy score: s_ψ(z) from the denoising model
    score_proxy = runner.reverse_model.score(z)

    # Per-sample norms
    gt_norm = torch.linalg.norm(score_gt, dim=-1)        # [N]
    proxy_norm = torch.linalg.norm(score_proxy, dim=-1)  # [N]
    max_norm = torch.maximum(gt_norm, proxy_norm)         # [N]
    z_norm = torch.linalg.norm(z, dim=-1)                # [N]

    # log max(‖score_gt‖, ‖score_proxy‖) − log(‖z‖ + 1)
    log_ratio = torch.log(max_norm + 1e-10) - torch.log(z_norm + 1.0)

    return {
        "z_norm": z_norm.cpu().numpy(),
        "score_gt_norm": gt_norm.cpu().numpy(),
        "score_proxy_norm": proxy_norm.cpu().numpy(),
        "max_score_norm": max_norm.cpu().numpy(),
        "log_ratio": log_ratio.cpu().numpy(),
    }


def evaluate_run(
    rec: RunRecord,
    *,
    device: str,
    n_samples: int,
    target_epochs: list[int],
) -> list[dict[str, Any]]:
    """Evaluate score-norm linearity at selected checkpoints for one run."""
    all_ckpts = find_all_checkpoints(rec.result_path)
    selected = _select_checkpoints(all_ckpts, target_epochs)
    if not selected:
        logger.warning("No valid checkpoints for %s at %s", rec.run_id, rec.result_path)
        return []

    cfg = prepare_config(rec, device=device)
    runner = build_runner_for_eval(rec, cfg)

    results: list[dict[str, Any]] = []
    for epoch, vi_model_path in selected:
        ckpt_dir = vi_model_path.parent
        reverse_model_path = ckpt_dir / "reverse_model.pt"

        if not reverse_model_path.exists():
            logger.warning(
                "reverse_model.pt not found at %s; skipping epoch %d.",
                ckpt_dir, epoch,
            )
            continue

        t0 = time.perf_counter()

        vi_state = torch.load(vi_model_path, map_location=runner.device)
        reverse_state = torch.load(reverse_model_path, map_location=runner.device)

        eval_results = _evaluate_single_checkpoint(
            runner, vi_state, reverse_state, n_samples, runner.device,
        )

        duration = time.perf_counter() - t0
        n = len(eval_results["z_norm"])

        for i in range(n):
            results.append({
                "target": rec.target,
                "seed": rec.seed,
                "epoch": epoch,
                "sample_idx": i,
                "z_norm": float(eval_results["z_norm"][i]),
                "score_gt_norm": float(eval_results["score_gt_norm"][i]),
                "score_proxy_norm": float(eval_results["score_proxy_norm"][i]),
                "max_score_norm": float(eval_results["max_score_norm"][i]),
                "log_ratio": float(eval_results["log_ratio"][i]),
                "run_id": rec.run_id,
            })

        logger.info(
            "  epoch %d: %d samples in %.1fs (median log_ratio=%.3f)",
            epoch, n, duration, float(np.median(eval_results["log_ratio"])),
        )

    # Cleanup
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# High-level entry point (called by run_finalization.py)
# ---------------------------------------------------------------------------


def evaluate_score_norm_linearity(
    records: list[RunRecord],
    cfg: Any,
) -> Path:
    """Evaluate score-norm linearity for selected DIVI runs at target checkpoints.

    Returns the path to the output CSV.
    """
    eval_cfg = cfg.evaluation.score_norm_linearity
    target_epochs = [int(e) for e in eval_cfg.checkpoints]
    n_samples = int(eval_cfg.n_samples)
    device = str(cfg.evaluation.get("device", "auto"))
    overwrite = bool(eval_cfg.get("overwrite", False))

    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    csv_path = root / "score_norm_linearity_results.csv"

    if csv_path.exists() and not overwrite:
        logger.info("Score norm linearity CSV already exists at %s; skipping.", csv_path)
        return csv_path

    # Filter to DIVI only
    divi_records = [r for r in records if r.runner_type.upper() == "DIVI"]
    if not divi_records:
        logger.warning("No DIVI records found for score norm linearity evaluation.")
        write_csv([], csv_path)
        return csv_path

    logger.info(
        "Evaluating score norm linearity: %d DIVI runs, epochs=%s, n_samples=%d",
        len(divi_records), target_epochs, n_samples,
    )

    all_results: list[dict[str, Any]] = []
    for rec in tqdm(divi_records, desc="Score norm linearity"):
        try:
            run_results = evaluate_run(
                rec,
                device=device,
                n_samples=n_samples,
                target_epochs=target_epochs,
            )
            all_results.extend(run_results)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", rec.run_id, e)
        except Exception as e:
            logger.error("Error evaluating %s: %s", rec.run_id, e, exc_info=True)

    write_csv(all_results, csv_path)
    logger.info("Score norm linearity results written to %s (%d rows).", csv_path, len(all_results))
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = "campaigns/default_config_grid/manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate score-norm linearity assumption on DIVI checkpoints.",
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
        default=["x_shaped", "Langevin_post"],
        help="Target names to evaluate (default: %(default)s).",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42],
        help="Seeds to evaluate (default: %(default)s).",
    )
    p.add_argument(
        "--checkpoints",
        nargs="*",
        type=int,
        default=[2000, 5000, 10000],
        help="Checkpoint epochs to evaluate (default: %(default)s).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=2048,
        help="Number of z samples per checkpoint (default: %(default)s).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="campaigns/default_config_grid/generated_reports/finalization",
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

    manifest_path = repo_path(args.manifest)
    if manifest_path is None or not manifest_path.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)
    manifest = load_manifest(args.manifest)
    records = completed_runs(manifest)

    # Filter
    divi_records = [r for r in records if r.runner_type.upper() == "DIVI"]
    if args.targets:
        target_set = set(args.targets)
        divi_records = [r for r in divi_records if r.target in target_set]
    if args.seeds:
        seed_set = set(args.seeds)
        divi_records = [r for r in divi_records if r.seed in seed_set]

    if not divi_records:
        print("No matching DIVI runs after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating score norm linearity for {len(divi_records)} DIVI runs...")

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
                target_epochs=args.checkpoints,
            )
            all_results.extend(run_results)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", rec.run_id, e)
        except Exception as e:
            logger.error("Error evaluating %s: %s", rec.run_id, e, exc_info=True)

    csv_path = output_dir / "score_norm_linearity_results.csv"
    if all_results:
        write_csv(all_results, csv_path)
        print(f"\nResults written to {csv_path} ({len(all_results)} rows)")
    else:
        print("No results produced.", file=sys.stderr)


if __name__ == "__main__":
    main()
