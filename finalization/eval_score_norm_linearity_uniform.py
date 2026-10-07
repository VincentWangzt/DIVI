"""Post-hoc evaluation of score-norm linearity with uniform-sphere z sampling.

Loads DIVI checkpoints from the campaign manifest and computes

    log max(||nabla_z log p(z)||_2, ||s_psi(z)||_2) - log(||z||_2 + 1)

using z samples drawn uniformly on the sphere (uniform radius, uniform
direction) rather than from the learned VI model.  This gives unbiased
coverage across all norm values, revealing growth behavior in the tails.

Usage (standalone)::

    python -m finalization.eval_score_norm_linearity_uniform \\
        --targets x_shaped Langevin_post --checkpoints 2000 5000 10000 \\
        --n-samples 10000 --device auto \\
        --output-dir campaigns/default_config_grid/generated_reports/finalization

Usage (via finalization pipeline)::

    python scripts/run_finalization.py --only score_norm_linearity_uniform_grid
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
# Uniform sphere sampling
# ---------------------------------------------------------------------------

DEFAULT_MAX_RADIUS: dict[str, float] = {
    "x_shaped": 10.0,
    "Langevin_post": 20.0,
}


def _sample_uniform_sphere(
    n_samples: int,
    z_dim: int,
    max_radius: float,
    device: str,
) -> torch.Tensor:
    """Sample z = r * u, where r ~ Uniform(0, max_radius), u ~ Uniform(S^{d-1}).

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    z_dim : int
        Dimensionality of z.
    max_radius : float
        Maximum radius for the uniform distribution.
    device : str
        Torch device.

    Returns
    -------
    torch.Tensor
        Shape ``[n_samples, z_dim]``.
    """
    # Random direction: sample standard Gaussian, normalize to unit sphere
    directions = torch.randn(n_samples, z_dim, device=device)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    # Random radius: uniform in [0, max_radius]
    radii = torch.rand(n_samples, device=device) * max_radius
    return radii.unsqueeze(-1) * directions


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
def _evaluate_single_checkpoint_uniform(
    runner: Any,
    vi_state: dict,
    reverse_state: dict,
    n_samples: int,
    z_dim: int,
    max_radius: float,
    device: str,
) -> dict[str, np.ndarray]:
    """Evaluate score-norm linearity at a single checkpoint using uniform z.

    Returns dict with keys: z_norm, score_gt_norm, score_proxy_norm,
    max_score_norm, log_ratio.
    Each is a 1-D numpy array of shape [n_samples].
    """
    runner.vi_model.load_state_dict(vi_state)
    runner.reverse_model.load_state_dict(reverse_state)
    runner.vi_model.eval()
    runner.reverse_model.eval()

    # Sample z uniformly on sphere with uniform radius
    z = _sample_uniform_sphere(n_samples, z_dim, max_radius, device)

    # Ground truth score: nabla_z log p(z)
    score_gt = runner.target_model.score(z)
    # Proxy score: s_psi(z) from the denoising model
    score_proxy = runner.reverse_model.score(z)

    # Per-sample norms
    gt_norm = torch.linalg.norm(score_gt, dim=-1)        # [N]
    proxy_norm = torch.linalg.norm(score_proxy, dim=-1)  # [N]
    max_norm = torch.maximum(gt_norm, proxy_norm)         # [N]
    z_norm = torch.linalg.norm(z, dim=-1)                # [N]

    # log max(||score_gt||, ||score_proxy||) - log(||z|| + 1)
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
    max_radius: float,
) -> list[dict[str, Any]]:
    """Evaluate score-norm linearity (uniform z) at selected checkpoints for one run."""
    all_ckpts = find_all_checkpoints(rec.result_path)
    selected = _select_checkpoints(all_ckpts, target_epochs)
    if not selected:
        logger.warning("No valid checkpoints for %s at %s", rec.run_id, rec.result_path)
        return []

    cfg = prepare_config(rec, device=device)
    runner = build_runner_for_eval(rec, cfg)

    # Determine z_dim from the runner's target model (set during runner __init__)
    z_dim = int(runner.config.target.z_dim)

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

        eval_results = _evaluate_single_checkpoint_uniform(
            runner, vi_state, reverse_state, n_samples, z_dim, max_radius, runner.device,
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


def evaluate_score_norm_linearity_uniform(
    records: list[RunRecord],
    cfg: Any,
) -> Path:
    """Evaluate score-norm linearity (uniform z) for selected DIVI runs.

    Returns the path to the output CSV.
    """
    eval_cfg = cfg.evaluation.score_norm_linearity_uniform
    target_epochs = [int(e) for e in eval_cfg.checkpoints]
    n_samples = int(eval_cfg.n_samples)
    device = str(cfg.evaluation.get("device", "auto"))
    overwrite = bool(eval_cfg.get("overwrite", False))
    max_radius_cfg = dict(eval_cfg.get("max_radius", {}))

    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    csv_path = root / "score_norm_linearity_uniform_results.csv"

    if csv_path.exists() and not overwrite:
        logger.info("Score norm linearity uniform CSV already exists at %s; skipping.", csv_path)
        return csv_path

    # Filter to DIVI only
    divi_records = [r for r in records if r.runner_type.upper() == "DIVI"]
    if not divi_records:
        logger.warning("No DIVI records found for score norm linearity uniform evaluation.")
        write_csv([], csv_path)
        return csv_path

    logger.info(
        "Evaluating score norm linearity (uniform z): %d DIVI runs, epochs=%s, n_samples=%d",
        len(divi_records), target_epochs, n_samples,
    )

    all_results: list[dict[str, Any]] = []
    for rec in tqdm(divi_records, desc="Score norm linearity (uniform)"):
        max_radius = float(max_radius_cfg.get(rec.target, DEFAULT_MAX_RADIUS.get(rec.target, 10.0)))
        try:
            run_results = evaluate_run(
                rec,
                device=device,
                n_samples=n_samples,
                target_epochs=target_epochs,
                max_radius=max_radius,
            )
            all_results.extend(run_results)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", rec.run_id, e)
        except Exception as e:
            logger.error("Error evaluating %s: %s", rec.run_id, e, exc_info=True)

    write_csv(all_results, csv_path)
    logger.info("Score norm linearity uniform results written to %s (%d rows).", csv_path, len(all_results))
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = "campaigns/default_config_grid/manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate score-norm linearity (uniform z) on DIVI checkpoints.",
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
        default=10000,
        help="Number of z samples per checkpoint (default: %(default)s).",
    )
    p.add_argument(
        "--max-radius",
        nargs="*",
        default=None,
        help="Per-target max radius as TARGET:RADIUS pairs (e.g., x_shaped:10 Langevin_post:20). "
             "Defaults to 10 for x_shaped, 20 for Langevin_post.",
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


def _parse_max_radius(raw: list[str] | None) -> dict[str, float]:
    """Parse max-radius CLI arg like ['x_shaped:10', 'Langevin_post:20']."""
    if not raw:
        return dict(DEFAULT_MAX_RADIUS)
    result = dict(DEFAULT_MAX_RADIUS)
    for item in raw:
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid max-radius format: {item!r}. Expected TARGET:RADIUS.")
        result[parts[0]] = float(parts[1])
    return result


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

    max_radius_map = _parse_max_radius(args.max_radius)

    print(f"Evaluating score norm linearity (uniform z) for {len(divi_records)} DIVI runs...")
    print(f"  Max radius map: {max_radius_map}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    all_results: list[dict[str, Any]] = []
    for rec in tqdm(divi_records, desc="Runs"):
        max_radius = max_radius_map.get(rec.target, 10.0)
        try:
            run_results = evaluate_run(
                rec,
                device=args.device,
                n_samples=args.n_samples,
                target_epochs=args.checkpoints,
                max_radius=max_radius,
            )
            all_results.extend(run_results)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", rec.run_id, e)
        except Exception as e:
            logger.error("Error evaluating %s: %s", rec.run_id, e, exc_info=True)

    csv_path = output_dir / "score_norm_linearity_uniform_results.csv"
    if all_results:
        write_csv(all_results, csv_path)
        print(f"\nResults written to {csv_path} ({len(all_results)} rows)")
    else:
        print("No results produced.", file=sys.stderr)


if __name__ == "__main__":
    main()
