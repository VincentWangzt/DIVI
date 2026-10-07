from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from models.target_models import target_distribution  # noqa: E402
from utils.logging import get_logger, set_file_handler  # noqa: E402
from utils.mcmc import SGLDConfig, SGLDSampler  # noqa: E402


def _set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for SGLD but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _float_or_none(value: str) -> float | None:
    if value.lower() in {"none", "null", "off"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive, finite, or 'none'")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate saved target samples with SGLD.")
    parser.add_argument("--target", default="Langevin_post", choices=sorted(target_distribution.keys()))
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--step-size", type=float, default=1.0e-4)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--num-chains", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-grad-norm", type=_float_or_none, default=1000.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")

    device = _resolve_device(args.device)
    _set_seed(args.seed, device.type == "cuda")
    target = target_distribution[args.target](device=device)
    z_dim = int(getattr(target, "z_dim", getattr(target, "dim", 2)))

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "baseline_sgld" / args.target / timestamp
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.pt"
    metadata_path = output_dir / "metadata.json"
    if not args.overwrite:
        existing = [path for path in (samples_path, metadata_path) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output files already exist. Pass --overwrite: {joined}")

    logger = get_logger("sgld_baseline")
    set_file_handler(output_dir.as_posix(), filename="run.log")
    logger.info("Starting SGLD baseline | target=%s | device=%s", args.target, device)
    logger.info(
        "Config: step_size=%s, num_samples=%s, burn_in=%s, thinning=%s, num_chains=%s, seed=%s, max_grad_norm=%s",
        args.step_size,
        args.num_samples,
        args.burn_in,
        args.thinning,
        args.num_chains,
        args.seed,
        args.max_grad_norm,
    )

    cfg = SGLDConfig(
        step_size=args.step_size,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        num_chains=args.num_chains,
        seed=args.seed,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )
    sampler = SGLDSampler(score_fn=target.score, dim=z_dim, cfg=cfg)

    start = time.perf_counter()
    samples = sampler.sample(progress_bar=True)
    elapsed = time.perf_counter() - start
    samples = samples.to(dtype=torch.float32, device="cpu")
    metadata = {
        "sampler": "SGLD",
        "target": args.target,
        "num_samples": int(samples.shape[0]),
        "z_dim": int(samples.shape[1]),
        "step_size": float(args.step_size),
        "burn_in": int(args.burn_in),
        "thinning": int(args.thinning),
        "num_chains": int(args.num_chains),
        "seed": int(args.seed),
        "device": str(device),
        "max_grad_norm": args.max_grad_norm,
        "runtime_sec": float(elapsed),
        "sample_mean_abs": float(samples.mean(dim=0).abs().mean().item()),
        "sample_std_mean": float(samples.std(dim=0, unbiased=True).mean().item()),
    }
    torch.save({"samples": samples, "metadata": metadata}, samples_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("Saved SGLD samples to %s, shape=%s, elapsed=%.3fs", samples_path, tuple(samples.shape), elapsed)

    if args.plot and hasattr(target, "trace_plot"):
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        try:
            target.trace_plot(
                samples,
                figpath=figures_dir.as_posix(),
                figname="trace.png",
                figtitle=f"{args.target} SGLD",
            )
            logger.info("Saved trace plot to %s", figures_dir / "trace.png")
        except Exception as exc:
            logger.warning("Trace plot failed: %s", exc)

    print(f"Wrote {samples_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
