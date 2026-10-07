"""Generate exact baseline samples for toy 2D targets.

Draws 100k samples (seed=42) for each target with an analytic sampler
and saves them under baselines/exact/<target>_exact_100k.pt.

Usage:
    python scripts/generate_exact_baselines.py
    python scripts/generate_exact_baselines.py --seed 42 --num-samples 100000
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.target_models import target_distribution

EXACT_TARGETS = ["banana", "multimodal", "8_gaussians", "x_shaped", "student_uc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, default="baselines/exact")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    for name in EXACT_TARGETS:
        torch.manual_seed(args.seed)
        target = target_distribution[name](device=device)
        samples = target.sample(args.num_samples).detach().cpu()

        out_path = os.path.join(args.output_dir, f"{name}_exact_100k.pt")
        torch.save(
            {"samples": samples, "target": name, "source": "exact_sampler",
             "num_samples": args.num_samples, "seed": args.seed},
            out_path,
        )
        print(f"[{name}] {samples.shape} -> {out_path}")


if __name__ == "__main__":
    main()
