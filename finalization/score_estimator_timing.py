"""Time native estimators on the same frozen DIVI checkpoints and inputs.

The timed implementations come from rebuttal-0726. Checkpoint loading, AISIVI
proposal fitting, input generation, and HMC reference evaluation are excluded.
"""
from __future__ import annotations

import math
import platform
import statistics
import sys
import time
import json
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

from .config import REPO_ROOT, repo_path
from .score_approximation import (
    build_frozen_estimators, evaluation_device, file_sha256, load_score_config,
    mean_and_se, seed_everything, select_checkpoints, shared_input_bank,
    stable_seed, validate_score_config, write_csv,
)

ScoreEstimator = Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]
DEFAULT_CONFIG = REPO_ROOT / "configs/finalization/score_estimator_timing.yaml"



def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def sivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the SIVI mixture score through autograd."""

    auxiliary_count = int(runner.training_reverse_sample_num)
    auxiliary_epsilon = runner.vi_model.sample_epsilon(
        num=auxiliary_count,
    )
    epsilon_aux = auxiliary_epsilon.unsqueeze(0).repeat(
        z.shape[0],
        1,
        1,
    )
    epsilon_aux = torch.cat(
        [epsilon_aux, generating_epsilon.unsqueeze(1)],
        dim=1,
    )
    z_aux = z.detach().unsqueeze(1).repeat(
        1,
        auxiliary_count + 1,
        1,
    )
    z_aux.requires_grad_(True)
    log_terms = runner.vi_model.logp(z_aux, epsilon_aux)
    log_mixture = torch.logsumexp(log_terms, dim=1) - math.log(
        auxiliary_count + 1
    )
    component_gradient = torch.autograd.grad(
        log_mixture.sum(),
        z_aux,
        create_graph=False,
    )[0]
    return component_gradient.sum(dim=1).detach()

def uivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the UIVI score with its native posterior HMC."""

    batch_size, z_dim = z.shape
    epsilon_dim = int(runner.epsilon_dim)
    retained_samples = int(runner.training_reverse_sample_num)
    burn_in_steps = int(runner.hmc_burn_in_steps)
    step_size = float(runner.hmc_step_size)
    leapfrog_steps = int(runner.hmc_leapfrog_steps)
    device = torch.device(runner.device)

    z_aux = z.unsqueeze(1).expand(
        batch_size,
        retained_samples,
        z_dim,
    ).clone().detach()
    epsilon_current = generating_epsilon.clone().detach().to(device)
    samples: list[torch.Tensor] = []
    for step in range(burn_in_steps + retained_samples):
        initial_momentum = torch.randn(
            batch_size,
            epsilon_dim,
            device=device,
        )
        initial_logp = runner._log_q_phi_eps_given_z(
            epsilon_current,
            z,
        )
        initial_kinetic = 0.5 * initial_momentum.square().sum(dim=-1)

        momentum = initial_momentum + 0.5 * step_size * (
            runner._grad_log_q_phi(epsilon_current, z)
        )
        epsilon_proposal = epsilon_current
        for leapfrog_index in range(leapfrog_steps):
            epsilon_proposal = epsilon_proposal + step_size * momentum
            gradient = runner._grad_log_q_phi(epsilon_proposal, z)
            if leapfrog_index != leapfrog_steps - 1:
                momentum = momentum + step_size * gradient
        momentum = momentum + 0.5 * step_size * gradient

        proposal_logp = runner._log_q_phi_eps_given_z(
            epsilon_proposal,
            z,
        )
        proposal_kinetic = 0.5 * momentum.square().sum(dim=-1)
        energy_change = (
            proposal_kinetic - proposal_logp
        ) - (initial_kinetic - initial_logp)
        acceptance_probability = torch.exp(
            (-energy_change).clamp(max=0)
        )
        accept = torch.rand_like(acceptance_probability) < (
            acceptance_probability
        )
        epsilon_current = torch.where(
            accept.unsqueeze(-1),
            epsilon_proposal,
            epsilon_current,
        )
        if step >= burn_in_steps:
            samples.append(epsilon_current.clone().detach())

    epsilon_aux = torch.stack(samples, dim=0).transpose(0, 1)
    with torch.no_grad():
        return runner.vi_model.score(z_aux, epsilon_aux).mean(
            dim=1
        ).detach()

def aisivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the AISIVI score using the proposal fitted to the frozen VI."""

    del generating_epsilon
    sample_count = int(runner.training_reverse_sample_num)
    with torch.no_grad():
        z_aux, epsilon_aux, log_q_reverse = (
            runner.reverse_model.sample(
                z,
                num_samples=sample_count,
            )
        )
        log_importance = (
            runner.vi_model.log_q_epsilon(epsilon_aux) - log_q_reverse
        ).clamp(max=10.0)
    z_aux.requires_grad_(True)
    log_terms = runner.vi_model.logp(
        z_aux,
        epsilon_aux,
    ) + log_importance
    log_mixture = torch.logsumexp(log_terms, dim=1) - math.log(
        sample_count
    )
    component_gradient = torch.autograd.grad(
        log_mixture.sum(),
        z_aux,
        create_graph=False,
    )[0]
    score = component_gradient.sum(dim=1).detach()
    if bool(runner.normalize_reverse_score):
        score = score - score.mean(dim=0, keepdim=True)
    return score

def divi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the DIVI score with one score-network forward pass."""

    del generating_epsilon
    with torch.no_grad():
        score = runner.reverse_model.score(z).detach()
        if bool(runner.normalize_reverse_score):
            score = score - score.mean(dim=0, keepdim=True)
    return score

def estimator_metadata(runner: Any, method: str) -> dict[str, Any]:
    normalized = method.upper()
    if normalized == "SIVI":
        auxiliaries = int(runner.training_reverse_sample_num) + 1
        return {
            "estimator": (
                "prior sampling + mixture logsumexp + autograd score"
            ),
            "native_auxiliary_samples": auxiliaries,
            "hmc_burn_in_steps": None,
            "hmc_leapfrog_steps": None,
        }
    if normalized == "UIVI":
        return {
            "estimator": "posterior HMC + conditional-score mean",
            "native_auxiliary_samples": int(
                runner.training_reverse_sample_num
            ),
            "hmc_burn_in_steps": int(runner.hmc_burn_in_steps),
            "hmc_leapfrog_steps": int(runner.hmc_leapfrog_steps),
        }
    if normalized == "AISIVI":
        return {
            "estimator": (
                "reverse-flow sampling + importance mixture + "
                "autograd score"
            ),
            "native_auxiliary_samples": int(
                runner.training_reverse_sample_num
            ),
            "hmc_burn_in_steps": None,
            "hmc_leapfrog_steps": None,
        }
    if normalized == "DIVI":
        return {
            "estimator": "score-network forward pass",
            "native_auxiliary_samples": 0,
            "hmc_burn_in_steps": None,
            "hmc_leapfrog_steps": None,
        }
    raise ValueError(f"Unsupported method: {method}")

def summarize_timings(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "latency_ms_mean": statistics.fmean(ordered),
        "latency_ms_sd": statistics.stdev(ordered),
        "latency_ms_median": statistics.median(ordered),
        "latency_ms_min": ordered[0],
        "latency_ms_max": ordered[-1],
    }

def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update({
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(
                properties.total_memory
            ),
            "gpu_compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
        })
    return metadata


ESTIMATORS: dict[str, ScoreEstimator] = {
    "SIVI": sivi_score, "UIVI": uivi_score, "AISIVI": aisivi_score, "DIVI": divi_score,
}


def load_timing_config(path: Path | str | None = None, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.merge(load_score_config(), OmegaConf.load(repo_path(path or DEFAULT_CONFIG)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    validate_score_config(cfg)
    sizes = [int(value) for value in cfg.evaluation.batch_sizes]
    if not sizes or len(sizes) != len(set(sizes)) or any(size < 1 for size in sizes):
        raise ValueError("Select unique, positive timing batch sizes.")
    if int(cfg.evaluation.warmup_calls) < 0 or int(cfg.evaluation.timed_calls) < 2:
        raise ValueError("warmup_calls must be nonnegative and timed_calls at least two.")
    return cfg


def benchmark_estimator(
    runner: Any, estimator: ScoreEstimator, *,
    epsilon_bank: torch.Tensor, z_bank: torch.Tensor,
    warmup_calls: int, timed_calls: int,
) -> tuple[list[float], int]:
    """Time score inference on a caller-supplied bank shared by all methods."""
    if warmup_calls < 0 or timed_calls < 2:
        raise ValueError("warmup_calls must be nonnegative and timed_calls at least two.")
    if z_bank.ndim != 3 or epsilon_bank.ndim != 3 or z_bank.shape[:2] != epsilon_bank.shape[:2]:
        raise ValueError("Expected matched input banks [calls,batch,dimension].")
    if z_bank.shape[0] != warmup_calls + timed_calls or z_bank.shape[1] < 1:
        raise ValueError("Input bank must cover exactly the warmup and timed calls.")
    batch_size = z_bank.shape[1]
    device = torch.device(runner.device)
    for index in range(warmup_calls):
        score = estimator(runner, z_bank[index], epsilon_bank[index])
        if not torch.isfinite(score).all():
            raise FloatingPointError("A warmup score estimate contained a non-finite value.")
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms, outputs = [], []
    for index in range(warmup_calls, warmup_calls + timed_calls):
        _synchronize(device)
        started_ns = time.perf_counter_ns()
        score = estimator(runner, z_bank[index], epsilon_bank[index])
        _synchronize(device)
        elapsed_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
        outputs.append(score)
    stacked = torch.stack(outputs)
    if stacked.shape != (timed_calls, batch_size, int(runner.z_dim)):
        raise RuntimeError(f"Unexpected score shape {tuple(stacked.shape)}.")
    if not torch.isfinite(stacked).all():
        raise FloatingPointError("A timed score estimate contained a non-finite value.")
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return elapsed_ms, peak_memory


def aggregate_timings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["target"], row["epoch"], row["method"], row["batch_size"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (target, epoch, method, batch_size), items in sorted(groups.items()):
        mean, se = mean_and_se([row["latency_ms_mean"] for row in items])
        summaries.append({"target": target, "epoch": epoch, "method": method,
                          "batch_size": batch_size, "seed_count": len(items),
                          "latency_ms_mean": mean, "latency_ms_se": se,
                          "latency_ms_per_z": mean / batch_size})
    return summaries


def run_benchmark(cfg: DictConfig) -> list[dict[str, Any]]:
    warmup_calls, timed_calls = int(cfg.evaluation.warmup_calls), int(cfg.evaluation.timed_calls)
    rows, repetitions, provenance = [], [], []
    device = evaluation_device(cfg)
    for checkpoint in select_checkpoints(cfg):
        runners = build_frozen_estimators(cfg, checkpoint)
        source = next(iter(runners.values()))
        provenance.append({"checkpoint_dir": str(checkpoint.checkpoint_dir),
                           "vi_model_sha256": file_sha256(checkpoint.checkpoint_dir / "vi_model.pt"),
                           "divi_score_sha256": file_sha256(checkpoint.checkpoint_dir / "reverse_model.pt"),
                           "aisivi_refit": getattr(runners.get("AISIVI"), "score_refit_metadata", None)})
        for batch_size in cfg.evaluation.batch_sizes:
            batch_size = int(batch_size)
            cell_seed = stable_seed(int(cfg.evaluation.seed), checkpoint.key, batch_size, "timing-inputs")
            epsilon, z = shared_input_bank(source, checkpoint, count=batch_size * (warmup_calls + timed_calls), seed=cell_seed)
            epsilon_bank = epsilon.reshape(warmup_calls + timed_calls, batch_size, -1)
            z_bank = z.reshape(warmup_calls + timed_calls, batch_size, -1)
            for method, runner in runners.items():
                seed_everything(stable_seed(cell_seed, method, "native"), use_cuda=device == "cuda")
                elapsed_ms, peak_memory = benchmark_estimator(
                    runner, ESTIMATORS[method], epsilon_bank=epsilon_bank, z_bank=z_bank,
                    warmup_calls=warmup_calls, timed_calls=timed_calls,
                )
                summary = summarize_timings(elapsed_ms)
                mean = summary["latency_ms_mean"]
                common = {"target": checkpoint.target, "seed": checkpoint.seed, "epoch": checkpoint.epoch,
                          "method": method, "batch_size": batch_size}
                rows.append({**common, **summary, "warmup_calls": warmup_calls, "timed_calls": timed_calls,
                             "timing_seed": cell_seed, "latency_ms_per_z": mean / batch_size,
                             "throughput_z_per_sec": 1000 * batch_size / mean, "peak_memory_bytes": peak_memory,
                             "vi_model_sha256": provenance[-1]["vi_model_sha256"],
                             **estimator_metadata(runner, method)})
                repetitions.extend({**common, "repetition": index + 1, "latency_ms": latency}
                                   for index, latency in enumerate(elapsed_ms))
                print(f"{checkpoint.target} seed={checkpoint.seed} {method} batch={batch_size}: {mean:.4f} ms", flush=True)
        del runners, source
        if device == "cuda":
            torch.cuda.empty_cache()
        output = repo_path(cfg.output.report_dir)
        write_csv(output / "timing_per_checkpoint.csv", rows)
        write_csv(output / "timing_repetitions.csv", repetitions)
        write_csv(output / "timing_summary.csv", aggregate_timings(rows))
        (output / "metadata.json").write_text(json.dumps({
            "protocol": "native estimator inference on common frozen DIVI checkpoints and inputs",
            "timing_definition": "synchronized wall time; excludes loading, proposal fitting, input generation, warmup, reference HMC, and diagnostics",
            "uncertainty": "per-checkpoint SD across calls; summary SE across source-seed means",
            "code_sha256": file_sha256(Path(__file__)),
            "score_setup_sha256": file_sha256(Path(__file__).with_name("score_approximation.py")),
            "environment": environment_metadata(torch.device(device)),
            "config": OmegaConf.to_container(cfg, resolve=True), "checkpoints": provenance,
        }, indent=2, allow_nan=False), encoding="utf-8")
    return rows
