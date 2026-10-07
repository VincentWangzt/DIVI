"""Compare native score estimators on the same frozen DIVI checkpoints.

HMC and estimator implementations are selected from rebuttal-0726. The release
workflow uses only posterior HMC references, one shared variational distribution,
and an AISIVI proposal refitted against that fixed distribution.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from runner.runners import Runners
from .artifacts import completed_runs, find_all_checkpoints, load_manifest, resolve_repo_path
from .config import REPO_ROOT, repo_path
from .runner_eval import remove_file_handlers

METHODS = ("SIVI", "UIVI", "AISIVI", "DIVI")
DEFAULT_CONFIG = REPO_ROOT / "configs/finalization/score_approximation.yaml"

def stable_seed(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)

def seed_everything(seed: int, *, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _conditional_parameters(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(vi_model, "_raw_parameters") or not hasattr(
        vi_model, "_variance_from_raw"
    ):
        raise TypeError(
            "Score analysis requires the ConditionalGaussian VI interface."
        )
    mu, var_raw = vi_model._raw_parameters(epsilon)
    var, _ = vi_model._variance_from_raw(var_raw)
    return mu, var

def conditional_logp_and_score(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu, var = _conditional_parameters(vi_model, epsilon)
    log_var = torch.log(var)
    dimension = z.shape[-1]
    logp = -0.5 * (
        dimension * math.log(2.0 * math.pi)
        + log_var.sum(dim=-1)
        + (((z - mu) ** 2) / var).sum(dim=-1)
    )
    score = -(z - mu) / var
    return logp, score

def diagonal_gaussian_mixture_block(
    z: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log component-sum and score for one shared component bank."""
    if z.ndim != 2 or mu.ndim != 2 or var.shape != mu.shape:
        raise ValueError("Expected z=[N,D] and mu,var=[K,D].")
    if z.shape[-1] != mu.shape[-1]:
        raise ValueError("z and component dimensions do not match.")

    inv_var = var.reciprocal()
    mu_inv_var = mu * inv_var
    component_const = -0.5 * (
        torch.log(var).sum(dim=-1)
        + (mu * mu_inv_var).sum(dim=-1)
        + z.shape[-1] * math.log(2.0 * math.pi)
    )
    log_components = (
        z @ mu_inv_var.transpose(0, 1)
        - 0.5 * (z * z) @ inv_var.transpose(0, 1)
        + component_const.unsqueeze(0)
    )
    log_sum = torch.logsumexp(log_components, dim=1)
    weights = torch.softmax(log_components, dim=1)
    weighted_inv_var = weights @ inv_var
    weighted_mu_inv_var = weights @ mu_inv_var
    score = weighted_mu_inv_var - z * weighted_inv_var
    return log_sum, score

def mixture_block_summary(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        mu, var = _conditional_parameters(vi_model, epsilon)
        return diagonal_gaussian_mixture_block(z, mu, var)

def merge_mixture_summaries(
    left_log_sum: torch.Tensor,
    left_score: torch.Tensor,
    right_log_sum: torch.Tensor,
    right_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge scores for two disjoint, unnormalised mixture component sets."""
    combined_log_sum = torch.logaddexp(left_log_sum, right_log_sum)
    left_weight = torch.exp(left_log_sum - combined_log_sum).unsqueeze(-1)
    right_weight = torch.exp(right_log_sum - combined_log_sum).unsqueeze(-1)
    combined_score = (
        left_weight * left_score + right_weight * right_score
    )
    return combined_log_sum, combined_score

def posterior_log_prob(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Unnormalised ``log q_phi(epsilon | z)`` for posterior HMC."""
    return vi_model.log_q_epsilon(epsilon) + vi_model.logp(z, epsilon)

def posterior_log_prob_and_grad(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the posterior log density and its epsilon gradient."""
    with torch.enable_grad():
        epsilon_grad = epsilon.detach().requires_grad_(True)
        log_prob = posterior_log_prob(vi_model, epsilon_grad, z)
        gradient = torch.autograd.grad(
            log_prob.sum(),
            epsilon_grad,
            create_graph=False,
            retain_graph=False,
        )[0]
    return log_prob.detach(), gradient.detach()

def gelman_rubin_rhat_from_moments(
    chain_means: torch.Tensor,
    chain_m2: torch.Tensor,
    *,
    draws: int,
) -> torch.Tensor:
    """Classical R-hat from per-chain means and centered sums of squares."""
    if chain_means.ndim != 3 or chain_m2.shape != chain_means.shape:
        raise ValueError(
            "chain_means and chain_m2 must both have shape [N,C,D]."
        )
    chains = chain_means.shape[1]
    if chains < 2 or draws < 2:
        raise ValueError("R-hat requires at least two chains and two draws.")

    between = draws * chain_means.var(dim=1, unbiased=True)
    within = (chain_m2 / (draws - 1.0)).mean(dim=1)
    variance_hat = ((draws - 1.0) / draws) * within + between / draws
    positive_within = within > 0
    rhat = torch.empty_like(within)
    rhat[positive_within] = torch.sqrt(
        (variance_hat[positive_within] / within[positive_within]).clamp_min(0)
    )
    both_constant = (~positive_within) & (between == 0)
    rhat[both_constant] = 1.0
    rhat[(~positive_within) & (~both_constant)] = float("inf")
    return rhat

def _finite_tensor_summary(
    values: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float | None]:
    flat = values.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    finite = flat[torch.isfinite(flat)]
    result: dict[str, float | None] = {
        f"{prefix}_nonfinite_fraction": float(
            1.0 - finite.numel() / max(1, flat.numel())
        ),
    }
    if finite.numel() == 0:
        result.update({
            f"{prefix}_median": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
        })
        return result
    result.update({
        f"{prefix}_median": float(finite.median().item()),
        f"{prefix}_p95": float(
            torch.quantile(finite, 0.95).item()
        ),
        f"{prefix}_max": float(finite.max().item()),
    })
    return result

def assess_hmc_reference_quality(
    diagnostics: dict[str, Any],
    quality_cfg: DictConfig,
) -> tuple[str, list[str]]:
    """Apply configured sampler-quality checks without discarding a cell."""
    checks = [
        (
            "hmc_divergence_fraction",
            "<=",
            float(quality_cfg.max_divergence_fraction),
        ),
        (
            "hmc_score_rhat_p95",
            "<=",
            float(quality_cfg.max_score_rhat_p95),
        ),
        (
            "hmc_epsilon_rhat_p95",
            "<=",
            float(quality_cfg.max_epsilon_rhat_p95),
        ),
        (
            "hmc_post_burn_acceptance_rate",
            ">=",
            float(quality_cfg.min_post_burn_acceptance_rate),
        ),
        (
            "hmc_post_burn_acceptance_min",
            ">=",
            float(quality_cfg.min_worst_chain_acceptance_rate),
        ),
    ]
    issues: list[str] = []
    for key, operator, threshold in checks:
        value = diagnostics.get(key)
        if value is None or not math.isfinite(float(value)):
            issues.append(f"{key}=nonfinite")
            continue
        numeric = float(value)
        passed = (
            numeric <= threshold
            if operator == "<="
            else numeric >= threshold
        )
        if not passed:
            issues.append(
                f"{key}={numeric:.6g} {operator} {threshold:.6g} failed"
            )
    for key in ("hmc_epsilon_rhat_nonfinite_fraction", "hmc_score_rhat_nonfinite_fraction"):
        if float(diagnostics.get(key, 0.0)) > 0:
            issues.append(f"{key}={diagnostics[key]:.6g}")
    return ("pass" if not issues else "warning"), issues

def posterior_hmc_reference_scores(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    total_samples: int,
    num_chains: int,
    burn_in_steps: int,
    thinning: int,
    step_size: float,
    leapfrog_steps: int,
    init_jitter_scale: float,
    adapt_step_size: bool,
    target_acceptance: float,
    adaptation_rate: float,
    min_step_size: float,
    max_step_size: float,
    divergence_threshold: float,
    accumulator_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate the marginal score with batched posterior HMC.

    The returned tensor has shape ``[C,N,Dz]``.  Each entry along the first
    axis is a chain-mean score and therefore serves as one independent
    reference replicate in the internal-L2 calculation.
    """
    if z.ndim != 2 or generating_epsilon.ndim != 2:
        raise ValueError("z and generating_epsilon must both be rank two.")
    if z.shape[0] != generating_epsilon.shape[0]:
        raise ValueError("z and generating_epsilon batch sizes must match.")
    if total_samples < 1 or num_chains < 2:
        raise ValueError(
            "HMC requires positive total_samples and at least two chains."
        )
    if total_samples % num_chains != 0:
        raise ValueError("total_samples must be divisible by num_chains.")
    if burn_in_steps < 0 or thinning < 1:
        raise ValueError("Invalid HMC burn-in or thinning.")
    if step_size <= 0 or leapfrog_steps < 1:
        raise ValueError("Invalid HMC step size or leapfrog count.")
    if init_jitter_scale < 0:
        raise ValueError("init_jitter_scale must be non-negative.")
    if not 0 < target_acceptance < 1:
        raise ValueError("target_acceptance must be in (0, 1).")
    if adaptation_rate < 0:
        raise ValueError("adaptation_rate must be non-negative.")
    if not 0 < min_step_size <= step_size <= max_step_size:
        raise ValueError(
            "Require min_step_size <= step_size <= max_step_size."
        )
    if divergence_threshold <= 0:
        raise ValueError("divergence_threshold must be positive.")

    draws_per_chain = total_samples // num_chains
    batch_size, z_dim = z.shape
    epsilon_dim = generating_epsilon.shape[-1]
    device = z.device
    dtype = generating_epsilon.dtype

    z_chains = z.detach().unsqueeze(1).expand(
        batch_size,
        num_chains,
        z_dim,
    )
    epsilon_current = generating_epsilon.detach().unsqueeze(1).expand(
        batch_size,
        num_chains,
        epsilon_dim,
    ).clone()
    # Always consume the jitter draw, including when its scale is zero.  This
    # permits controlled initialization ablations to use common random numbers
    # for all subsequent HMC momenta and accept/reject uniforms.
    jitter = torch.randn_like(epsilon_current)
    jitter[:, 0, :].zero_()
    epsilon_current = (
        epsilon_current + jitter * init_jitter_scale
    )

    log_step = torch.full(
        (batch_size, num_chains, 1),
        math.log(step_size),
        device=device,
        dtype=dtype,
    )
    min_log_step = math.log(min_step_size)
    max_log_step = math.log(max_step_size)
    accepted_sum = torch.zeros(
        batch_size,
        num_chains,
        device=device,
        dtype=torch.float64,
    )
    retained_accept_sum = torch.zeros_like(accepted_sum)
    retained_transitions = 0
    divergence_count = torch.zeros_like(accepted_sum)
    squared_jump_sum = torch.zeros_like(accepted_sum)
    epsilon_mean = torch.zeros(
        batch_size,
        num_chains,
        epsilon_dim,
        device=device,
        dtype=accumulator_dtype,
    )
    epsilon_m2 = torch.zeros_like(epsilon_mean)
    score_mean = torch.zeros(
        batch_size,
        num_chains,
        z_dim,
        device=device,
        dtype=accumulator_dtype,
    )
    score_m2 = torch.zeros_like(score_mean)
    retained_draws = 0

    total_transitions = burn_in_steps + draws_per_chain * thinning
    for transition in range(total_transitions):
        transition_step = log_step.exp()
        epsilon_before = epsilon_current
        momentum_initial = torch.randn_like(epsilon_current)
        log_prob_initial, gradient = posterior_log_prob_and_grad(
            vi_model,
            epsilon_current,
            z_chains,
        )
        kinetic_initial = 0.5 * momentum_initial.square().sum(dim=-1)

        momentum = (
            momentum_initial
            + 0.5 * transition_step * gradient
        )
        epsilon_proposed = epsilon_current
        log_prob_proposed = log_prob_initial
        for leapfrog_index in range(leapfrog_steps):
            epsilon_proposed = (
                epsilon_proposed + transition_step * momentum
            )
            log_prob_proposed, gradient = posterior_log_prob_and_grad(
                vi_model,
                epsilon_proposed,
                z_chains,
            )
            if leapfrog_index != leapfrog_steps - 1:
                momentum = momentum + transition_step * gradient
        momentum = momentum + 0.5 * transition_step * gradient
        kinetic_proposed = 0.5 * momentum.square().sum(dim=-1)

        delta_h = (
            kinetic_proposed - log_prob_proposed
            - kinetic_initial + log_prob_initial
        )
        finite_transition = (
            torch.isfinite(delta_h)
            & torch.isfinite(log_prob_initial)
            & torch.isfinite(log_prob_proposed)
        )
        log_acceptance = torch.where(
            finite_transition,
            (-delta_h).clamp(max=0),
            torch.full_like(delta_h, -torch.inf),
        )
        acceptance_probability = torch.exp(log_acceptance)
        accept = (
            torch.log(torch.rand_like(log_acceptance))
            < log_acceptance
        )
        epsilon_current = torch.where(
            accept.unsqueeze(-1),
            epsilon_proposed,
            epsilon_current,
        ).detach()

        accepted_sum += accept.to(torch.float64)
        divergence_count += (
            (~finite_transition) | (delta_h.abs() > divergence_threshold)
        ).to(torch.float64)
        squared_jump_sum += (
            epsilon_current - epsilon_before
        ).square().sum(dim=-1).to(torch.float64)

        if adapt_step_size and transition < burn_in_steps:
            gain = adaptation_rate / math.sqrt(transition + 1.0)
            log_step = (
                log_step
                + gain
                * (
                    acceptance_probability.detach().unsqueeze(-1)
                    - target_acceptance
                )
            ).clamp(min=min_log_step, max=max_log_step)

        if transition >= burn_in_steps:
            retained_accept_sum += accept.to(torch.float64)
            retained_transitions += 1
            retained_index = transition - burn_in_steps
            if retained_index % thinning == 0:
                with torch.no_grad():
                    epsilon_value = epsilon_current.to(
                        dtype=accumulator_dtype,
                    )
                    score_value = vi_model.score(
                        z_chains,
                        epsilon_current,
                    ).detach().to(
                        dtype=accumulator_dtype,
                    )
                    retained_draws += 1
                    epsilon_delta = epsilon_value - epsilon_mean
                    epsilon_mean += epsilon_delta / retained_draws
                    epsilon_m2 += epsilon_delta * (
                        epsilon_value - epsilon_mean
                    )
                    score_delta = score_value - score_mean
                    score_mean += score_delta / retained_draws
                    score_m2 += score_delta * (
                        score_value - score_mean
                    )

    if retained_draws != draws_per_chain:
        raise RuntimeError(
            "Posterior HMC retained an unexpected number of samples."
        )
    chain_score_means = score_mean.permute(1, 0, 2).contiguous()

    epsilon_rhat = gelman_rubin_rhat_from_moments(
        epsilon_mean,
        epsilon_m2,
        draws=retained_draws,
    )
    score_rhat = gelman_rubin_rhat_from_moments(
        score_mean,
        score_m2,
        draws=retained_draws,
    )
    post_burn_acceptance = retained_accept_sum / max(
        1,
        retained_transitions,
    )
    total_acceptance = accepted_sum / total_transitions
    final_step_size = log_step.exp().squeeze(-1)
    diagnostics: dict[str, Any] = {
        "hmc_num_chains": num_chains,
        "hmc_samples_per_chain": draws_per_chain,
        "hmc_total_samples": total_samples,
        "hmc_burn_in_steps": burn_in_steps,
        "hmc_thinning": thinning,
        "hmc_leapfrog_steps": leapfrog_steps,
        "hmc_acceptance_rate": float(total_acceptance.mean().item()),
        "hmc_post_burn_acceptance_rate": float(
            post_burn_acceptance.mean().item()
        ),
        "hmc_post_burn_acceptance_min": float(
            post_burn_acceptance.min().item()
        ),
        "hmc_divergence_fraction": float(
            divergence_count.sum().item()
            / (total_transitions * batch_size * num_chains)
        ),
        "hmc_mean_squared_jump_distance": float(
            (
                squared_jump_sum
                / total_transitions
            ).mean().item()
        ),
        "hmc_final_step_size_median": float(
            final_step_size.median().item()
        ),
        "hmc_final_step_size_p05": float(
            torch.quantile(final_step_size, 0.05).item()
        ),
        "hmc_final_step_size_p95": float(
            torch.quantile(final_step_size, 0.95).item()
        ),
        **_finite_tensor_summary(
            epsilon_rhat,
            prefix="hmc_epsilon_rhat",
        ),
        **_finite_tensor_summary(
            score_rhat,
            prefix="hmc_score_rhat",
        ),
    }
    return chain_score_means, diagnostics

def native_sivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    auxiliary_count = int(runner.training_reverse_sample_num)
    auxiliary_epsilon = runner.vi_model.sample_epsilon(num=auxiliary_count)
    auxiliary_log_sum, auxiliary_score = mixture_block_summary(
        runner.vi_model,
        z,
        auxiliary_epsilon,
    )
    with torch.no_grad():
        generating_logp, generating_score = conditional_logp_and_score(
            runner.vi_model,
            z,
            generating_epsilon,
        )
        _, score = merge_mixture_summaries(
            auxiliary_log_sum,
            auxiliary_score,
            generating_logp,
            generating_score,
        )
    return score, {"native_auxiliary_samples": auxiliary_count + 1}

def native_uivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    z_aux, epsilon_aux, acceptance_rate = runner.sample_epsilon_hmc(
        z,
        eps_init=generating_epsilon,
        num_samples=int(runner.training_reverse_sample_num),
        burn_in_steps=int(runner.hmc_burn_in_steps),
        step_size=float(runner.hmc_step_size),
        leapfrog_steps=int(runner.hmc_leapfrog_steps),
    )
    with torch.no_grad():
        score = runner.vi_model.score(z_aux, epsilon_aux).mean(dim=1)
    return score, {
        "native_auxiliary_samples": int(runner.training_reverse_sample_num),
        "uivi_hmc_acceptance_rate": float(acceptance_rate),
    }

def _sample_aisivi_reverse_adaptively(
    runner: Any,
    z: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
]:
    sample_count = int(runner.training_reverse_sample_num)

    def sample_chunk(
        count: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[int],
    ]:
        try:
            sampled_z, sampled_epsilon, sampled_log_prob = (
                runner.reverse_model.sample(
                    z,
                    num_samples=count,
                )
            )
            return (
                sampled_z,
                sampled_epsilon,
                sampled_log_prob,
                [count],
            )
        except RuntimeError as error:
            is_reverse_sampling_failure = (
                "Failed to obtain finite samples from RealNVP"
                in str(error)
            )
            if not is_reverse_sampling_failure or count <= 1:
                raise
            left_count = count // 2
            right_count = count - left_count
            left = sample_chunk(left_count)
            right = sample_chunk(right_count)
            return (
                torch.cat([left[0], right[0]], dim=1),
                torch.cat([left[1], right[1]], dim=1),
                torch.cat([left[2], right[2]], dim=1),
                [*left[3], *right[3]],
            )

    return sample_chunk(sample_count)

def _native_aisivi_score_chunk(
    runner: Any,
    z: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_count = int(runner.training_reverse_sample_num)
    with torch.no_grad():
        (
            z_aux,
            epsilon_aux,
            log_q_reverse,
            auxiliary_chunks,
        ) = _sample_aisivi_reverse_adaptively(
            runner,
            z,
        )
        raw_importance = (
            runner.vi_model.log_q_epsilon(epsilon_aux) - log_q_reverse
        )
        if not torch.isfinite(raw_importance).all():
            raise FloatingPointError(
                "AISIVI produced non-finite importance weights."
            )
        clipped_fraction = (raw_importance > 10.0).float().mean()
        importance = raw_importance.clamp(max=10.0)
        conditional_logp, conditional_score = conditional_logp_and_score(
            runner.vi_model,
            z_aux,
            epsilon_aux,
        )
        log_terms = conditional_logp + importance
        finite = torch.isfinite(log_terms)
        if (~finite).all(dim=1).any():
            raise FloatingPointError(
                "AISIVI produced a row without any finite score terms."
            )
        safe_log_terms = torch.where(
            finite,
            log_terms,
            torch.full_like(log_terms, -torch.inf),
        )
        weights = torch.softmax(safe_log_terms, dim=1)
        safe_conditional_score = torch.where(
            finite.unsqueeze(-1),
            conditional_score,
            torch.zeros_like(conditional_score),
        )
        score = (
            weights.unsqueeze(-1) * safe_conditional_score
        ).sum(dim=1)
        if bool(runner.normalize_reverse_score):
            score = score - score.mean(dim=0, keepdim=True)
        ess = weights.square().sum(dim=1).reciprocal()
    return score, {
        "native_auxiliary_samples": sample_count,
        "importance_clipped_fraction": float(clipped_fraction.item()),
        "importance_ess_mean": float(ess.mean().item()),
        "importance_ess_min": float(ess.min().item()),
        "aisivi_min_effective_auxiliary_chunk_size": min(
            auxiliary_chunks
        ),
        "aisivi_auxiliary_split_count": len(auxiliary_chunks) - 1,
    }

def native_aisivi_score(
    runner: Any,
    z: torch.Tensor,
    *,
    z_chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if z_chunk_size is None:
        z_chunk_size = z.shape[0]
    if z_chunk_size < 1:
        raise ValueError("AISIVI z chunk size must be positive.")

    scores: list[torch.Tensor] = []
    diagnostics: list[tuple[int, dict[str, float]]] = []

    def evaluate_with_adaptive_splitting(
        chunk: torch.Tensor,
    ) -> list[tuple[torch.Tensor, dict[str, float]]]:
        try:
            return [_native_aisivi_score_chunk(runner, chunk)]
        except RuntimeError as error:
            is_reverse_sampling_failure = (
                "Failed to obtain finite samples from RealNVP"
                in str(error)
            )
            if not is_reverse_sampling_failure or chunk.shape[0] <= 1:
                raise
            midpoint = chunk.shape[0] // 2
            return [
                *evaluate_with_adaptive_splitting(chunk[:midpoint]),
                *evaluate_with_adaptive_splitting(chunk[midpoint:]),
            ]

    for start in range(0, z.shape[0], z_chunk_size):
        chunk = z[start:start + z_chunk_size]
        chunk_results = evaluate_with_adaptive_splitting(chunk)
        offset = 0
        for chunk_score, chunk_diagnostics in chunk_results:
            effective_size = int(chunk_score.shape[0])
            scores.append(chunk_score)
            diagnostics.append((effective_size, chunk_diagnostics))
            offset += effective_size
        if offset != chunk.shape[0]:
            raise RuntimeError(
                "AISIVI adaptive chunks did not cover the input batch."
            )

    total = sum(size for size, _ in diagnostics)
    configured_chunks = math.ceil(z.shape[0] / z_chunk_size)
    merged = {
        "native_auxiliary_samples": int(
            diagnostics[0][1]["native_auxiliary_samples"]
        ),
        "importance_clipped_fraction": sum(
            size * values["importance_clipped_fraction"]
            for size, values in diagnostics
        ) / total,
        "importance_ess_mean": sum(
            size * values["importance_ess_mean"]
            for size, values in diagnostics
        ) / total,
        "importance_ess_min": min(
            values["importance_ess_min"]
            for _, values in diagnostics
        ),
        "aisivi_z_chunk_size": z_chunk_size,
        "aisivi_min_effective_z_chunk_size": min(
            size for size, _ in diagnostics
        ),
        "aisivi_adaptive_split_count": (
            len(diagnostics) - configured_chunks
        ),
        "aisivi_min_effective_auxiliary_chunk_size": min(
            values["aisivi_min_effective_auxiliary_chunk_size"]
            for _, values in diagnostics
        ),
        "aisivi_auxiliary_split_count": sum(
            int(values["aisivi_auxiliary_split_count"])
            for _, values in diagnostics
        ),
    }
    return torch.cat(scores, dim=0), merged

def native_divi_score(
    runner: Any,
    z: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        score = runner.reverse_model.score(z).detach()
        if bool(runner.normalize_reverse_score):
            score = score - score.mean(dim=0, keepdim=True)
    return score, {"native_auxiliary_samples": 0}

def method_native_score(
    runner: Any,
    method: str,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    aisivi_z_chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    method_upper = method.upper()
    if method_upper == "SIVI":
        return native_sivi_score(runner, z, generating_epsilon)
    if method_upper == "UIVI":
        return native_uivi_score(runner, z, generating_epsilon)
    if method_upper == "AISIVI":
        return native_aisivi_score(
            runner,
            z,
            z_chunk_size=aisivi_z_chunk_size,
        )
    if method_upper == "DIVI":
        return native_divi_score(runner, z)
    raise ValueError(f"Unsupported score-analysis method: {method}")


@dataclass(frozen=True)
class FrozenCheckpoint:
    run_dir: Path
    checkpoint_dir: Path
    config_path: Path
    target: str
    seed: int
    epoch: int

    @property
    def key(self) -> str:
        return fingerprint({"target": self.target, "seed": self.seed, "epoch": self.epoch,
                            "vi_model": file_sha256(self.checkpoint_dir / "vi_model.pt"),
                            "saved_config": file_sha256(self.config_path)})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def score_implementation_hashes() -> dict[str, str]:
    """Include model implementations in cache identity as well as this module."""
    return {"analysis": file_sha256(Path(__file__)),
            "vi_model": file_sha256(REPO_ROOT / "models/vi_model.py"),
            "reverse_model": file_sha256(REPO_ROOT / "models/reverse_model.py"),
            "torch": str(torch.__version__)}


def load_score_config(path: Path | str | None = None, overrides: list[str] | None = None) -> DictConfig:
    config_path = repo_path(path or DEFAULT_CONFIG)
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return validate_score_config(cfg)


def validate_score_config(cfg: DictConfig) -> DictConfig:
    methods = [str(method).upper() for method in cfg.selection.methods]
    if not methods or len(methods) != len(set(methods)) or any(method not in METHODS for method in methods):
        raise ValueError(f"Select unique score estimators from {METHODS}; NFVI defines another variational family.")
    cfg.selection.methods = methods
    if int(cfg.evaluation.forward_batch_size) < 2:
        raise ValueError("forward_batch_size must be at least two.")
    return cfg


def select_checkpoints(cfg: DictConfig) -> list[FrozenCheckpoint]:
    """Use explicit DIVI run directories or the existing standard manifest."""
    targets = set(str(target) for target in cfg.selection.targets)
    seeds = cfg.selection.get("seeds", "auto")
    seed_set = None if seeds == "auto" else {int(seed) for seed in seeds}
    run_dirs = list(cfg.selection.get("run_dirs", []))
    if not run_dirs:
        records = completed_runs(load_manifest(cfg.selection.manifest_path))
        run_dirs = [record.result_path for record in records
                    if record.method.upper() == "DIVI" and record.target in targets
                    and (seed_set is None or record.seed in seed_set)]
    requested = cfg.selection.get("checkpoint_epochs", "all")
    epochs = None if requested == "all" else {int(epoch) for epoch in requested}
    selected: list[FrozenCheckpoint] = []
    seen: set[tuple[str, int, int]] = set()
    for raw in run_dirs:
        run_dir = resolve_repo_path(str(raw))
        if run_dir is None or not run_dir.is_dir():
            raise FileNotFoundError(f"DIVI run directory not found: {raw}")
        snapshot = run_dir / "full_config.yaml"
        if not snapshot.is_file():
            raise FileNotFoundError(f"Saved training configuration is required: {snapshot}")
        saved = OmegaConf.load(snapshot)
        if saved.runner_type != "DIVI":
            raise ValueError(f"Expected a DIVI source run: {run_dir}")
        target, seed = str(saved.target_type), int(saved.seed)
        if target not in targets or (seed_set is not None and seed not in seed_set):
            continue
        available = dict(find_all_checkpoints(run_dir))
        if epochs is not None and not epochs.issubset(available):
            raise FileNotFoundError(f"Missing checkpoint epochs {sorted(epochs - available.keys())} in {run_dir}")
        for epoch, vi_path in sorted(available.items()):
            if epochs is not None and epoch not in epochs:
                continue
            key = (target, seed, epoch)
            if key in seen:
                raise ValueError(f"Duplicate source checkpoint for target/seed/epoch {key}")
            if not (vi_path.parent / "reverse_model.pt").is_file():
                raise FileNotFoundError(f"Matching DIVI score checkpoint is required: {vi_path.parent}")
            seen.add(key)
            selected.append(FrozenCheckpoint(run_dir, vi_path.parent, snapshot, target, seed, epoch))
    if not selected:
        raise ValueError("No DIVI checkpoints match the requested targets, seeds, and epochs.")
    return sorted(selected, key=lambda item: (item.target, item.seed, item.epoch))


def evaluation_device(cfg: DictConfig) -> str:
    requested = str(cfg.evaluation.device)
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("evaluation.device must be auto, cpu, or cuda.")
    return requested


def _absolute_model_paths(config: DictConfig) -> None:
    defaults = {
        "target_config_path": f"configs/targets/{config.target_type}.yaml",
        "vi_model_config_path": f"configs/vi_models/{config.vi_model_type}.yaml",
    }
    if "reverse_model_type" in config:
        defaults["reverse_model_config_path"] = f"configs/reverse_models/{config.reverse_model_type}.yaml"
    for key, default in defaults.items():
        path = resolve_repo_path(str(config.get(key, default)))
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Model configuration not found: {path}")
        config[key] = path.as_posix()


def _build_frozen_runner(cfg: DictConfig, checkpoint: FrozenCheckpoint, method: str) -> Any:
    saved = OmegaConf.load(checkpoint.config_path)
    if str(saved.vi_model_type) != "ConditionalGaussian":
        raise ValueError("This score comparison currently supports the ConditionalGaussian toy family.")
    if method == "DIVI":
        runner_config = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
    else:
        template = str(cfg.estimator_configs[method]).format(target=checkpoint.target)
        path = repo_path(template)
        runner_config = OmegaConf.load(path)
        runner_config = OmegaConf.merge(runner_config, {
            "target": OmegaConf.to_container(saved.target, resolve=True),
            "vi_model": OmegaConf.to_container(saved.vi_model, resolve=True),
            "vi_model_type": str(saved.vi_model_type),
        })
    device = evaluation_device(cfg)
    runner_config.runner_type = method
    runner_config.target_type = checkpoint.target
    runner_config.seed = checkpoint.seed
    runner_config.device = device
    runner_config.use_cuda = device == "cuda"
    # Training snapshots resolve ${device}; override nested literal CUDA values
    # as well, so GPU-trained checkpoints can be evaluated on CPU.
    for component in ("target", "vi_model", "reverse_model", "hmc"):
        if component in runner_config:
            runner_config[component].device = device
    runner_config.config_path = checkpoint.config_path.as_posix()
    runner_config.resume = {"enabled": False}
    if method == "UIVI":
        runner_config.reverse_model_type = "HMC"
    worker = f"{checkpoint.target}/seed{checkpoint.seed}/epoch{checkpoint.epoch}/{method}"
    runner_config.output = {
        "results_dir": str(repo_path(cfg.output.scratch_results_dir) / worker),
        "tb_dir": str(repo_path(cfg.output.scratch_tb_dir) / worker),
    }
    _absolute_model_paths(runner_config)
    seed_everything(stable_seed(checkpoint.key, method, "initialization"), use_cuda=device == "cuda")
    runner = Runners[method](runner_config)
    runner.writer.close()
    remove_file_handlers()
    runner.vi_model.load_state_dict(torch.load(checkpoint.checkpoint_dir / "vi_model.pt", map_location=device, weights_only=True))
    runner.vi_model.eval()
    for parameter in runner.vi_model.parameters():
        parameter.requires_grad_(False)
    runner.curr_epoch = checkpoint.epoch
    if method == "DIVI":
        runner.reverse_model.load_state_dict(torch.load(checkpoint.checkpoint_dir / "reverse_model.pt", map_location=device, weights_only=True))
        runner.reverse_model.eval()
        for parameter in runner.reverse_model.parameters():
            parameter.requires_grad_(False)
    return runner


def _refit_aisivi(runner: Any, cfg: DictConfig, checkpoint: FrozenCheckpoint) -> dict[str, Any]:
    """Fit only the reverse proposal; cache a completed fit for accuracy/timing reuse."""
    fit = cfg.evaluation.aisivi_refit
    steps, batch_size = int(fit.steps), int(fit.batch_size)
    if steps < 1 or batch_size < 1:
        raise ValueError("AISIVI refit steps and batch size must be positive.")
    model_config = OmegaConf.to_container(runner.config, resolve=True)
    for key in ("output", "device", "config_path", "use_cuda"):
        model_config.pop(key, None)
    fit_key = fingerprint({
        "checkpoint": checkpoint.key, "config": model_config,
        "fit": OmegaConf.to_container(fit, resolve=True), "code": score_implementation_hashes(),
    })
    cache = repo_path(cfg.output.cache_dir) / "aisivi" / f"{fit_key}.pt"
    if bool(cfg.evaluation.get("resume", True)) and cache.is_file():
        payload = torch.load(cache, map_location=runner.device, weights_only=True)
        if payload["fingerprint"] != fit_key:
            raise ValueError("AISIVI proposal cache fingerprint mismatch.")
        runner.reverse_model.load_state_dict(payload["state"])
        metadata = payload["metadata"]
    else:
        seed_everything(stable_seed(checkpoint.key, "AISIVI", "refit"), use_cuda=runner.device == "cuda")
        runner.reverse_model.train()
        optimizer = runner.training_reverse_optimizer
        if optimizer is None:
            raise TypeError("AISIVI refitting requires an optimizer-based reverse proposal.")
        started = time.perf_counter()
        for step in range(steps):
            epsilon, z = runner.vi_model.sampling(num=batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = -runner.reverse_model.log_prob(epsilon, z).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite AISIVI refit loss at step {step + 1}.")
            loss.backward()
            if runner.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(runner.reverse_model.parameters(), runner.grad_clip)
            optimizer.step()
            if runner.training_reverse_scheduler is not None:
                runner.training_reverse_scheduler.step()
        metadata = {"steps": steps, "batch_size": batch_size,
                    "loss": float(loss.detach()), "fit_time_sec": time.perf_counter() - started}
    if not all(torch.isfinite(value).all() for value in runner.reverse_model.state_dict().values()):
        raise FloatingPointError("AISIVI proposal contains non-finite parameters.")
    if not (bool(cfg.evaluation.get("resume", True)) and cache.is_file()):
        save_cache(cache, {"fingerprint": fit_key, "state": runner.reverse_model.state_dict(), "metadata": metadata})
    runner.reverse_model.eval()
    for parameter in runner.reverse_model.parameters():
        parameter.requires_grad_(False)
    return {**metadata, "fingerprint": fit_key, "proposal_sha256": file_sha256(cache)}


def build_frozen_estimators(cfg: DictConfig, checkpoint: FrozenCheckpoint) -> dict[str, Any]:
    runners: dict[str, Any] = {}
    for method in cfg.selection.methods:
        runner = _build_frozen_runner(cfg, checkpoint, str(method))
        if method == "AISIVI":
            runner.score_refit_metadata = _refit_aisivi(runner, cfg, checkpoint)
        runners[str(method)] = runner
    return runners


def shared_input_bank(runner: Any, checkpoint: FrozenCheckpoint, *, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 1:
        raise ValueError("Input sample count must be positive.")
    seed_everything(stable_seed(seed, checkpoint.key, "inputs"), use_cuda=runner.device == "cuda")
    with torch.no_grad():
        epsilon, z = runner.vi_model.sampling(num=count)
    if not torch.isfinite(epsilon).all() or not torch.isfinite(z).all():
        raise FloatingPointError("Frozen checkpoint produced non-finite samples.")
    return epsilon, z


def compute_score_metrics(method_score: torch.Tensor, reference_scores: torch.Tensor) -> dict[str, float]:
    if reference_scores.ndim != 3 or method_score.shape != reference_scores.shape[1:]:
        raise ValueError("Expected method [N,D] and HMC chain scores [C,N,D].")
    if reference_scores.shape[0] < 2:
        raise ValueError("At least two HMC chains are required.")
    if not torch.isfinite(reference_scores).all() or not torch.isfinite(method_score).all():
        raise FloatingPointError("Score calculation produced non-finite values.")
    reference_mean = reference_scores.mean(0)
    internal = (reference_scores - reference_mean).square().sum(-1).mean()
    error = (method_score.to(reference_mean.dtype) - reference_mean).square().sum(-1)
    return {"method_l2": float(error.mean()), "reference_internal_l2": float(internal),
            "reference_mean_mcse_l2": float(internal / (reference_scores.shape[0] - 1))}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    """Only publish a complete cache file, so interrupted fits can be retried."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def mean_and_se(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=True) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def summarize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["target"], row["epoch"], row["method"]), []).append(row)
    summaries = []
    for (target, epoch, method), items in sorted(groups.items()):
        mean, se = mean_and_se([row["method_l2"] for row in items])
        summaries.append({"target": target, "epoch": epoch, "method": method,
                          "seed_count": len(items), "method_l2_mean": mean, "method_l2_se": se,
                          "reference_quality_warnings": sum(row["reference_quality"] != "pass" for row in items)})
    return summaries


def run_analysis(cfg: DictConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    for checkpoint in select_checkpoints(cfg):
        print(f"Frozen DIVI: {checkpoint.target}, seed {checkpoint.seed}, epoch {checkpoint.epoch}", flush=True)
        runners = build_frozen_estimators(cfg, checkpoint)
        source = next(iter(runners.values()))
        reference_cfg = cfg.evaluation.reference
        epsilon, z = shared_input_bank(source, checkpoint, count=int(cfg.evaluation.forward_batch_size), seed=int(cfg.evaluation.seed))
        reference_key = fingerprint({"checkpoint": checkpoint.key,
                                     "evaluation": OmegaConf.to_container(cfg.evaluation, resolve=True),
                                     "code": score_implementation_hashes()})
        cache = repo_path(cfg.output.cache_dir) / "hmc" / f"{reference_key}.pt"
        if bool(cfg.evaluation.get("resume", True)) and cache.is_file():
            payload = torch.load(cache, map_location=source.device, weights_only=True)
            if payload["fingerprint"] != reference_key:
                raise ValueError("HMC cache fingerprint mismatch.")
            reference_scores, diagnostics = payload["scores"], payload["diagnostics"]
        else:
            seed_everything(stable_seed(checkpoint.key, "reference", int(cfg.evaluation.seed)), use_cuda=source.device == "cuda")
            kwargs = OmegaConf.to_container(reference_cfg, resolve=True)
            kwargs.pop("quality", None)
            kwargs["accumulator_dtype"] = getattr(torch, str(kwargs["accumulator_dtype"]))
            reference_scores, diagnostics = posterior_hmc_reference_scores(source.vi_model, z, epsilon, **kwargs)
            save_cache(cache, {"fingerprint": reference_key, "scores": reference_scores,
                               "diagnostics": diagnostics})
        quality, issues = assess_hmc_reference_quality(diagnostics, reference_cfg.quality)
        checkpoint_metadata.append({"checkpoint_dir": str(checkpoint.checkpoint_dir),
                                    "reference_fingerprint": reference_key,
                                    "aisivi_refit": getattr(runners.get("AISIVI"), "score_refit_metadata", None)})
        for method, runner in runners.items():
            seed_everything(stable_seed(checkpoint.key, method, "estimate"), use_cuda=source.device == "cuda")
            score, method_diagnostics = method_native_score(
                runner, method, z, epsilon, aisivi_z_chunk_size=int(cfg.evaluation.aisivi_z_chunk_size),
            )
            metrics = compute_score_metrics(score, reference_scores)
            rows.append({"target": checkpoint.target, "seed": checkpoint.seed, "epoch": checkpoint.epoch,
                         "method": method, **metrics, "reference_quality": quality,
                         "reference_quality_issues": json.dumps(issues), **diagnostics, **method_diagnostics,
                         "vi_model_sha256": file_sha256(checkpoint.checkpoint_dir / "vi_model.pt"),
                         "divi_score_sha256": file_sha256(checkpoint.checkpoint_dir / "reverse_model.pt")})
        output = repo_path(cfg.output.report_dir)
        write_csv(output / "checkpoint_metrics.csv", rows)
        write_csv(output / "score_summary.csv", summarize_scores(rows))
        (output / "metadata.json").write_text(json.dumps({
            "protocol": "native estimators on common frozen DIVI checkpoints; posterior HMC reference",
            "code_sha256": file_sha256(Path(__file__)), "config": OmegaConf.to_container(cfg, resolve=True),
            "checkpoints": checkpoint_metadata,
        }, indent=2, allow_nan=False), encoding="utf-8")
        del runners
    return rows
