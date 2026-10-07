from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.reverse_model import BaseReverseConditionalModel
from models.vi_model import BaseVIModel

from .proposal import (
    gaussian_reverse_cache,
    gaussian_reverse_mean,
    module_dtype,
    reverse_model_log_prob,
)
from .types import ELMEstimate, KDEELMEstimate, LogQEstimate


LOG_ZERO = float("-inf")
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def load_baseline_sample_store(path: Path) -> torch.Tensor:
    """Load a saved baseline sample tensor from either a raw tensor or dict store."""
    samples = torch.load(path, map_location="cpu")
    if isinstance(samples, dict):
        if "samples" not in samples:
            raise KeyError(f"Sample store {path} does not contain a 'samples' key.")
        samples = samples["samples"]
    samples = torch.as_tensor(samples, dtype=torch.float32)
    if samples.ndim != 2:
        raise ValueError(
            f"Baseline samples must have shape [num_samples, z_dim], got {tuple(samples.shape)}."
        )
    if samples.shape[0] < 1:
        raise ValueError("Baseline sample store is empty.")
    return samples


def sample_reference_samples(
    baseline_samples: np.ndarray | torch.Tensor,
    num_ref_samples: int,
    device: torch.device | str,
    *,
    replace: bool = False,
) -> torch.Tensor:
    """Choose the reference points used in the outer ELM expectation.

    ELM estimates

        E_{z ~ r}[log q_phi(z)],

    where ``r`` is represented in this codebase by stored baseline samples,
    typically MCMC samples. This helper returns up to ``num_ref_samples`` rows
    with shape ``[N_ref, z_dim]`` on the requested device. The estimators below
    treat these points as fixed conditioning locations.
    """
    if baseline_samples is None:
        raise RuntimeError("Baseline samples are required for expected log marginal.")
    if num_ref_samples < 1:
        raise ValueError("num_ref_samples must be at least 1.")

    # Keep the outer expectation cheap by subsampling the baseline store when it
    # is larger than the requested reference budget.
    total = int(baseline_samples.shape[0])
    if total > num_ref_samples:
        indices = np.random.choice(total, num_ref_samples, replace=replace)
        reference_samples = baseline_samples[indices]
    else:
        reference_samples = baseline_samples
    return torch.as_tensor(reference_samples, device=device)


def summarize_tensor(values: torch.Tensor, prefix: str) -> dict[str, float]:
    """Return finite-value summaries for diagnostic vectors such as ESS.

    ``values`` is usually a one-dimensional tensor with one diagnostic per
    reference sample. Non-finite entries are ignored so that a few failed
    reference points do not hide the usable part of the diagnostic report.
    """
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.1, 0.5, 0.9], device=finite.device, dtype=finite.dtype),
    )
    return {
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_p10": float(quantiles[0].item()),
        f"{prefix}_median": float(quantiles[1].item()),
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_p90": float(quantiles[2].item()),
        f"{prefix}_max": float(finite.max().item()),
    }


def elm_stats(log_q_values: torch.Tensor) -> tuple[float, float, float, int]:
    """Summarize per-reference log marginal estimates.

    Args:
        log_q_values: Tensor with shape ``[N_ref]`` containing estimates of
            ``log q_phi(z_i)`` for fixed reference samples ``z_i``.

    Returns:
        ``(mean, stderr, std, count)`` over finite entries. The mean is the
        scalar ELM estimate, and ``stderr`` measures variability across the
        reference samples rather than the inner importance-sampling noise.
    """
    finite = log_q_values[torch.isfinite(log_q_values)]
    count = int(finite.numel())
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(finite.mean().item())
    std = float(finite.std(unbiased=True).item()) if count > 1 else 0.0
    stderr = std / math.sqrt(count)
    return mean, stderr, std, count


def _resolve_kde_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("KDE dtype must be torch.float32 or torch.float64.")
        return dtype
    dtype_name = str(dtype).lower()
    if dtype_name in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    if dtype_name in {"float64", "double", "fp64", "torch.float64"}:
        return torch.float64
    raise ValueError(f"Unsupported KDE dtype: {dtype!r}.")


def _kde_oom_message(exc: BaseException) -> RuntimeError:
    return RuntimeError(
        "CUDA ran out of memory while evaluating KDE expected log marginal. "
        "Retry with smaller --dim-chunk, --ref-chunk, or --model-chunk."
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


@torch.no_grad()
def kde_expected_log_marginal(
    reference_samples: torch.Tensor | np.ndarray,
    model_samples: torch.Tensor | np.ndarray,
    *,
    dim_chunk: int = 25,
    ref_chunk: int = 500,
    model_chunk: int = 20000,
    min_bandwidth: float = 1.0e-6,
    dtype: torch.dtype | str = torch.float32,
    device: torch.device | str | None = None,
) -> KDEELMEstimate:
    r"""Estimate paper-style expected log marginal with coordinate-wise KDE.

    The estimator fits one Gaussian KDE per coordinate from generated model
    samples and evaluates the sum of coordinate log densities at fixed
    reference samples:

        (1 / N) * sum_i sum_j log \hat q_j(x_ij).

    Model-sample chunks are combined exactly in log space. Chunk sizes only
    control peak memory.
    """
    if dim_chunk < 1:
        raise ValueError("dim_chunk must be at least 1.")
    if ref_chunk < 1:
        raise ValueError("ref_chunk must be at least 1.")
    if model_chunk < 1:
        raise ValueError("model_chunk must be at least 1.")
    if min_bandwidth <= 0.0:
        raise ValueError("min_bandwidth must be positive.")

    kde_dtype = _resolve_kde_dtype(dtype)
    if device is None:
        if isinstance(model_samples, torch.Tensor):
            kde_device = model_samples.device
        else:
            kde_device = torch.device("cpu")
    else:
        kde_device = torch.device(device)
    if kde_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("KDE device is CUDA, but CUDA is not available.")

    start = time.perf_counter()
    try:
        ref = torch.as_tensor(
            reference_samples,
            device=kde_device,
            dtype=kde_dtype,
        )
        model = torch.as_tensor(
            model_samples,
            device=kde_device,
            dtype=kde_dtype,
        )
    except Exception as exc:
        if _is_cuda_oom(exc):
            raise _kde_oom_message(exc) from exc
        raise

    if ref.ndim != 2:
        raise ValueError(f"reference_samples must have shape [N, D], got {tuple(ref.shape)}.")
    if model.ndim != 2:
        raise ValueError(f"model_samples must have shape [M, D], got {tuple(model.shape)}.")
    if ref.shape[1] != model.shape[1]:
        raise ValueError(
            "reference_samples and model_samples must have the same dimension, "
            f"got {ref.shape[1]} and {model.shape[1]}."
        )
    if model.shape[0] < 1:
        raise ValueError("model_samples must contain at least one sample.")
    if ref.shape[0] < 1:
        raise ValueError("reference_samples must contain at least one sample.")

    n_ref = int(ref.shape[0])
    n_model = int(model.shape[0])
    z_dim = int(ref.shape[1])

    # Scott's factor for a one-dimensional KDE is M^{-1/5}. The unbiased
    # coordinate std matches scipy.stats.gaussian_kde's 1D covariance.
    if n_model > 1:
        coord_std = model.std(dim=0, unbiased=True)
    else:
        coord_std = torch.zeros(z_dim, device=kde_device, dtype=kde_dtype)
    raw_bandwidths = coord_std * (float(n_model) ** (-1.0 / 5.0))
    finite_positive_bw = torch.isfinite(raw_bandwidths) & (raw_bandwidths > 0)
    min_bw_tensor = torch.full_like(raw_bandwidths, float(min_bandwidth))
    bandwidths = torch.where(
        finite_positive_bw,
        torch.maximum(raw_bandwidths, min_bw_tensor),
        min_bw_tensor,
    )
    clamped_mask = ~finite_positive_bw | (raw_bandwidths < float(min_bandwidth))

    per_ref_log_values = torch.zeros(n_ref, device=kde_device, dtype=kde_dtype)
    log_norm_model = math.log(float(n_model))

    try:
        for dim_start in range(0, z_dim, dim_chunk):
            dim_stop = min(dim_start + dim_chunk, z_dim)
            dim_slice = slice(dim_start, dim_stop)
            model_dim = model[:, dim_slice].transpose(0, 1).contiguous()
            bandwidth_dim = bandwidths[dim_slice]
            log_bandwidth_dim = torch.log(bandwidth_dim)

            for ref_start in range(0, n_ref, ref_chunk):
                ref_stop = min(ref_start + ref_chunk, n_ref)
                ref_dim = ref[ref_start:ref_stop, dim_slice].transpose(0, 1).contiguous()
                log_kernel_sum = torch.full(
                    (dim_stop - dim_start, ref_stop - ref_start),
                    LOG_ZERO,
                    device=kde_device,
                    dtype=kde_dtype,
                )

                for model_start in range(0, n_model, model_chunk):
                    model_stop = min(model_start + model_chunk, n_model)
                    model_part = model_dim[:, model_start:model_stop]
                    scaled_diff = (
                        ref_dim.unsqueeze(2) - model_part.unsqueeze(1)
                    ) / bandwidth_dim.view(-1, 1, 1)
                    kernel_log = -0.5 * scaled_diff.square()
                    log_kernel_sum = torch.logaddexp(
                        log_kernel_sum,
                        torch.logsumexp(kernel_log, dim=2),
                    )

                log_density = (
                    log_kernel_sum
                    - log_norm_model
                    - log_bandwidth_dim.unsqueeze(1)
                    - HALF_LOG_2PI
                )
                per_ref_log_values[ref_start:ref_stop] += log_density.sum(dim=0)
    except Exception as exc:
        if _is_cuda_oom(exc):
            raise _kde_oom_message(exc) from exc
        raise

    elapsed = time.perf_counter() - start
    finite_values = per_ref_log_values[torch.isfinite(per_ref_log_values)]
    finite_count = int(finite_values.numel())
    if finite_count == 0:
        value = float("nan")
        stderr = float("nan")
        std = float("nan")
        min_value = float("nan")
        max_value = float("nan")
    else:
        value = float(finite_values.mean().item())
        std = float(finite_values.std(unbiased=True).item()) if finite_count > 1 else 0.0
        stderr = std / math.sqrt(finite_count)
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())

    diagnostics: dict[str, Any] = {
        "estimator": "kde_expected_log_marginal",
        "bandwidth_rule": "scott",
        "num_ref_samples": n_ref,
        "num_model_samples": n_model,
        "z_dim": z_dim,
        "dtype": str(kde_dtype).replace("torch.", ""),
        "device": str(kde_device),
        "dim_chunk": int(dim_chunk),
        "ref_chunk": int(ref_chunk),
        "model_chunk": int(model_chunk),
        "min_bandwidth": float(min_bandwidth),
        "num_bandwidth_clamped_dims": int(clamped_mask.sum().item()),
        "runtime_sec": float(elapsed),
        "num_finite_ref_log_values": finite_count,
        "std_across_refs": float(std),
        "min_per_ref_log": float(min_value),
        "max_per_ref_log": float(max_value),
    }

    return KDEELMEstimate(
        value=value,
        stderr=stderr,
        per_reference_log_values=per_ref_log_values.detach().cpu(),
        diagnostics=diagnostics,
        bandwidths=bandwidths.detach().cpu(),
    )


@torch.no_grad()
def estimate_log_q_prior(
    vi_model: BaseVIModel,
    reference_samples: torch.Tensor,
    *,
    num_samples: int,
    epsilon_batch_size: int,
) -> LogQEstimate:
    """Estimate ``log q_phi(z)`` by direct Monte Carlo over the epsilon prior.

    Mathematically, the marginal variational density is

        q_phi(z) = E_{epsilon ~ p(epsilon)}[q_phi(z | epsilon)].

    For each fixed reference sample ``z_i`` this function draws ``K`` auxiliary
    variables from the VI prior and computes

        log hat q_K(z_i) =
            log((1 / K) * sum_k q_phi(z_i | epsilon_k)).

    Tensor shapes:
        - ``reference_samples``: ``[N_ref, z_dim]``.
        - ``epsilon`` per chunk: ``[chunk, epsilon_dim]``.
        - expanded ``z`` and ``epsilon``: ``[N_ref, chunk, dim]``.
        - returned ``log_q_values``: ``[N_ref]``.

    The estimate is computed with ``logsumexp`` for numerical stability. Since
    the logarithm is outside the inner average, finite ``K`` estimates are
    Jensen-biased downward when ``q_phi(z | epsilon)`` is highly variable.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if epsilon_batch_size < 1:
        raise ValueError("epsilon_batch_size must be at least 1.")

    # Estimation is read-only. Preserve the caller's training mode while
    # disabling stochastic training layers such as dropout, if present.
    was_training = vi_model.training
    vi_model.eval()
    start = time.perf_counter()
    reference_samples = reference_samples.to(
        device=getattr(vi_model, "device", reference_samples.device),
        dtype=module_dtype(vi_model),
    )  # shape: [N_ref, z_dim]
    n_ref = int(reference_samples.shape[0])  # scalar: N_ref
    chunk_log_means: list[torch.Tensor] = []
    chunk_sizes: list[int] = []
    remaining = num_samples

    while remaining > 0:
        # Process epsilon samples in chunks so the logical MC budget can be
        # large without materializing an ``[N_ref, K, dim]`` tensor at once.
        current = min(epsilon_batch_size, remaining)
        epsilon = vi_model.sample_epsilon(num=current)  # shape: [current, epsilon_dim]

        # Broadcast fixed reference points against the current epsilon chunk.
        # ``logp`` returns log q_phi(z_i | epsilon_k), shape [N_ref, current].
        z_expanded = reference_samples.unsqueeze(1).expand(
            -1,
            current,
            -1,
        )  # shape: [N_ref, current, z_dim]
        eps_expanded = epsilon.unsqueeze(0).expand(
            n_ref,
            -1,
            -1,
        )  # shape: [N_ref, current, epsilon_dim]
        log_q = vi_model.logp(
            z_expanded,
            eps_expanded,
        )  # shape: [N_ref, current]

        # Store log of the chunk mean density for each z_i:
        # log((1/current) sum_k q_phi(z_i | epsilon_k)).
        chunk_log_mean = torch.logsumexp(log_q, dim=1) - math.log(current)
        # shape: [N_ref], one log mean density per reference sample.
        chunk_log_means.append(chunk_log_mean)
        chunk_sizes.append(current)
        remaining -= current

    # Combine chunk means with their actual chunk sizes. This preserves the
    # same estimate regardless of how ``num_samples`` is split into batches.
    stacked = torch.stack(chunk_log_means, dim=1)  # shape: [N_ref, n_chunks]
    weights = torch.tensor(
        chunk_sizes,
        device=stacked.device,
        dtype=stacked.dtype,
    )  # shape: [n_chunks]
    log_q_values = (
        torch.logsumexp(stacked + torch.log(weights).unsqueeze(0), dim=1)
        - math.log(num_samples)
    )  # shape: [N_ref]
    mean, stderr, log_q_std, finite_count = elm_stats(log_q_values)
    runtime_sec = time.perf_counter() - start
    if was_training:
        vi_model.train()

    return LogQEstimate(
        log_q_values=log_q_values.detach().cpu(),
        stderr=stderr,
        diagnostics={
            "estimator": "prior",
            "num_ref_samples": int(n_ref),
            "num_finite_ref_log_q": int(finite_count),
            "num_samples": int(num_samples),
            "epsilon_batch_size": int(epsilon_batch_size),
            "runtime_sec": float(runtime_sec),
            "log_q_mean": float(mean),
            "log_q_std": float(log_q_std),
        },
        ess_values=None,
    )


@torch.no_grad()
def estimate_log_q_reverse_is(
    vi_model: BaseVIModel,
    reverse_model: BaseReverseConditionalModel,
    reference_samples: torch.Tensor,
    *,
    num_is_samples: int,
    is_batch_size: int = 1024,
    proposal_cache: dict[str, torch.Tensor] | None = None,
) -> LogQEstimate:
    """Estimate ``log q_phi(z)`` with a fitted reverse importance proposal.

    Direct prior sampling can miss the small set of epsilon values that make a
    sharp conditional density ``q_phi(z | epsilon)`` large at a fixed reference
    point. Reverse-IS instead samples

        epsilon_ik ~ q_psi(epsilon | z_i)

    from a fitted reverse proposal and uses the unbiased density estimator

        hat q_K(z_i) = (1 / K) * sum_k
            q_phi(z_i | epsilon_ik) p(epsilon_ik)
            / q_psi(epsilon_ik | z_i).

    This function returns ``log hat q_K(z_i)`` for every reference point, then
    ``summarize_elm`` averages those values to produce the scalar ELM.

    Tensor shapes:
        - ``reference_samples``: ``[N_ref, z_dim]``.
        - ``z_aux`` per chunk: ``[N_ref, chunk, z_dim]``.
        - sampled ``epsilon``: ``[N_ref, chunk, epsilon_dim]``.
        - ``log_weight``: ``[N_ref, chunk]``.
        - returned ``log_q_values`` and ``ess_values``: ``[N_ref]``.

    The implementation accumulates sums in log space and treats non-finite
    weights as zero contribution. For a Gaussian proposal, ``proposal_cache``
    stores the conditional parameters and avoids repeated distribution setup.
    """
    if num_is_samples < 1:
        raise ValueError("num_is_samples must be at least 1.")
    if is_batch_size < 1:
        raise ValueError("is_batch_size must be at least 1.")

    # Match the VI model's device/dtype and preserve both modules' modes.
    device = getattr(vi_model, "device", reference_samples.device)
    vi_dtype = module_dtype(vi_model)
    was_vi_training = vi_model.training
    was_rev_training = reverse_model.training
    vi_model.eval()
    reverse_model.eval()

    reference_samples = reference_samples.to(
        device=device,
        dtype=vi_dtype,
    )  # shape: [N_ref, z_dim]
    n_ref = int(reference_samples.shape[0])  # scalar: N_ref
    dtype = reference_samples.dtype
    start = time.perf_counter()

    finite_weight_count = 0
    total_weight_count = 0

    # Gaussian reverse proposals have closed-form sampling/log-density code
    # below. Other proposal families use their model methods directly.
    gaussian_cache = proposal_cache or gaussian_reverse_cache(reverse_model, dtype=dtype)
    log_weight_sum = torch.full(
        (n_ref,),
        LOG_ZERO,
        device=device,
        dtype=dtype,
    )  # shape: [N_ref], stores log(sum_k w_ik)
    log_weight_square_sum = torch.full(
        (n_ref,),
        LOG_ZERO,
        device=device,
        dtype=dtype,
    )  # shape: [N_ref], stores log(sum_k w_ik^2)

    for is_start in range(0, num_is_samples, is_batch_size):
        # Chunk over the inner IS budget. ``z_aux`` repeats each reference
        # sample across the chunk dimension so each z_i gets its own epsilons.
        chunk_n = min(is_batch_size, num_is_samples - is_start)
        z_aux = reference_samples.unsqueeze(1).expand(
            n_ref,
            chunk_n,
            -1,
        )  # shape: [N_ref, chunk_n, z_dim]

        if gaussian_cache is None:
            # Generic path: ask the reverse model to sample epsilon | z.
            _, epsilon, _ = reverse_model.sample(
                reference_samples,
                num_samples=chunk_n,
            )  # epsilon shape: [N_ref, chunk_n, epsilon_dim]
        else:
            # Fast Gaussian path: epsilon = A z + b + L eta, eta ~ N(0, I).
            mean_reverse = gaussian_reverse_mean(
                z_aux,
                gaussian_cache,
            )  # shape: [N_ref, chunk_n, epsilon_dim]
            noise = torch.randn_like(mean_reverse)  # same shape as mean_reverse
            epsilon = mean_reverse + torch.matmul(
                noise,
                gaussian_cache["chol"].transpose(0, 1),
            )  # shape: [N_ref, chunk_n, epsilon_dim]

        # log weight = log q_phi(z | eps) + log p(eps) - log q_psi(eps | z).
        # Averaging exp(log_weight) over eps estimates q_phi(z).
        log_prior = vi_model.log_q_epsilon(epsilon)  # shape: [N_ref, chunk_n]
        log_proposal = reverse_model_log_prob(
            reverse_model,
            epsilon,
            z_aux,
            proposal_cache=gaussian_cache,
        )  # shape: [N_ref, chunk_n]
        log_conditional = vi_model.logp(
            z_aux,
            epsilon,
        )  # shape: [N_ref, chunk_n]
        log_weight = log_conditional + log_prior - log_proposal
        # shape: [N_ref, chunk_n]
        finite_mask = torch.isfinite(log_weight)  # shape: [N_ref, chunk_n]
        finite_weight_count += int(finite_mask.sum().item())
        total_weight_count += int(log_weight.numel())

        # Replace NaN/+inf/-inf weights with log(0) before logsumexp so they
        # cannot poison the entire reference point.
        safe_log_weight = torch.where(
            finite_mask,
            log_weight,
            torch.full_like(log_weight, LOG_ZERO),
        )  # shape: [N_ref, chunk_n]

        # Accumulate sum_k w_k and sum_k w_k^2 in log space across chunks.
        log_weight_sum = torch.logaddexp(
            log_weight_sum,
            torch.logsumexp(safe_log_weight, dim=1),
        )  # shape: [N_ref]
        log_weight_square_sum = torch.logaddexp(
            log_weight_square_sum,
            torch.logsumexp(2.0 * safe_log_weight, dim=1),
        )  # shape: [N_ref]

    # Convert sum_k w_k into log((1/K) sum_k w_k). ESS is computed per z_i as
    # (sum_k w_k)^2 / sum_k w_k^2, then clamped to the possible range [0, K].
    log_q_values = log_weight_sum - math.log(num_is_samples)  # shape: [N_ref]
    ess_values = torch.exp(
        2.0 * log_weight_sum - log_weight_square_sum
    )  # shape: [N_ref]
    ess_values = torch.where(
        torch.isfinite(ess_values),
        ess_values.clamp(min=0.0, max=float(num_is_samples)),
        torch.zeros_like(ess_values),
    )  # shape: [N_ref]
    mean, stderr, log_q_std, finite_count = elm_stats(log_q_values)
    runtime_sec = time.perf_counter() - start
    if was_vi_training:
        vi_model.train()
    if was_rev_training:
        reverse_model.train()

    diagnostics: dict[str, Any] = {
        "estimator": "reverse_is",
        "proposal_type": type(reverse_model).__name__,
        "num_ref_samples": int(n_ref),
        "num_finite_ref_log_q": int(finite_count),
        "num_is_samples": int(num_is_samples),
        "is_batch_size": int(is_batch_size),
        "finite_weight_count": int(finite_weight_count),
        "total_weight_count": int(total_weight_count),
        "finite_weight_fraction": (
            float(finite_weight_count / total_weight_count)
            if total_weight_count
            else float("nan")
        ),
        "runtime_sec": float(runtime_sec),
        "log_q_mean": float(mean),
        "log_q_std": float(log_q_std),
        "used_gaussian_fast_path": bool(gaussian_cache is not None),
    }
    diagnostics.update(summarize_tensor(ess_values, "ess"))

    return LogQEstimate(
        log_q_values=log_q_values.detach().cpu(),
        stderr=stderr,
        diagnostics=diagnostics,
        ess_values=ess_values.detach().cpu(),
    )


def summarize_elm(log_q: LogQEstimate) -> ELMEstimate:
    """Average per-reference log marginal values into the scalar ELM.

    ``log_q.log_q_values`` contains estimates of ``log q_phi(z_i)``. This
    function computes their finite-sample mean, which estimates
    ``E_{z ~ r}[log q_phi(z)]``, and forwards diagnostics such as ESS.
    """
    value, stderr, _, _ = elm_stats(log_q.log_q_values)
    diagnostics = dict(log_q.diagnostics)
    diagnostics["stderr"] = float(stderr)
    diagnostics["value"] = float(value)
    return ELMEstimate(
        value=value,
        stderr=stderr,
        log_q_values=log_q.log_q_values,
        diagnostics=diagnostics,
        ess_values=log_q.ess_values,
    )
