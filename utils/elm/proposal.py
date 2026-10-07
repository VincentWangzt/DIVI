from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from models.reverse_model import (
    BaseReverseConditionalModel,
    ConditionalGaussianReverse,
    ConditionalMixtureOfGaussianReverse,
    ReverseModel,
)
from models.vi_model import BaseVIModel
from utils.logging import get_logger

from .types import ReverseProposalFit


logger = get_logger()

PROPOSAL_TYPE_ALIASES = {
    "gaussian": "ConditionalGaussianReverse",
    "mog": "ConditionalMixtureOfGaussianReverse",
    "realnvp": "ConditionalRealNVP",
}
PROPOSAL_TYPE_CANONICAL_NAMES = {
    value: key for key, value in PROPOSAL_TYPE_ALIASES.items()
}


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    """Infer the dtype a module uses for numeric work.

    ELM evaluation must put reference samples and proposal tensors in the same
    dtype as the VI or reverse model. Parameters are preferred; buffers cover
    direct-fit proposals such as the Gaussian model, whose fitted statistics are
    stored as buffers rather than trainable parameters.
    """
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.dtype
    return torch.float32


def canonical_reverse_model_type(proposal_type: str) -> tuple[str, str]:
    """Map a user-facing proposal alias to the registered reverse-model class.

    The CLI accepts compact aliases such as ``gaussian`` and ``realnvp``. The
    model registry in ``models.reverse_model`` uses class-like names, so this
    helper returns both the registry key and the short alias used in diagnostics.
    """
    canonical = PROPOSAL_TYPE_ALIASES.get(proposal_type, proposal_type)
    if canonical not in ReverseModel:
        supported = ", ".join(sorted(PROPOSAL_TYPE_ALIASES))
        raise ValueError(
            f"Unsupported proposal_type={proposal_type!r}. Supported aliases: {supported}."
        )
    return canonical, PROPOSAL_TYPE_CANONICAL_NAMES.get(canonical, canonical)


def default_reverse_model_config_path(canonical_type: str) -> Path:
    """Return the default YAML config path for a registered reverse model."""
    return Path("configs") / "reverse_models" / f"{canonical_type}.yaml"


def resolve_reverse_model_config(
    vi_model: BaseVIModel,
    proposal_type: str,
    proposal_config_path: str | Path | None,
) -> tuple[str, str, Path, DictConfig, dict[str, Any]]:
    """Load and specialize a reverse-proposal config for the current VI model.

    Reverse proposals estimate ``q_psi(epsilon | z)`` and therefore need the
    current VI dimensions ``epsilon_dim`` and ``z_dim``. The config files keep
    those fields templated for normal experiment configs; this resolver writes
    concrete values, chooses the VI device, and returns both a ``DictConfig``
    for model construction and a plain dict for JSON diagnostics.
    """
    canonical_type, proposal_alias = canonical_reverse_model_type(proposal_type)
    config_path = (
        Path(proposal_config_path)
        if proposal_config_path is not None
        else default_reverse_model_config_path(canonical_type)
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Reverse proposal config not found: {config_path}")

    # Specialize the proposal config to the already-built VI model. This keeps
    # standalone ELM evaluation independent from the original experiment YAML.
    device = getattr(vi_model, "device", next(vi_model.parameters()).device)
    raw_config = OmegaConf.load(config_path)
    raw_config.z_dim = int(vi_model.z_dim)
    raw_config.epsilon_dim = int(vi_model.epsilon_dim)
    raw_config.device = str(device)
    if "logit" in raw_config:
        # Uniform epsilon models need the RealNVP proposal to use the same
        # bounded-support transform as the VI model.
        raw_config.logit = bool(getattr(vi_model, "uniform", False))
    resolved_container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved_container, dict):
        raise RuntimeError(
            f"Resolved reverse proposal config must be a mapping: {config_path}"
        )
    resolved_cfg: DictConfig = OmegaConf.create(resolved_container)  # type: ignore[assignment]
    resolved_cfg.device = str(device)
    return canonical_type, proposal_alias, config_path, resolved_cfg, resolved_container


def gaussian_reverse_cache(
    reverse_model: torch.nn.Module,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    """Cache closed-form conditional-Gaussian quantities for fast reverse-IS.

    A fitted ``ConditionalGaussianReverse`` stores a joint Gaussian over
    ``[epsilon, z]``. Conditioning gives

        epsilon | z ~ N(A z + b, cond_cov).

    Reverse-IS repeatedly needs samples and log probabilities from this same
    conditional Gaussian, so the cache stores ``A``, ``b``, the Cholesky factor
    of ``cond_cov``, the precision matrix, and ``log det(cond_cov)``. Non-
    Gaussian proposals return ``None`` and use their own model methods.
    """
    if not isinstance(reverse_model, ConditionalGaussianReverse):
        return None
    A, b, cond_cov = reverse_model._conditional_params()
    # A shape: [epsilon_dim, z_dim]; b shape: [epsilon_dim];
    # cond_cov shape: [epsilon_dim, epsilon_dim].
    A = A.to(dtype=dtype)
    b = b.to(dtype=dtype)
    cond_cov = cond_cov.to(dtype=dtype)
    chol = torch.linalg.cholesky(cond_cov)  # shape: [epsilon_dim, epsilon_dim]
    precision = torch.cholesky_inverse(chol)  # shape: [epsilon_dim, epsilon_dim]
    log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()  # scalar
    return {
        "A": A,
        "b": b,
        "chol": chol,
        "precision": precision,
        "log_det": log_det,
    }


def gaussian_reverse_mean(
    z: torch.Tensor,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate the cached Gaussian conditional mean ``A z + b``.

    ``z`` may have leading dimensions such as ``[N_ref, K, z_dim]``; the final
    dimension is the latent dimension and all leading dimensions are preserved.
    """
    # Input shape: [..., z_dim]; output shape: [..., epsilon_dim].
    return torch.matmul(z, cache["A"].transpose(0, 1)) + cache["b"]


def gaussian_reverse_log_prob(
    epsilon: torch.Tensor,
    z: torch.Tensor,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute ``log q_psi(epsilon | z)`` using the cached Gaussian factors."""
    mean = gaussian_reverse_mean(z, cache)  # shape: [..., epsilon_dim]
    diff = epsilon - mean  # shape: [..., epsilon_dim]
    quad = torch.einsum(
        "...i,ij,...j->...",
        diff,
        cache["precision"],
        diff,
    )  # shape: [...]
    dim = epsilon.shape[-1]  # scalar: epsilon_dim
    return -0.5 * (
        dim * torch.log(torch.tensor(2.0 * torch.pi, device=epsilon.device, dtype=epsilon.dtype))
        + cache["log_det"]
        + quad
    )


def reverse_model_log_prob(
    reverse_model: BaseReverseConditionalModel,
    epsilon: torch.Tensor,
    z: torch.Tensor,
    proposal_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Evaluate proposal log density while preserving caller tensor layout.

    Args:
        reverse_model: Fitted model for ``q_psi(epsilon | z)``.
        epsilon: Samples with shape ``[..., epsilon_dim]``.
        z: Conditioning values with shape ``[..., z_dim]``.
        proposal_cache: Optional Gaussian cache from ``gaussian_reverse_cache``.

    Returns:
        Tensor with shape ``[...]`` containing ``log q_psi(epsilon | z)``. The
        generic path flattens leading dimensions because the reverse models are
        batch-oriented, then reshapes back to match the estimator tensors.
    """
    if proposal_cache is not None:
        return gaussian_reverse_log_prob(epsilon, z, proposal_cache)

    # Generic proposal models expect a 2-D batch. Convert dtype/device to the
    # proposal model, then restore the estimator's dtype/device after scoring.
    model_dtype = module_dtype(reverse_model)
    epsilon_flat = epsilon.reshape(-1, epsilon.shape[-1]).to(
        device=reverse_model.device,
        dtype=model_dtype,
    )  # shape: [prod(...), epsilon_dim]
    z_flat = z.reshape(-1, z.shape[-1]).to(
        device=reverse_model.device,
        dtype=model_dtype,
    )  # shape: [prod(...), z_dim]
    log_prob = reverse_model.log_prob(epsilon_flat, z_flat)
    # shape: [prod(...)]
    return log_prob.reshape(epsilon.shape[:-1]).to(
        device=epsilon.device,
        dtype=epsilon.dtype,
    )  # shape: [...]


def sample_vi_joint(
    vi_model: BaseVIModel,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw joint samples ``(epsilon, z)`` from the fitted VI model.

    These samples are the training data for reverse proposals: the proposal
    learns the conditional density of the auxiliary variable given the produced
    latent value, ``q_psi(epsilon | z)``.
    """
    with torch.no_grad():
        epsilon_samples, z_samples = vi_model.sampling(num=num_samples)
        # epsilon_samples shape: [num_samples, epsilon_dim]
        # z_samples shape: [num_samples, z_dim]
    return epsilon_samples.detach(), z_samples.detach()


def estimate_fit_nll(
    reverse_model: BaseReverseConditionalModel,
    epsilon: torch.Tensor,
    z: torch.Tensor,
    *,
    batch_size: int,
    proposal_cache: dict[str, torch.Tensor] | None = None,
) -> float:
    """Estimate proposal fit quality as average ``-log q_psi(epsilon | z)``.

    The inputs are held-out or freshly sampled VI joint pairs. A lower value
    indicates the reverse proposal assigns higher conditional likelihood to
    epsilons that actually produced the observed ``z`` values.
    """
    losses: list[torch.Tensor] = []
    epsilon = epsilon.reshape(-1, epsilon.shape[-1])
    # shape: [N_eval, epsilon_dim]
    z = z.reshape(-1, z.shape[-1])
    # shape: [N_eval, z_dim]
    with torch.no_grad():
        for start in range(0, int(epsilon.shape[0]), batch_size):
            eps_batch = epsilon[start:start + batch_size]
            # shape: [batch, epsilon_dim]
            z_batch = z[start:start + batch_size]
            # shape: [batch, z_dim]
            log_prob = reverse_model_log_prob(
                reverse_model,
                eps_batch,
                z_batch,
                proposal_cache=proposal_cache,
            )  # shape: [batch]
            losses.append((-log_prob).detach().cpu())
    return float(torch.cat(losses).mean().item())


def build_reverse_proposal(
    vi_model: BaseVIModel,
    *,
    proposal_type: str,
    proposal_config_path: str | Path | None,
) -> tuple[BaseReverseConditionalModel, dict[str, Any], dict[str, Any]]:
    """Construct an unfitted reverse proposal and initial diagnostics.

    This is useful when callers need to own the fitting loop themselves. Most
    ELM workflows use ``fit_reverse_proposal``, which calls the same resolver
    and then performs either direct fitting or optimizer fitting.
    """
    canonical_type, proposal_alias, config_path, proposal_cfg, resolved_config = (
        resolve_reverse_model_config(
            vi_model,
            proposal_type,
            proposal_config_path,
        )
    )
    # Instantiate through the same registry used by the runner system.
    reverse_model = ReverseModel[canonical_type](config=proposal_cfg).to(proposal_cfg.device)
    diagnostics: dict[str, Any] = {
        "proposal_type": proposal_alias,
        "proposal_class": canonical_type,
        "proposal_config_path": config_path.as_posix(),
    }
    if isinstance(reverse_model, ConditionalMixtureOfGaussianReverse):
        diagnostics["num_components"] = int(reverse_model.num_components)
    return reverse_model, diagnostics, resolved_config


def collect_vi_joint_samples(
    vi_model: BaseVIModel,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect ``num_samples`` pairs from ``p(epsilon) q_phi(z | epsilon)``."""
    return sample_vi_joint(vi_model, num_samples)


def run_direct_fit(
    reverse_model: BaseReverseConditionalModel,
    *,
    epsilon_samples: torch.Tensor,
    z_samples: torch.Tensor,
) -> dict[str, Any]:
    """Fit proposals that have a closed-form or one-call estimator.

    Gaussian and MoG reverse models can fit their joint distribution from a
    fixed set of VI samples. They then expose ``q_psi(epsilon | z)`` through
    Gaussian conditioning formulas implemented in ``models.reverse_model``.
    """
    fit_nll, fit_steps = reverse_model.fit(
        epsilon_samples,
        z_samples,
        initialize=True,
    )
    return {
        "fit_nll": float(fit_nll),
        "fit_steps": int(fit_steps),
    }


def run_optimizer_fit(
    reverse_model: BaseReverseConditionalModel,
    vi_model: BaseVIModel,
    *,
    fit_batch_size: int,
    fit_epochs: int,
    fit_lr: float | None,
    proposal_cfg: DictConfig,
    log_every: int,
) -> dict[str, Any]:
    """Train a neural reverse proposal by maximum conditional likelihood.

    Each epoch draws a fresh batch ``(epsilon, z)`` from the current VI model and
    minimizes

        -E[log q_psi(epsilon | z)].

    This path is used for proposals such as ConditionalRealNVP. It deliberately
    samples online rather than reusing one finite training set, which makes the
    objective closer to the VI joint distribution.
    """
    resolved_lr = float(
        fit_lr
        if fit_lr is not None
        else proposal_cfg.get("lr", proposal_cfg.get("warmup", {}).get("lr", 1.0e-2))
    )
    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=resolved_lr)
    losses: list[float] = []
    best_loss = float("inf")
    optimizer_steps = 0
    reverse_model.train()

    for epoch in range(1, fit_epochs + 1):
        # Fresh VI joint batch: epsilon is the auxiliary noise, z is the sample
        # produced by q_phi(z | epsilon).
        epsilon_samples, z_samples = collect_vi_joint_samples(vi_model, fit_batch_size)
        # epsilon_samples shape: [fit_batch_size, epsilon_dim]
        # z_samples shape: [fit_batch_size, z_dim]
        optimizer.zero_grad()
        log_prob = reverse_model.log_prob(epsilon_samples, z_samples)
        # shape: [fit_batch_size]
        loss = -torch.mean(log_prob)  # scalar
        loss_value = float(loss.item())
        losses.append(loss_value)
        if torch.isfinite(loss):
            # Non-finite losses are skipped so one unstable batch does not
            # destroy the proposal state.
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            best_loss = min(best_loss, loss_value)
        else:
            logger.warning(
                "Skipping non-finite reverse proposal optimizer step for %s at epoch %s.",
                type(reverse_model).__name__,
                epoch,
            )
        if log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == fit_epochs):
            logger.info(
                "Reverse proposal fit [%s] epoch=%s/%s loss=%.6f best=%.6f",
                type(reverse_model).__name__,
                epoch,
                fit_epochs,
                loss_value,
                best_loss,
            )

    reverse_model.eval()
    return {
        "fit_steps": int(optimizer_steps),
        "fit_lr": float(resolved_lr),
        "fit_loss_initial": float(losses[0]) if losses else float("nan"),
        "fit_loss_final": float(losses[-1]) if losses else float("nan"),
        "fit_loss_best": float(best_loss) if losses else float("nan"),
    }


def finalize_reverse_proposal_fit(
    reverse_model: BaseReverseConditionalModel,
    vi_model: BaseVIModel,
    *,
    diagnostics: dict[str, Any],
    fit_batch_size: int,
    num_fit_samples: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    """Evaluate the fitted proposal and build any reusable estimator cache."""
    # Use fresh VI samples for an apples-to-apples proposal NLL diagnostic.
    eval_epsilon, eval_z = collect_vi_joint_samples(vi_model, num_fit_samples)

    # Only Gaussian proposals currently have a closed-form cache. For other
    # proposal families this returns None and reverse-IS calls the model methods.
    proposal_cache = gaussian_reverse_cache(reverse_model, dtype=module_dtype(vi_model))
    fit_nll = estimate_fit_nll(
        reverse_model,
        eval_epsilon,
        eval_z,
        batch_size=fit_batch_size,
        proposal_cache=proposal_cache,
    )
    diagnostics["fit_nll"] = float(fit_nll)
    return diagnostics, proposal_cache


def fit_reverse_proposal(
    vi_model: BaseVIModel,
    *,
    proposal_type: str = "gaussian",
    proposal_config_path: str | Path | None = None,
    num_fit_samples: int = 32768,
    fit_batch_size: int = 8192,
    fit_epochs: int = 1000,
    fit_lr: float | None = None,
    log_every: int = 100,
) -> ReverseProposalFit:
    """Fit ``q_psi(epsilon | z)`` for reverse-IS ELM estimation.

    The reverse proposal is a conditional density estimator trained from VI
    joint samples ``epsilon ~ p(epsilon)``, ``z ~ q_phi(z | epsilon)``. Once
    fitted, it helps estimate the marginal density at reference points through
    importance weights

        q_phi(z | epsilon) p(epsilon) / q_psi(epsilon | z).

    Args:
        vi_model: Trained variational model exposing ``sampling``,
            ``sample_epsilon``, ``log_q_epsilon``, and ``logp``.
        proposal_type: Alias or registered class name for the reverse proposal.
        proposal_config_path: Optional config override; otherwise the default
            file under ``configs/reverse_models`` is used.
        num_fit_samples: Number of VI joint pairs for direct-fit proposals and
            final fit diagnostics.
        fit_batch_size: Online batch size for optimizer-fit proposals and NLL
            evaluation batch size.
        fit_epochs: Number of optimizer epochs for neural proposals.
        fit_lr: Optional optimizer learning rate override.
        log_every: Optimizer progress logging interval; set to 0 to silence.

    Returns:
        ``ReverseProposalFit`` containing the fitted model, diagnostics, any
        Gaussian cache, and the resolved config used to construct the proposal.
    """
    if num_fit_samples < 2:
        raise ValueError("num_fit_samples must be at least 2.")
    if fit_batch_size < 1:
        raise ValueError("fit_batch_size must be at least 1.")
    if fit_epochs < 1:
        raise ValueError("fit_epochs must be at least 1.")

    canonical_type, proposal_alias, config_path, proposal_cfg, resolved_config = (
        resolve_reverse_model_config(
            vi_model,
            proposal_type,
            proposal_config_path,
        )
    )
    reverse_model = ReverseModel[canonical_type](config=proposal_cfg).to(proposal_cfg.device)
    use_optimizer = bool(proposal_cfg.get("use_optimizer", True))
    fit_mode = "optimizer" if use_optimizer else "direct_fit"

    # Diagnostics are persisted by the evaluator script and make budget sweeps
    # reproducible: proposal family, fitting mode, budgets, and config path.
    diagnostics: dict[str, Any] = {
        "proposal_type": proposal_alias,
        "proposal_class": canonical_type,
        "proposal_config_path": config_path.as_posix(),
        "fit_mode": fit_mode,
        "fit_samples": int(num_fit_samples),
        "fit_batch_size": int(fit_batch_size if use_optimizer else num_fit_samples),
        "fit_epochs": int(fit_epochs if use_optimizer else 1),
    }
    if isinstance(reverse_model, ConditionalMixtureOfGaussianReverse):
        diagnostics["num_components"] = int(reverse_model.num_components)

    was_vi_training = vi_model.training
    vi_model.eval()
    start = time.perf_counter()

    try:
        if use_optimizer:
            # Neural proposals learn q_psi(epsilon | z) with gradient descent.
            diagnostics.update(
                run_optimizer_fit(
                    reverse_model,
                    vi_model,
                    fit_batch_size=fit_batch_size,
                    fit_epochs=fit_epochs,
                    fit_lr=fit_lr,
                    proposal_cfg=proposal_cfg,
                    log_every=log_every,
                )
            )
        else:
            # Direct-fit proposals estimate their parameters from one VI joint
            # sample set, e.g. empirical joint Gaussian moments or GMM EM.
            epsilon_samples, z_samples = collect_vi_joint_samples(vi_model, num_fit_samples)
            # epsilon_samples shape: [num_fit_samples, epsilon_dim]
            # z_samples shape: [num_fit_samples, z_dim]
            diagnostics.update(
                run_direct_fit(
                    reverse_model,
                    epsilon_samples=epsilon_samples,
                    z_samples=z_samples,
                )
            )
            diagnostics["fit_lr"] = None

        reverse_model.eval()
        # Add a held-out-style NLL diagnostic and build the Gaussian fast-path
        # cache used by estimate_log_q_reverse_is.
        diagnostics, proposal_cache = finalize_reverse_proposal_fit(
            reverse_model,
            vi_model,
            diagnostics=diagnostics,
            fit_batch_size=fit_batch_size,
            num_fit_samples=num_fit_samples,
        )
    finally:
        if was_vi_training:
            vi_model.train()

    diagnostics["fit_runtime_sec"] = float(time.perf_counter() - start)

    return ReverseProposalFit(
        reverse_model=reverse_model,
        proposal_type=proposal_alias,
        fit_mode=fit_mode,
        diagnostics=diagnostics,
        cache=proposal_cache,
        resolved_config=resolved_config,
    )


def save_reverse_proposal_fit(
    output_dir: str | Path,
    proposal_fit: ReverseProposalFit,
    *,
    save_state: bool = False,
) -> tuple[Path, Path | None]:
    """Persist fitted-proposal metadata, and optionally the model state.

    ``proposal_fit.json`` is small and records the fitting setup for later
    interpretation of ELM sweeps. ``proposal_state.pt`` is optional because many
    workflows only need the scalar diagnostics and summary tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_json_path = output_dir / "proposal_fit.json"
    fit_json_path.write_text(
        json.dumps(
            {
                **proposal_fit.diagnostics,
                "resolved_config": proposal_fit.resolved_config,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    state_path: Path | None = None
    if save_state:
        state_path = output_dir / "proposal_state.pt"
        torch.save(
            {
                "proposal_type": proposal_fit.proposal_type,
                "fit_mode": proposal_fit.fit_mode,
                "diagnostics": proposal_fit.diagnostics,
                "resolved_config": proposal_fit.resolved_config,
                "state_dict": proposal_fit.reverse_model.state_dict(),
            },
            state_path,
        )
    return fit_json_path, state_path
