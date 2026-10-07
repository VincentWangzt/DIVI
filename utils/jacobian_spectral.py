"""Spectral-norm utilities for parameter Jacobians of the VI model.

Computes  E_ε[‖∇_φ μ_φ(ε)‖₂⁴]  and  E_ε[‖∇_φ σ_φ(ε)‖₂⁴]  where the norm
is the matrix 2-norm (largest singular value) of the d_z × d_φ Jacobian.

This supports empirical validation of the Bounded Reparameterization Assumption
used in the DIVI convergence theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class AssumptionBoundResult:
    """Result of evaluating the bounded-gradient assumption."""

    n_samples: int
    d_z: int
    d_phi: int

    # E_ε[‖∇_φ μ_φ(ε)‖₂²]  and  E_ε[‖∇_φ σ_φ(ε)‖₂²]
    mean_sq_spectral_mu: float = 0.0
    mean_sq_spectral_std: float = 0.0

    # max of the two = empirical VI derivative fourth moment
    M_eps: float = 0.0

    # Standard errors of the mean (Monte Carlo)
    std_err_mu: float = 0.0
    std_err_std: float = 0.0

    # Per-sample maximums (worst-case bounds)
    max_spectral_mu: float = 0.0
    max_spectral_std: float = 0.0


def _get_component_output(
    model: nn.Module,
    epsilon: Tensor,
    component: Literal["mu", "std"],
) -> Tensor:
    """Get μ_φ(ε) or σ_φ(ε) as a differentiable function of model parameters.

    Args:
        model: A VI model with ``getmu()`` and ``getstd()`` methods.
        epsilon: Input noise of shape ``(B, d_eps)``.
        component: ``"mu"`` for the mean, ``"std"`` for the standard deviation
            (after softplus/exp transform).

    Returns:
        Output tensor of shape ``(B, d_z)``.
    """
    if component == "mu":
        return model.getmu(epsilon)
    elif component == "std":
        return model.getstd(epsilon)
    else:
        raise ValueError(f"component must be 'mu' or 'std', got {component!r}")


def compute_param_jacobian(
    model: nn.Module,
    epsilon_single: Tensor,
    component: Literal["mu", "std"],
) -> Tensor:
    """Compute the full parameter Jacobian for a single ε sample.

    Computes  J = ∂f_φ(ε)/∂φ  ∈ R^{d_z × d_φ}  where f is either μ or σ.

    Uses d_z backward passes (VJP approach): for each output dimension d,
    backpropagate a unit vector to get row d of the Jacobian.

    Args:
        model: VI model with ``getmu()``/``getstd()`` methods.
        epsilon_single: Single ε vector of shape ``(d_eps,)``.
        component: ``"mu"`` or ``"std"``.

    Returns:
        Jacobian matrix of shape ``(d_z, d_phi)``.
    """
    eps = epsilon_single.unsqueeze(0)  # (1, d_eps)
    output = _get_component_output(model, eps, component)  # (1, d_z)
    output = output.squeeze(0)  # (d_z,)
    d_z = output.shape[0]

    rows = []
    for d in range(d_z):
        model.zero_grad()
        output[d].backward(retain_graph=(d < d_z - 1))
        row = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), device=p.device)
            for p in model.parameters()
        ])
        rows.append(row)

    return torch.stack(rows)  # (d_z, d_phi)


def compute_jacobian_spectral_norms(
    model: nn.Module,
    epsilon_batch: Tensor,
    component: Literal["mu", "std"],
) -> Tensor:
    """Compute spectral norms of the parameter Jacobian for a batch of ε.

    Args:
        model: VI model with ``getmu()``/``getstd()`` methods.
        epsilon_batch: Batch of ε vectors, shape ``(B, d_eps)``.
        component: ``"mu"`` or ``"std"``.

    Returns:
        Spectral norms (largest singular values), shape ``(B,)``.
    """
    norms = []
    for i in range(epsilon_batch.shape[0]):
        J = compute_param_jacobian(model, epsilon_batch[i], component)
        sv = torch.linalg.svdvals(J)
        norms.append(sv[0])

    return torch.stack(norms)


def evaluate_assumption_bound(
    model: nn.Module,
    epsilon_batch: Tensor,
) -> AssumptionBoundResult:
    """Evaluate the Bounded Reparameterization Assumption for a batch of ε samples.

    Computes  E_ε[‖∇_φ μ_φ(ε)‖₂⁴]  and  E_ε[‖∇_φ σ_φ(ε)‖₂⁴]  via
    Monte Carlo over the supplied ε batch.

    Args:
        model: VI model with ``getmu()``/``getstd()`` methods and
            standard ``parameters()`` interface.
        epsilon_batch: ε samples of shape ``(B, d_eps)``.

    Returns:
        :class:`AssumptionBoundResult` with all computed quantities.
    """
    d_z = model.z_dim
    d_phi = sum(p.numel() for p in model.parameters())
    n = epsilon_batch.shape[0]

    norms_mu = compute_jacobian_spectral_norms(model, epsilon_batch, "mu")
    norms_std = compute_jacobian_spectral_norms(model, epsilon_batch, "std")

    sq_mu = norms_mu ** 4
    sq_std = norms_std ** 4

    mean_sq_mu = sq_mu.mean().item()
    mean_sq_std = sq_std.mean().item()

    return AssumptionBoundResult(
        n_samples=n,
        d_z=d_z,
        d_phi=d_phi,
        mean_sq_spectral_mu=mean_sq_mu,
        mean_sq_spectral_std=mean_sq_std,
        M_eps=max(mean_sq_mu, mean_sq_std),
        std_err_mu=(sq_mu.std().item() / math.sqrt(n)) if n > 1 else 0.0,
        std_err_std=(sq_std.std().item() / math.sqrt(n)) if n > 1 else 0.0,
        max_spectral_mu=norms_mu.max().item(),
        max_spectral_std=norms_std.max().item(),
    )


# ---------------------------------------------------------------------------
# Sanity check: verify against analytical Jacobian of a linear model
# ---------------------------------------------------------------------------

def sanity_check(device: str = "cpu") -> bool:
    """Verify Jacobian computation against an analytical solution.

    Builds ``y = W @ x + b`` where x = ε (d_eps=5, d_z=2) and checks that
    the computed spectral norm matches ``torch.linalg.svdvals`` of the
    analytically constructed Jacobian.

    The analytical Jacobian of y w.r.t. vec(W) and b is::

        J = [ε^T ⊗ I_{d_z}  |  I_{d_z}]   shape (d_z, d_z*d_eps + d_z)

    Returns:
        ``True`` if the check passes.

    Raises:
        AssertionError: If the computed spectral norm doesn't match.
    """
    d_eps, d_z = 5, 2
    dev = torch.device(device)

    linear = nn.Linear(d_eps, d_z).to(dev)

    # Wrap in a minimal object that looks like our VI model
    class _LinearWrapper(nn.Module):
        def __init__(self, layer: nn.Linear):
            super().__init__()
            self.layer = layer
            self.z_dim = layer.out_features

        def getmu(self, epsilon: Tensor) -> Tensor:
            return self.layer(epsilon)

        def getstd(self, epsilon: Tensor) -> Tensor:
            return self.layer(epsilon)  # same for this test

    wrapper = _LinearWrapper(linear)

    eps = torch.randn(d_eps, device=dev)

    # Compute via our function
    computed_J = compute_param_jacobian(wrapper, eps, "mu")
    computed_sv = torch.linalg.svdvals(computed_J)[0]

    # Analytical Jacobian: J_W rows are kron(ε^T, e_d), J_b rows are e_d
    W = linear.weight.data  # (d_z, d_eps)
    # Jacobian w.r.t. weight: for output d, gradient w.r.t. W[d,:] = ε
    # So J_W row d = [0...0, ε^T, 0...0] with ε^T in position d*(d_eps):(d+1)*d_eps
    J_W = torch.zeros(d_z, d_z * d_eps, device=dev)
    for d in range(d_z):
        J_W[d, d * d_eps:(d + 1) * d_eps] = eps

    # Jacobian w.r.t. bias: J_b = I_{d_z}
    J_b = torch.eye(d_z, device=dev)

    # Full analytical Jacobian
    J_analytical = torch.cat([J_W, J_b], dim=1)
    analytical_sv = torch.linalg.svdvals(J_analytical)[0]

    assert torch.allclose(computed_sv, analytical_sv, atol=1e-5), (
        f"Sanity check FAILED: computed {computed_sv.item():.6f} vs "
        f"analytical {analytical_sv.item():.6f}"
    )
    return True
