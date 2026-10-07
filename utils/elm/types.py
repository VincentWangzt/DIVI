from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.reverse_model import BaseReverseConditionalModel


@dataclass
class ReverseProposalFit:
    """Fitted reverse proposal and metadata for reverse-IS ELM.

    ``reverse_model`` estimates ``q_psi(epsilon | z)``. ``cache`` is currently
    used for the Gaussian fast path, and ``diagnostics``/``resolved_config`` are
    written by evaluator scripts to make reproduction easier.
    """

    reverse_model: BaseReverseConditionalModel
    proposal_type: str
    fit_mode: str
    diagnostics: dict[str, Any]
    cache: dict[str, torch.Tensor] | None = None
    resolved_config: dict[str, Any] | None = None


@dataclass
class LogQEstimate:
    """Per-reference estimates of ``log q_phi(z_i)``.

    ``log_q_values`` has shape ``[N_ref]``. ``ess_values`` is present for
    reverse-IS estimates and contains one effective sample size per reference
    point.
    """

    log_q_values: torch.Tensor
    stderr: float
    diagnostics: dict[str, Any]
    ess_values: torch.Tensor | None = None


@dataclass
class ELMEstimate:
    """Scalar expected log marginal summary.

    ``value`` is the finite-reference mean of ``log_q_values``, estimating
    ``E_{z ~ r}[log q_phi(z)]``. Diagnostics from the underlying estimator are
    forwarded with the scalar value and standard error added.
    """

    value: float
    stderr: float
    log_q_values: torch.Tensor
    diagnostics: dict[str, Any]
    ess_values: torch.Tensor | None = None


@dataclass
class KDEELMEstimate:
    """Paper-style coordinate-marginal KDE expected log marginal summary.

    ``value`` is the finite-reference mean of ``per_reference_log_values``.
    Each per-reference value is the sum of one-dimensional KDE log densities
    across coordinates.
    """

    value: float
    stderr: float
    per_reference_log_values: torch.Tensor
    diagnostics: dict[str, Any]
    bandwidths: torch.Tensor
