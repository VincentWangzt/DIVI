"""Target distributions for SIVI experiments.

Provides log-density (``logp``) and score-function (``score``) implementations
for toy 2-D distributions, a Bayesian neural network (BNN) posterior, and
a logistic-regression waveform posterior.

Classes are registered in the :data:`target_distribution` dict and
instantiated by name in ``runner/base_runner.py``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "TargetModel",
    "Toy_2D",
    "Banana_shape",
    "X_shaped",
    "Multimodal",
    "EightGaussians",
    "StudentTFullDim",
    "LRwaveform",
    "Bnn",
    "Langevin_post",
    "target_distribution",
    "DEFAULT_BBOX",
]

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TargetModel(Protocol):
    """Structural interface for target distributions used by the runner.

    All toy 2-D models conform to this protocol.  Data-dependent models
    (``Bnn``, ``LRwaveform``) have extended ``logp``/``score`` signatures
    and are used through a different code path.
    """

    z_dim: int
    device: torch.device

    def logp(self, X: torch.Tensor) -> torch.Tensor: ...
    def score(self, X: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BBOX: dict[str, list[float]] = {
    "multimodal": [-5, 5, -5, 5],
    "8_gaussians": [-6, 6, -6, 6],
    "banana": [-3.5, 3.5, -6, 1],
    "x_shaped": [-5, 5, -5, 5],
    "student_uc": [-5, 5, -5, 5],
}

# Banana_shape covariance determinant (det of [[1, -0.9], [-0.9, 1]])
_BANANA_COV_SCALE: float = 0.19

# X_shaped covariance determinant (det of [[2, ±1.8], [±1.8, 2]])
_X_SHAPED_COV_SCALE: float = 0.76

# Multimodal mixture component separation along x-axis
_MULTIMODAL_SPREAD: float = 2.0

# Eight-Gaussians FFJORD-style ring parameters
_EIGHT_GAUSSIANS_RADIUS: float = 4.0
_EIGHT_GAUSSIANS_SIGMA: float = 0.5
_EIGHT_GAUSSIANS_NUM_COMPONENTS: int = 8

# StudentT degrees of freedom and scale-matrix seed
_STUDENT_T_DF: float = 2.0
_STUDENT_T_SEED: int = 50

# Visualization grid resolutions
_CONTOUR_GRID_RES: complex = 100j
_QUIVER_GRID_RES: complex = 30j
_CONTOUR_N_SAMPLES: int = 10_000

# Langevin posterior synthetic-data seed
_LANGEVIN_SEED: int = 2022

# BNN defaults
_BNN_N_HIDDEN: int = 50
_BNN_MAX_PARAM: float = 50.0
_BNN_ALPHA: float = 0.01


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Toy_2D:
    """Abstract base class for 2-D toy target distributions.

    Subclasses must implement :meth:`logp` and :meth:`score`.
    A default :meth:`contour_plot` is provided for 2-D distributions.

    Parameters
    ----------
    device : torch.device
        Computation device.
    name : str
        Human-readable name used in plot titles and logging.
    """

    z_dim: int = 2

    def __init__(self, device: torch.device, name: str = "") -> None:
        assert name != "", "Please provide a name for the 2D toy distribution."
        self.device = device
        self.name = name

    def logp(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the unnormalized log-density.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, z_dim)``.

        Returns
        -------
        torch.Tensor
            Log-density values, shape ``(batch,)`` or ``(batch, 1)``.
        """
        raise NotImplementedError

    def score(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute the score function :math:`\nabla_X \log p(X)`.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, z_dim)``.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, z_dim)``.
        """
        raise NotImplementedError

    def contour_plot(
        self,
        bbox: list[float],
        fnet: torch.nn.Module | None = None,
        samples: NDArray | None = None,
        save_to_path: str | None = None,
        quiver: bool = True,
        t: int | None = None,
    ) -> None:
        """Render a contour plot of the log-density with optional score quiver.

        Parameters
        ----------
        bbox : list[float]
            Bounding box ``[x_min, x_max, y_min, y_max]``.
        fnet : torch.nn.Module or None
            Score network whose outputs are shown as quiver arrows.
        samples : numpy.ndarray or None
            ``(N, 2)`` array of samples to overlay.  If *None*, samples are
            drawn from ``self.sample``.
        save_to_path : str or None
            File path to save the figure.
        quiver : bool
            Whether to draw the score quiver field.
        t : int or None
            Optional time step shown in the title.
        """
        plt.cla()
        fig, ax = plt.subplots(figsize=(5, 5))
        xx, yy = np.mgrid[
            bbox[0] : bbox[1] : _CONTOUR_GRID_RES,
            bbox[2] : bbox[3] : _CONTOUR_GRID_RES,
        ]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(
            -np.reshape(
                self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(),
                xx.shape,
            )
        )
        if samples is None:
            samples = self.sample(_CONTOUR_N_SAMPLES).cpu().numpy()

        cxx, cyy = np.mgrid[
            bbox[0] : bbox[1] : _QUIVER_GRID_RES,
            bbox[2] : bbox[3] : _QUIVER_GRID_RES,
        ]

        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
        ax.contourf(xx, yy, f, cmap="Blues", alpha=0.8, levels=11)
        ax.plot(samples[:, 0], samples[:, 1], ".", markersize=2, color="#ff7f0e")
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            scores = np.reshape(
                fnet(torch.Tensor(cpositions.T).to(self.device))
                .detach()
                .cpu()
                .numpy(),
                cpositions.T.shape,
            )
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if t:
            ax.set_title("t = {}".format(t), fontsize=30, y=1.04)
        else:
            ax.set_title(f"{self.name}", fontsize=20, y=1.04)
        if save_to_path is not None:
            plt.savefig(save_to_path, bbox_inches="tight")
        plt.close()

    def trace_plot(
        self,
        u: torch.Tensor,
        figpath: str | None = None,
        figname: str | None = None,
        figtitle: str = "",
    ) -> None:
        """Render a trace plot (implemented only by ``Langevin_post``)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Toy 2-D distributions
# ---------------------------------------------------------------------------


class Banana_shape(Toy_2D):
    """Correlated 2-D Gaussian mapped through a banana-shaped nonlinearity.

    The distribution applies the transform ``Y = (X_0, X_0^2 + X_1 + 1)``
    and evaluates a Gaussian with covariance determinant
    :data:`_BANANA_COV_SCALE`.

    Parameters
    ----------
    device : torch.device
        Computation device.
    """

    name: str = "banana_shape"

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device, name="Banana_shape")
        self._sigmasqinv: torch.Tensor = torch.tensor(
            [[1.0, -0.9], [-0.9, 1.0]], device=self.device
        ) / _BANANA_COV_SCALE
        self._cov: torch.Tensor = torch.tensor(
            [[1.0, 0.9], [0.9, 1.0]], device=self.device
        )

    def logp(self, X: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized log-density of the banana distribution.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Log-density values, shape ``(batch, 1)``.
        """
        Y = torch.stack((X[:, 0], X[:, 0] ** 2 + X[:, 1] + 1), 1)
        return (
            -0.5 * 2 * np.log(2 * np.pi)
            - 0.5 * np.log(_BANANA_COV_SCALE)
            - 0.5
            * torch.matmul(
                torch.matmul(Y[:, None, :], self._sigmasqinv), Y[:, :, None]
            ).squeeze(-1)
        )

    def score(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_X \log p(X)` for the banana distribution.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, 2)``.
        """
        Y = torch.matmul(
            torch.stack((X[:, 0], X[:, 0] ** 2 + X[:, 1] + 1), 1),
            torch.tensor([[1.0, -0.9], [-0.9, 1.0]], device=self.device),
        )
        return (
            -torch.stack((Y[:, 0] + 2 * X[:, 0] * Y[:, 1], Y[:, 1]), 1)
            / _BANANA_COV_SCALE
        )

    def sample(self, N: int) -> torch.Tensor:
        """Draw exact samples by inverting the banana transform."""
        Y = torch.distributions.MultivariateNormal(
            loc=torch.zeros(2, device=self.device),
            covariance_matrix=self._cov,
        ).sample((N,))
        return torch.stack((Y[:, 0], Y[:, 1] - Y[:, 0] ** 2 - 1.0), dim=1)


class X_shaped(Toy_2D):
    """Two-component Gaussian mixture forming an X-shaped density in 2-D.

    Each component has a precision matrix with off-diagonal entries of
    opposite sign, scaled by :data:`_X_SHAPED_COV_SCALE`.

    Parameters
    ----------
    device : torch.device
        Computation device.
    """

    name: str = "x_shaped"

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device, name="X_shaped")
        self._sigmasqinv_0: torch.Tensor = torch.tensor(
            [[2.0, -1.8], [-1.8, 2.0]], device=self.device
        ) / _X_SHAPED_COV_SCALE
        self._sigmasqinv_1: torch.Tensor = torch.tensor(
            [[2.0, 1.8], [1.8, 2.0]], device=self.device
        ) / _X_SHAPED_COV_SCALE
        self._cov_0: torch.Tensor = torch.tensor(
            [[2.0, 1.8], [1.8, 2.0]], device=self.device
        )
        self._cov_1: torch.Tensor = torch.tensor(
            [[2.0, -1.8], [-1.8, 2.0]], device=self.device
        )

    def logp(self, X: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized log-density of the X-shaped mixture.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Log-density values, shape ``(batch, 1)``.
        """
        return -0.5 * 2 * np.log(2 * np.pi) - 0.5 * np.log(
            _X_SHAPED_COV_SCALE * 4
        ) + torch.logsumexp(
            torch.stack(
                (
                    -1
                    / 2
                    * torch.matmul(
                        torch.matmul(X[:, None, :], self._sigmasqinv_0),
                        X[:, :, None],
                    ).squeeze(-1),
                    -1
                    / 2
                    * torch.matmul(
                        torch.matmul(X[:, None, :], self._sigmasqinv_1),
                        X[:, :, None],
                    ).squeeze(-1),
                ),
                1,
            ),
            dim=1,
        )

    def score(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_X \log p(X)` for the X-shaped mixture.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, 2)``.
        """
        Y = F.softmax(
            torch.stack(
                (
                    -1
                    / 2
                    * torch.matmul(
                        torch.matmul(X[:, None, :], self._sigmasqinv_0),
                        X[:, :, None],
                    ).squeeze(-1),
                    -1
                    / 2
                    * torch.matmul(
                        torch.matmul(X[:, None, :], self._sigmasqinv_1),
                        X[:, :, None],
                    ).squeeze(-1),
                ),
                1,
            ),
            dim=1,
        )

        return -Y[:, 0] * torch.matmul(
            self._sigmasqinv_0, X[:, :, None]
        ).squeeze(-1) - Y[:, 1] * torch.matmul(
            self._sigmasqinv_1, X[:, :, None]
        ).squeeze(-1)

    def sample(self, N: int) -> torch.Tensor:
        """Draw exact samples from the two-component Gaussian mixture."""
        component_mask = torch.rand(N, device=self.device) < 0.5
        samples_0 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(2, device=self.device),
            covariance_matrix=self._cov_0,
        ).sample((N,))
        samples_1 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(2, device=self.device),
            covariance_matrix=self._cov_1,
        ).sample((N,))
        return torch.where(component_mask.unsqueeze(-1), samples_0, samples_1)


class Multimodal(Toy_2D):
    """Symmetric two-mode Gaussian mixture in 2-D.

    Two isotropic Gaussians centred at ``(±spread, 0)`` with unit variance.

    Parameters
    ----------
    device : torch.device
        Computation device.
    """

    name: str = "multimodal"

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device, name="Multimodal")
        self._means: torch.Tensor = torch.tensor(
            [[_MULTIMODAL_SPREAD, 0.0], [-_MULTIMODAL_SPREAD, 0.0]],
            device=self.device,
        )

    def logp(self, X: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized log-density of the multimodal mixture.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Log-density values, shape ``(batch,)``.
        """
        return (
            -0.5 * 2 * np.log(2 * np.pi)
            - np.log(_MULTIMODAL_SPREAD)
            + torch.logsumexp(
                -torch.sum(
                    (X.unsqueeze(1) - self._means.unsqueeze(0)) ** 2, dim=-1
                )
                / 2.0
                / 1**2,
                dim=1,
            )
        )

    def score(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_X \log p(X)` for the multimodal mixture.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, 2)``.
        """
        Y = F.softmax(
            torch.stack(
                (
                    -1 / 2 * ((X[:, 0] + _MULTIMODAL_SPREAD) ** 2 + X[:, 1] ** 2),
                    -1 / 2 * ((X[:, 0] - _MULTIMODAL_SPREAD) ** 2 + X[:, 1] ** 2),
                ),
                1,
            ),
            dim=1,
        )
        return -torch.stack(
            (
                Y[:, 0] * (X[:, 0] + _MULTIMODAL_SPREAD)
                + Y[:, 1] * (X[:, 0] - _MULTIMODAL_SPREAD),
                X[:, 1],
            ),
            1,
        )

    def sample(self, N: int) -> torch.Tensor:
        """Draw exact samples from the symmetric Gaussian mixture."""
        component = torch.randint(0, 2, (N,), device=self.device)
        noise = torch.randn(N, 2, device=self.device)
        return self._means[component] + noise


class EightGaussians(Toy_2D):
    """Eight-component isotropic Gaussian mixture arranged on a ring.

    This follows the looser FFJORD-style toy target: eight equally weighted
    modes on a circle of radius :data:`_EIGHT_GAUSSIANS_RADIUS`, each with
    covariance ``sigma^2 I`` where ``sigma =``
    :data:`_EIGHT_GAUSSIANS_SIGMA`.

    Parameters
    ----------
    device : torch.device
        Computation device.
    """

    name: str = "8_gaussians"

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device, name="EightGaussians")
        sqrt2 = float(np.sqrt(2.0))
        centers = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [1.0 / sqrt2, 1.0 / sqrt2],
                [1.0 / sqrt2, -1.0 / sqrt2],
                [-1.0 / sqrt2, 1.0 / sqrt2],
                [-1.0 / sqrt2, -1.0 / sqrt2],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.radius: float = _EIGHT_GAUSSIANS_RADIUS
        self.sigma: float = _EIGHT_GAUSSIANS_SIGMA
        self._var: float = self.sigma**2
        self._centers: torch.Tensor = centers * self.radius
        self._log_weight: float = -np.log(_EIGHT_GAUSSIANS_NUM_COMPONENTS)
        self._log_norm: float = -np.log(2 * np.pi * self._var)

    def _component_log_probs(self, X: torch.Tensor) -> torch.Tensor:
        """Compute per-component log-densities up to equal mixture weights."""
        sq_dist = torch.sum(
            (X.unsqueeze(1) - self._centers.unsqueeze(0)) ** 2,
            dim=-1,
        )
        return self._log_weight + self._log_norm - sq_dist / (2.0 * self._var)

    def logp(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the normalized log-density of the 8-Gaussians mixture.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Log-density values, shape ``(batch,)``.
        """
        return torch.logsumexp(self._component_log_probs(X), dim=1)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_X \log p(X)` for the mixture.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, 2)``.
        """
        responsibilities = F.softmax(self._component_log_probs(X), dim=1)
        component_scores = (self._centers.unsqueeze(0) - X.unsqueeze(1)) / self._var
        return torch.sum(
            responsibilities.unsqueeze(-1) * component_scores,
            dim=1,
        )

    def sample(self, N: int) -> torch.Tensor:
        """Draw exact samples from the mixture distribution.

        Parameters
        ----------
        N : int
            Number of samples.

        Returns
        -------
        torch.Tensor
            Samples, shape ``(N, 2)``.
        """
        indices = torch.randint(
            _EIGHT_GAUSSIANS_NUM_COMPONENTS,
            size=(N,),
            device=self.device,
        )
        noise = torch.randn((N, self.z_dim), device=self.device) * self.sigma
        return self._centers[indices] + noise


class StudentTFullDim(Toy_2D):
    """General *d*-dimensional Student-*t* distribution with random scale matrix.

    The scale matrix ``A`` is generated from a random seed so that results
    are reproducible.  The score is obtained via automatic differentiation.

    Parameters
    ----------
    device : torch.device
        Computation device.
    """

    name: str = "StudentTFullDim"

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device, name="StudentTFullDim")
        self.df: float = _STUDENT_T_DF
        self.seed: int = _STUDENT_T_SEED
        self.dim: int = 2
        rng = np.random.RandomState(self.seed)
        A_sqrt = rng.uniform(-1.0, 1.0, size=(self.dim, self.dim))
        A = A_sqrt @ A_sqrt.T
        A = np.linalg.inv(A)
        A /= np.linalg.det(A)  # unit determinant
        self.A: torch.Tensor = torch.from_numpy(A).float().to(device)
        self.A_inv: torch.Tensor = (
            torch.from_numpy(np.linalg.inv(A)).float().to(device)
        )
        from torch.distributions.studentT import StudentT

        self.dist = StudentT(df=self.df)

    def logp(self, X: torch.Tensor) -> torch.Tensor:
        """Compute log-density of the Student-*t* distribution.

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, dim)``.

        Returns
        -------
        torch.Tensor
            Log-density values, shape ``(batch,)``.
        """
        assert self.A.shape[0] == X.shape[-1]
        B = X.shape[0]
        Z = torch.bmm(
            self.A_inv.unsqueeze(0).expand(B, -1, -1), X.unsqueeze(-1)
        ).squeeze(-1)  # (B, D)
        return self.dist.log_prob(Z).sum(-1)  # (B,)

    def sample_gt_impl(self, B: int) -> torch.Tensor:
        """Draw exact samples from the Student-*t* distribution.

        Parameters
        ----------
        B : int
            Number of samples.

        Returns
        -------
        torch.Tensor
            Samples, shape ``(B, dim)``.
        """
        Z = self.dist.sample([B, self.dim]).to(self.device)
        X = torch.bmm(
            self.A.unsqueeze(0).expand(B, -1, -1), Z.unsqueeze(-1)
        ).squeeze(-1)  # (B, D)
        return X

    def sample(self, N: int) -> torch.Tensor:
        """Draw exact samples (alias for :meth:`sample_gt_impl`)."""
        return self.sample_gt_impl(N)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_X \log p(X)` analytically.

        For independent univariate Student-*t* marginals in the transformed
        space ``Z = A_inv @ X``, the per-component score in Z-space is::

            dlogp/dZ_i = -(df+1)/df * Z_i / (1 + Z_i^2/df)

        Chain rule back to X-space gives::

            score_X = A_inv^T @ dlogp/dZ

        Parameters
        ----------
        X : torch.Tensor
            Input points, shape ``(batch, dim)``.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, dim)``.
        """
        B = X.shape[0]
        # Z = A_inv @ X, shape (B, D)
        Z = torch.bmm(
            self.A_inv.unsqueeze(0).expand(B, -1, -1), X.unsqueeze(-1)
        ).squeeze(-1)
        # Per-component score in Z-space
        dlogp_dZ = -(self.df + 1.0) / self.df * Z / (1.0 + Z ** 2 / self.df)
        # Chain rule: score_X = A_inv^T @ dlogp_dZ  (A_inv is symmetric for this construction)
        score_X = torch.bmm(
            self.A_inv.T.unsqueeze(0).expand(B, -1, -1), dlogp_dZ.unsqueeze(-1)
        ).squeeze(-1)
        return score_X


# ---------------------------------------------------------------------------
# Data-dependent targets
# ---------------------------------------------------------------------------


class LRwaveform:
    """Logistic-regression posterior on the UCI *waveform* dataset.

    Unlike the toy 2-D targets, ``logp`` and ``score`` require a data batch
    (``batchdataset``, ``batchlabel``) to evaluate the likelihood term.

    Parameters
    ----------
    device : torch.device
        Computation device.
    alpha : float
        L2 regularisation strength (prior precision).
    """

    name: str = "LRwaveform"

    def __init__(self, device: torch.device, alpha: float = _BNN_ALPHA) -> None:
        self.device = device
        self.alpha = alpha

    def logp(
        self,
        Z: torch.Tensor,
        batchdataset: torch.Tensor,
        batchlabel: torch.Tensor,
        scale_sto: float = 1,
    ) -> torch.Tensor:
        r"""Compute the (stochastic) log-posterior.

        Parameters
        ----------
        Z : torch.Tensor
            Weight vectors, shape ``(T, x_dim + 1)``.
        batchdataset : torch.Tensor
            Mini-batch features, shape ``(n, x_dim + 1)``.
        batchlabel : torch.Tensor
            Mini-batch labels, shape ``(n, 1)``.
        scale_sto : float
            Stochastic scaling factor ``num_datasets / batchsize``.

        Returns
        -------
        torch.Tensor
            :math:`\mathbb{E}_{Y|X}\log p(Y|X,z)`, scalar.
        """
        B = Z.shape[0]
        W = Z
        inner_prod = torch.mm(batchdataset, W.t())
        logpy_xz = batchlabel.reshape(-1, 1) * inner_prod + F.logsigmoid(
            -inner_prod
        )
        return torch.logsumexp(logpy_xz, dim=1).mean(0) - np.log(B)

    def score(
        self,
        Z: torch.Tensor,
        batchdataset: torch.Tensor,
        batchlabel: torch.Tensor,
        scale_sto: float = 1,
    ) -> torch.Tensor:
        r"""Compute the score of the logistic-regression posterior.

        Parameters
        ----------
        Z : torch.Tensor
            Weight vectors, shape ``(batch, x_dim + 1)``.
        batchdataset : torch.Tensor
            Mini-batch features, shape ``(n, x_dim + 1)``.
        batchlabel : torch.Tensor
            Mini-batch labels, shape ``(n, 1)``.
        scale_sto : float
            Stochastic scaling factor ``num_datasets / batchsize``.

        Returns
        -------
        torch.Tensor
            :math:`-Z\alpha + \nabla_Z\log p(Y|X,Z)`, shape ``(batch, x_dim + 1)``.
        """
        W = Z
        YX = torch.mm(batchlabel.reshape(-1, 1).t(), batchdataset)
        inner_prod = torch.mm(batchdataset, W.t())
        score_W = -W * self.alpha + (
            YX
            - torch.sum(
                torch.sigmoid(inner_prod).unsqueeze(2) * batchdataset.unsqueeze(1),
                dim=0,
            )
        ) * scale_sto
        return score_W


class Bnn:
    """Bayesian neural network posterior (single hidden layer, ReLU).

    Used for regression tasks (e.g. Boston housing).  The flat parameter
    vector ``Z`` is unpacked into weights and biases via
    :meth:`_unpack_weights`.

    Parameters
    ----------
    device : torch.device
        Computation device.
    d : int
        Input dimensionality.
    n_hidden : int
        Number of hidden units.
    loglambda : float
        Log weight-decay precision.
    loggamma : float
        Log observation precision.
    """

    name: str = "Bnn"

    def __init__(
        self,
        device: torch.device,
        d: int,
        n_hidden: int = _BNN_N_HIDDEN,
        loglambda: float = 0,
        loggamma: float = 0,
    ) -> None:
        self.device = device
        self.n_hidden = n_hidden
        self.d = d
        self.dim_vars: int = (self.d + 1) * self.n_hidden + (self.n_hidden + 1) + 2
        self.dim_wb: int = self.dim_vars - 2
        self.loggamma: float | torch.Tensor = loggamma
        self.loglambda: float = loglambda

    def _unpack_weights(
        self, Z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice a flat parameter vector into network weights and biases.

        Parameters
        ----------
        Z : torch.Tensor
            Flat parameter vectors, shape ``(batch, dim_vars)`` or a prefix
            thereof.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(W1, b1, W2, b2)`` with shapes
            ``(B, d, n_hidden)``, ``(B, n_hidden)``,
            ``(B, n_hidden, 1)``, ``(B, 1)``.
        """
        d, h = self.d, self.n_hidden
        W1 = Z[:, : d * h].reshape(-1, d, h)  # [B, d, hidden]
        b1 = Z[:, d * h : (d + 1) * h].reshape(-1, h)  # [B, hidden]
        W2 = Z[:, (d + 1) * h : (d + 1) * h + h][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -1].reshape(-1, 1)  # [B, 1]
        return W1, b1, W2, b2

    def _forward(
        self, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor,
        b2: torch.Tensor, X: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the BNN forward pass (ReLU activation).

        Parameters
        ----------
        W1, b1, W2, b2 : torch.Tensor
            Unpacked network parameters.
        X : torch.Tensor
            Input features, shape ``(n, d)``.

        Returns
        -------
        torch.Tensor
            Predictions, shape ``(B, n, 1)``.
        """
        return (
            torch.matmul(
                torch.max(
                    torch.matmul(X, W1) + b1[:, None, :],
                    torch.tensor([0.0], device=self.device),
                ),
                W2,
            )
            + b2[:, None, :]
        )

    def logp(
        self,
        Z: torch.Tensor,
        batchdataset: torch.Tensor,
        batchlabel: torch.Tensor,
        scale_sto: float = 1,
        max_param: float = _BNN_MAX_PARAM,
    ) -> torch.Tensor:
        r"""Compute the log-posterior :math:`\log P(W|Y,X)`.

        Parameters
        ----------
        Z : torch.Tensor
            Flat parameter vectors, shape ``(batch, dim_vars)``.
        batchdataset : torch.Tensor
            Mini-batch features, shape ``(n, d)``.
        batchlabel : torch.Tensor
            Mini-batch targets, shape ``(n, 1)``.
        scale_sto : float
            Stochastic scaling factor.
        max_param : float
            Clamp value for precision parameters.

        Returns
        -------
        torch.Tensor
            Log-posterior values, shape ``(batch,)``.
        """
        log_gamma = self.loggamma * torch.ones(Z.size(0), device=self.device)
        log_lambda = self.loglambda * torch.ones(Z.size(0), device=self.device)
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1, b1, W2, b2 = self._unpack_weights(Z)
        dnn_predict = self._forward(W1, b1, W2, b2, batchdataset)  # [B, n, 1]
        log_lik_data = -0.5 * batchdataset.shape[0] * (
            np.log(2 * np.pi) - log_gamma
        ) - (gamma_ / 2) * torch.sum(
            ((dnn_predict - batchlabel).squeeze(2)) ** 2, 1
        )
        log_prior_w = -0.5 * self.dim_wb * (
            np.log(2 * np.pi) - log_lambda
        ) - (lambda_ / 2) * (
            (W1**2).sum((1, 2))
            + (W2**2).sum((1, 2))
            + (b1**2).sum(1)
            + (b2**2).sum(1)
        )
        return log_lik_data * scale_sto + log_prior_w

    def score(
        self,
        Z: torch.Tensor,
        batchdataset: torch.Tensor,
        batchlabel: torch.Tensor,
        scale_sto: float = 1,
        max_param: float = _BNN_MAX_PARAM,
    ) -> torch.Tensor:
        r"""Compute the score :math:`\nabla_Z \log P(W|Y,X)`.

        Parameters
        ----------
        Z : torch.Tensor
            Flat parameter vectors, shape ``(batch, dim_vars)``.
        batchdataset : torch.Tensor
            Mini-batch features, shape ``(n, d)``.
        batchlabel : torch.Tensor
            Mini-batch targets, shape ``(n, 1)``.
        scale_sto : float
            Stochastic scaling factor.
        max_param : float
            Clamp value for precision parameters.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, dim_vars)``.
        """
        batch_Z = Z.shape[0]
        num_data = batchdataset.shape[0]
        log_gamma = self.loggamma * torch.ones(
            (batch_Z, 1), device=self.device
        )  # [B, 1]
        log_lambda = self.loglambda * torch.ones(
            (batch_Z, 1), device=self.device
        )
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1, b1, W2, b2 = self._unpack_weights(Z)

        dnn_onelinear = torch.matmul(batchdataset, W1) + b1[:, None, :]
        dnn_relu_onelinear = torch.max(
            dnn_onelinear, torch.tensor([0.0], device=self.device)
        )
        dnn_grad_relu = (
            torch.sign(dnn_onelinear) + 1
        ) / 2  # shape = [B, n, hidden]
        dnn_predict = (
            torch.matmul(dnn_relu_onelinear, W2) + b2[:, None, :]
        )  # shape = [B, n, 1]
        nabla_predict_b1 = dnn_grad_relu * W2.transpose(1, 2)  # [B, n, hidden]
        nabla_predict_W1 = nabla_predict_b1[:, :, None, :] * batchdataset[
            None, :, :, None
        ]  # [B, n, d, hidden]
        nabla_predict_W2 = dnn_relu_onelinear  # [B, n, hidden]
        nabla_predict_b2 = torch.ones_like(dnn_predict, device=self.device)  # [B, n, 1]

        nabla_predict_wb = torch.cat(
            (
                nabla_predict_W1.reshape(batch_Z, num_data, -1),
                nabla_predict_b1,
                nabla_predict_W2,
                nabla_predict_b2,
            ),
            dim=2,
        )
        nabla_wb = (
            scale_sto
            * gamma_
            * ((batchlabel - dnn_predict) * nabla_predict_wb).sum(1)
            - lambda_ * Z
        )
        return nabla_wb  # shape = [B, self.dim_vars]

    def rmse_llk(
        self,
        Z: torch.Tensor,
        batchdataset: torch.Tensor,
        batchlabel: torch.Tensor,
        mean_y_train: torch.Tensor,
        std_y_train: torch.Tensor,
        max_param: float = _BNN_MAX_PARAM,
    ) -> tuple[float, float]:
        r"""Compute test RMSE and test log-likelihood.

        Parameters
        ----------
        Z : torch.Tensor
            Flat parameter vectors, shape ``(batch, dim_vars)``.
        batchdataset : torch.Tensor
            Test features, shape ``(n, d)``.
        batchlabel : torch.Tensor
            Test targets, shape ``(n, 1)``.
        mean_y_train, std_y_train : torch.Tensor
            Training set target mean and standard deviation.
        max_param : float
            Clamp value for precision parameters.

        Returns
        -------
        tuple[float, float]
            ``(test_rmse, test_log_likelihood)``.
        """
        log_gamma = self.loggamma * torch.ones(
            (Z.size(0), 1), device=self.device
        )  # [B, 1]
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        W1, b1, W2, b2 = self._unpack_weights(Z)
        dnn_predict = self._forward(W1, b1, W2, b2, batchdataset)
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train  # [B, n, 1]
        predict_mean = dnn_predict_true.mean(0)
        test_rmse = (((predict_mean - batchlabel) ** 2).mean()) ** (0.5)
        logpy_xz = -0.5 * (
            np.log(2 * np.pi) - log_gamma[:, None, :]
        ) - 0.5 * gamma_[:, None, :] * (
            dnn_predict_true - batchlabel[None, :, :]
        ) ** 2
        test_llk = (
            torch.logsumexp(logpy_xz.squeeze(2), dim=0).mean()
            - np.log(Z.shape[0])
        )
        return test_rmse.item(), test_llk.item()

    def predict_y(
        self,
        Z: torch.Tensor,
        batchdataset: torch.Tensor,
        mean_y_train: torch.Tensor,
        std_y_train: torch.Tensor,
        max_param: float = _BNN_MAX_PARAM,
    ) -> torch.Tensor:
        r"""Predict response variable :math:`\hat{y}`.

        Parameters
        ----------
        Z : torch.Tensor
            Flat parameter vectors, shape ``(batch, dim_vars)``.
        batchdataset : torch.Tensor
            Input features, shape ``(n, d)``.
        mean_y_train, std_y_train : torch.Tensor
            Training set target mean and standard deviation.
        max_param : float
            Clamp value for precision parameters.

        Returns
        -------
        torch.Tensor
            Predictions in original scale, shape ``(B, n, 1)``.
        """
        W1, b1, W2, b2 = self._unpack_weights(Z)
        dnn_predict = self._forward(W1, b1, W2, b2, batchdataset)
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train
        return dnn_predict_true

    def model_selection(
        self,
        Z: torch.Tensor,
        batchdataset: torch.Tensor,
        batchlabel: torch.Tensor,
        mean_y_train: torch.Tensor,
        std_y_train: torch.Tensor,
        max_param: float = _BNN_MAX_PARAM,
    ) -> None:
        """Update ``loggamma`` heuristically based on prediction residuals.

        Parameters
        ----------
        Z : torch.Tensor
            Flat parameter vectors, shape ``(batch, dim_vars)``.
        batchdataset : torch.Tensor
            Training features, shape ``(n, d)``.
        batchlabel : torch.Tensor
            Training targets, shape ``(n, 1)``.
        mean_y_train, std_y_train : torch.Tensor
            Training set target mean and standard deviation.
        max_param : float
            Clamp value for precision parameters.
        """
        W1, b1, W2, b2 = self._unpack_weights(Z)
        dnn_predict = self._forward(W1, b1, W2, b2, batchdataset)
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train  # [B, n, 1]
        log_gamma_heu = -torch.log(
            ((dnn_predict_true - batchlabel[None, :, :]) ** 2).mean(1)
        )
        self.loggamma = log_gamma_heu


# ---------------------------------------------------------------------------
# Langevin SDE posterior
# ---------------------------------------------------------------------------


class Langevin_post(Toy_2D):
    """Posterior for a discretised double-well Langevin SDE.

    Generates synthetic observation data from a known trajectory and
    defines the log-posterior and score over the latent path.

    Parameters
    ----------
    num_interval : int
        Number of time discretisation intervals.
    num_obs : int
        Number of observations.
    beta : float
        Drift strength of the double-well potential.
    T : float
        Total simulation time.
    sigma : float
        Observation noise standard deviation.
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        num_interval: int = 100,
        num_obs: int = 20,
        beta: float = 10.0,
        T: float = 1.0,
        sigma: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        self.beta = beta
        self.sigma = sigma
        self.T = T
        self.dt: float = T / num_interval
        self.dim: int = num_interval
        self.z_dim: int = num_interval  # Override Toy_2D's z_dim=2
        self.device = device
        self.u_step: int = int(num_interval / num_obs)
        self.num_obs = num_obs
        self.upper_mask: torch.Tensor = torch.triu(
            torch.ones((self.dim, self.dim), device=device)
        ).contiguous().bool()
        self.upper1_mask: torch.Tensor = (
            1
            - torch.triu(
                torch.ones((self.dim, self.dim), device=device)
            ).transpose(0, 1)
        ).contiguous().bool()
        self.u_mask: torch.Tensor = (
            torch.arange(1, self.dim + 1) % self.u_step == 0
        )

        data_generator = torch.Generator().manual_seed(_LANGEVIN_SEED)
        xs = torch.randn((self.dim, 1), generator=data_generator)
        u = torch.zeros((1,))
        us_list: list[torch.Tensor] = []
        for i in range(self.dim):
            u = (
                u
                + self.beta * u * (1 - u**2) * self.dt
                + xs[i] * np.sqrt(self.dt)
            )
            us_list.append(u)
        self.u: torch.Tensor = torch.tensor(us_list).to(self.device)
        us = (torch.stack(us_list).T)[:, self.u_step - 1 :: self.u_step]
        noise = torch.randn(
            us.shape,
            generator=data_generator,
            dtype=us.dtype,
            device=us.device,
        )

        data = us + noise * self.sigma
        self.data: torch.Tensor = data.to(self.device)
        self.xs: torch.Tensor = xs.to(self.device)
        self.us: torch.Tensor = us.to(self.device)

    def _drift(self, us: torch.Tensor) -> torch.Tensor:
        """Compute the one-step drift mean for all interior time steps.

        Parameters
        ----------
        us : torch.Tensor
            Latent path values, shape ``(batch, dim)``.

        Returns
        -------
        torch.Tensor
            Drift means, shape ``(batch, dim - 1)``.
        """
        return (
            us[:, :-1]
            + self.beta
            * (us[:, :-1] - us[:, :-1] ** 3)
            / (1 + us[:, :-1] ** 2)
            * self.dt
        )

    def logp(self, us: torch.Tensor) -> torch.Tensor:
        """Compute the log-posterior of the latent SDE path.

        Parameters
        ----------
        us : torch.Tensor
            Latent path values, shape ``(batch, dim)``.

        Returns
        -------
        torch.Tensor
            Log-posterior values, shape ``(batch,)``.
        """
        us_mean = self._drift(us)
        us_mean_pad = torch.concatenate(
            [torch.zeros((us.shape[0], 1), device=self.device), us_mean],
            dim=-1,
        )
        logp = -torch.sum(
            (us - us_mean_pad) ** 2 / (2 * self.dt), dim=-1
        ) - torch.sum(
            (us[:, None, self.u_mask] - self.data[None, :, :]) ** 2
            / (2 * self.sigma**2),
            dim=(-1, -2),
        )
        return logp

    def score(self, us: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_{us} \log p(us | \text{data})`.

        Parameters
        ----------
        us : torch.Tensor
            Latent path values, shape ``(batch, dim)``.

        Returns
        -------
        torch.Tensor
            Score vectors, shape ``(batch, dim)``.
        """
        us_mean = self._drift(us)
        us_mean_pad = torch.concatenate(
            [torch.zeros((us.shape[0], 1), device=self.device), us_mean],
            dim=-1,
        )
        score_ll_part = -torch.sum(
            (us[:, None, self.u_mask] - self.data[None, :, :])
            / (self.sigma**2),
            dim=1,
        )
        score_ll = torch.zeros_like(us, device=self.device)
        score_ll[:, self.u_mask] = score_ll_part

        score_prior_1 = -(us - us_mean_pad) / (self.dt)
        score_prior_2 = -torch.concatenate(
            [
                (us_mean - us[:, 1:])
                / (self.dt)
                * (
                    1
                    - self.beta * self.dt
                    - self.beta
                    * self.dt
                    * 2
                    * (us[:, :-1] ** 2 - 1)
                    / (us[:, :-1] ** 2 + 1) ** 2
                ),
                torch.zeros((us.shape[0], 1), device=self.device),
            ],
            dim=-1,
        )
        return (score_prior_1 + score_prior_2) + score_ll

    def trace_plot(
        self,
        u: torch.Tensor,
        figpath: str | None = None,
        figname: str | None = None,
        figtitle: str = "",
    ) -> None:
        """Plot the posterior mean path with confidence interval and observations.

        Parameters
        ----------
        u : torch.Tensor
            Posterior samples of the latent path, shape ``(n_samples, dim)``.
        figpath : str or None
            Directory to save the figure.
        figname : str or None
            File name for the saved figure.
        figtitle : str
            Plot title.
        """
        u_np = u.detach().cpu().numpy()

        u_mean = u_np.mean(0)
        low_CI_bound, high_CI_bound = st.t.interval(
            0.95, len(u_mean), loc=u_mean, scale=np.std(u_np, 0)
        )
        u_true = self.u.detach().cpu().numpy().flatten()

        t = np.arange(self.dt, self.T + self.dt, self.dt)
        plt.plot(t, u_true, color="magenta", label="true path")
        plt.plot(t, u_mean, color="blue", label="sample path")
        plt.plot(t, low_CI_bound, color="black", linewidth=1.0)
        plt.plot(t, high_CI_bound, color="black", linewidth=1.0)
        plt.fill_between(
            t,
            low_CI_bound,
            high_CI_bound,
            facecolor="aqua",
            alpha=0.3,
            label="confidence interval",
        )
        obs_interval = self.T / self.num_obs
        plt.scatter(
            np.arange(obs_interval, self.T + obs_interval, obs_interval),
            self.data.detach().cpu().numpy(),
            color="r",
            marker=".",
            linewidth=0.5,
            label="observation",
        )

        plt.legend()
        plt.grid("on")
        plt.title(figtitle)
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, figname), dpi=600)
        plt.close()

    def contour_plot(
        self,
        bbox: list[float],
        fnet: torch.nn.Module | None = None,
        samples: NDArray | None = None,
        save_to_path: str | None = None,
        quiver: bool = True,
        t: int | None = None,
    ) -> None:
        """Not applicable for high-dimensional Langevin posterior."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

target_distribution: dict[str, type] = {
    "banana": Banana_shape,
    "multimodal": Multimodal,
    "8_gaussians": EightGaussians,
    "x_shaped": X_shaped,
    "student_uc": StudentTFullDim,
    "LRwaveform": LRwaveform,
    "Bnn_boston": Bnn,
    "Bnn_concrete": Bnn,
    "Bnn_power": Bnn,
    "Bnn_protein": Bnn,
    "Bnn_winered": Bnn,
    "Bnn_yacht": Bnn,
    "Langevin_post": Langevin_post,
}
