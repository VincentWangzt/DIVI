import torch
import numpy as np
from typing import Tuple, Optional


def _pairwise_squared_distances(
    samples_x: torch.Tensor,
    samples_y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances without materializing an
    [N, M, D] difference tensor.
    """
    if samples_y is None:
        samples_y = samples_x

    x_norm = (samples_x**2).sum(dim=-1, keepdim=True)
    y_norm = (samples_y**2).sum(dim=-1).unsqueeze(0)
    pairwise_dists = x_norm + y_norm - 2 * samples_x @ samples_y.transpose(0, 1)
    return pairwise_dists.clamp_min_(0.0)


class BaseKernel:
    """
    Base class for kernel functions.
    """

    def __init__(self, h: float = -1, name: str = 'BaseKernel'):
        self._h: float = h
        self.name: str = name

    @property
    def h(self) -> float:
        return self._h

    @h.setter
    def h(self, value: float):
        self._h = value

    def fit_h(self, samples: torch.Tensor) -> float:
        """
        Fit the kernel width h from samples.

        Args:
            samples (torch.Tensor): Tensor of shape [N, D]

        Returns:
            h (float): The fitted kernel width
        """
        raise NotImplementedError

    def pair_eval(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
        fit_h: bool = False,
        detach_h: bool = True,
    ) -> torch.Tensor:
        """
        Calculate the pairwise kernel matrix.

        Args:
            samples_x (torch.Tensor): Tensor of shape [N, D]
            samples_y (torch.Tensor, optional): Tensor of shape [M, D] or None (if None, use samples_x)
            fit_h (bool): Whether to fit h from the samples
        Returns:
            kxy (torch.Tensor): Pairwise kernel matrix of shape [N, M]
        """
        raise NotImplementedError

    def grad_all(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate k(x,y), grad_x, grad_y, and trace(grad_xy) simultaneously.

        Args:
            samples_x (torch.Tensor): Tensor of shape [N, D]
            samples_y (torch.Tensor, optional): Tensor of shape [M, D] or None (if None, use samples_x)

        Returns:
            (kxy, grad_x, grad_y, grad_xy_trace) (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing:
                - kxy (torch.Tensor): Pairwise kernel matrix [N, M]
                - grad_x (torch.Tensor): Gradient wrt x [N, M, D]
                - grad_y (torch.Tensor): Gradient wrt y [N, M, D]
                - grad_xy_trace (torch.Tensor): Trace of mixed gradient [N, M]
        """
        raise NotImplementedError


class GaussianKernel(BaseKernel):
    """
    Gaussian Kernel: k(x,y) = exp(-||x-y||^2 / (2h^2))
    """

    def __init__(self, h: float = -1, name: str = 'GaussianKernel'):
        super().__init__(h, name)

    def fit_h(self, samples: torch.Tensor) -> float:

        # Median heuristic
        pairwise_dists = _pairwise_squared_distances(samples)
        h = torch.median(pairwise_dists)
        # Using the formula from the original gaussian_kernel function: h = sqrt(0.5 * median / log(n+1))
        self.h = torch.sqrt(0.5 * h / np.log(samples.shape[0] + 1)).item()
        return self.h

    def pair_eval(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
        fit_h: bool = False,
        detach_h: bool = True,
    ) -> torch.Tensor:

        if samples_y is None:
            samples_y = samples_x

        pairwise_dists = _pairwise_squared_distances(samples_x, samples_y)
        if fit_h or self.h < 0:
            h = torch.median(
                pairwise_dists.detach() if detach_h else pairwise_dists
            )
            h = torch.sqrt(0.5 * h / np.log(samples_x.shape[0] + 1))
            self.h = h.detach().item()
        else:
            h = torch.as_tensor(
                self.h,
                device=samples_x.device,
                dtype=samples_x.dtype,
            )

        kxy = torch.exp(-pairwise_dists / h**2 / 2)
        return kxy

    def grad_all(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if samples_y is None:
            samples_y = samples_x

        if self.h < 0:
            self.fit_h(samples_x)

        diff = samples_x[:, None, :] - samples_y[None, :, :]
        pairwise_dists = (diff**2).sum(-1)
        kxy = torch.exp(-pairwise_dists / self.h**2 / 2)

        grad_x = -diff / self.h**2 * kxy[:, :, None]
        grad_y = diff / self.h**2 * kxy[:, :, None]

        dim = samples_x.shape[-1]
        term1 = dim / self.h**2
        term2 = pairwise_dists / self.h**4
        grad_xy_trace = kxy * (term1 - term2)

        return kxy, grad_x, grad_y, grad_xy_trace


class IMQKernel(BaseKernel):
    """
    Inverse Multi-Quadric Kernel: k(x,y) = (1 + ||x-y||^2/h)^(-1/2)
    This corresponds to beta=-0.5, c=1 in usual IMQ definition k(x,y)=(c^2 + ||x-y||^2)^beta
    """

    def __init__(self, h: float = -1, name: str = 'IMQKernel'):
        super().__init__(h, name)

    def fit_h(self, samples: torch.Tensor) -> float:

        pairwise_dists = ((samples[:, None, :] -
                           samples[None, :, :])**2).sum(-1)
        h = torch.median(pairwise_dists)
        self.h = h / np.log(samples.shape[0] + 1)
        return self.h

    def pair_eval(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
        fit_h: bool = False,
        detach_h: bool = True,
    ) -> torch.Tensor:

        if samples_y is None:
            samples_y = samples_x

        pairwise_dists = ((samples_x[:, None, :] -
                           samples_y[None, :, :])**2).sum(-1)
        if fit_h or self.h < 0:
            h = torch.median(
                pairwise_dists.detach() if detach_h else pairwise_dists
            )
            h = h / np.log(samples_x.shape[0] + 1)
            self.h = h.detach().item()
        else:
            h = torch.as_tensor(
                self.h,
                device=samples_x.device,
                dtype=samples_x.dtype,
            )

        kxy = (1 + pairwise_dists / h)**(-0.5)
        return kxy

    def grad_all(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if samples_y is None:
            samples_y = samples_x

        if self.h < 0:
            self.fit_h(samples_x)

        diff = samples_x[:, None, :] - samples_y[None, :, :]
        pairwise_dists = (diff**2).sum(-1)
        kxy = (1 + pairwise_dists / self.h)**(-0.5)

        k3 = kxy**3
        grad_x = -(1 / self.h) * k3[:, :, None] * diff
        grad_y = (1 / self.h) * k3[:, :, None] * diff

        dim = samples_x.shape[-1]
        k5 = kxy**5
        term1 = (dim / self.h) * k3
        term2 = (3 / self.h**2) * pairwise_dists * k5
        grad_xy_trace = term1 - term2

        return kxy, grad_x, grad_y, grad_xy_trace


class LaplaceKernel(BaseKernel):
    """
    Laplace Kernel: k(x,y) = exp(-||x-y||_1 / h)
    """

    def __init__(self, h: float = -1, name: str = 'LaplaceKernel'):
        super().__init__(h, name)

    def fit_h(self, samples: torch.Tensor) -> float:
        # Median heuristic
        pairwise_dists = torch.abs(samples[:, None, :] -
                                   samples[None, :, :]).sum(-1)
        h = torch.median(pairwise_dists)
        self.h = (h / np.log(samples.shape[0] + 1)).item()
        return self.h

    def pair_eval(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
        fit_h: bool = False,
        detach_h: bool = True,
    ) -> torch.Tensor:

        if samples_y is None:
            samples_y = samples_x

        pairwise_dists = torch.abs(samples_x[:, None, :] -
                                   samples_y[None, :, :]).sum(-1)
        if fit_h or self.h < 0:
            h = torch.median(
                pairwise_dists.detach() if detach_h else pairwise_dists
            )
            h = h / np.log(samples_x.shape[0] + 1)
            self.h = h.detach().item()
        else:
            h = torch.as_tensor(
                self.h,
                device=samples_x.device,
                dtype=samples_x.dtype,
            )

        kxy = torch.exp(-pairwise_dists / h)
        return kxy

    def grad_all(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if samples_y is None:
            samples_y = samples_x

        if self.h < 0:
            self.fit_h(samples_x)

        diff = samples_x[:, None, :] - samples_y[None, :, :]
        pairwise_dists = torch.abs(diff).sum(-1)
        kxy = torch.exp(-pairwise_dists / self.h)

        sgn = torch.sign(diff)
        grad_x = -sgn / self.h * kxy[:, :, None]
        grad_y = sgn / self.h * kxy[:, :, None]

        dim = samples_x.shape[-1]
        grad_xy_trace = -(dim / self.h**2) * kxy

        return kxy, grad_x, grad_y, grad_xy_trace


class RieszKernel(BaseKernel):
    """
    Riesz Kernel: k(x,y) = (||x||_1 + ||y||_1 - ||x-y||_1) / h
    """

    def __init__(self, h: float = -1, name: str = 'RieszKernel'):
        super().__init__(h, name)

    def fit_h(self, samples: torch.Tensor) -> float:

        norm_x = samples.norm(1, dim=-1)
        diff_norm = (samples[:, None, :] - samples[None, :, :]).norm(1, dim=-1)
        pairwise_dists = norm_x[:, None] + norm_x[None, :] - diff_norm
        h = torch.median(pairwise_dists)
        self.h = (h / np.log(samples.shape[0] + 1)).item()
        return self.h

    def pair_eval(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
        fit_h: bool = False,
        detach_h: bool = True,
    ) -> torch.Tensor:

        if samples_y is None:
            samples_y = samples_x

        norm_x = samples_x.norm(1, dim=-1)
        norm_y = samples_y.norm(1, dim=-1)
        diff_norm = (samples_x[:, None, :] - samples_y[None, :, :]).norm(
            1, dim=-1)
        pairwise_dists = norm_x[:, None] + norm_y[None, :] - diff_norm
        if fit_h or self.h < 0:
            h = torch.median(
                pairwise_dists.detach() if detach_h else pairwise_dists
            )
            h = h / np.log(samples_x.shape[0] + 1)
            self.h = h.detach().item()
        else:
            h = torch.as_tensor(
                self.h,
                device=samples_x.device,
                dtype=samples_x.dtype,
            )

        kxy = pairwise_dists / h
        return kxy

    def grad_all(
        self,
        samples_x: torch.Tensor,
        samples_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if samples_y is None:
            samples_y = samples_x

        if self.h < 0:
            self.fit_h(samples_x)

        norm_x = samples_x.norm(1, dim=-1)
        norm_y = samples_y.norm(1, dim=-1)
        diff = samples_x[:, None, :] - samples_y[None, :, :]
        diff_norm = diff.norm(1, dim=-1)

        kxy = (norm_x[:, None] + norm_y[None, :] - diff_norm) / self.h

        sgn_x = torch.sign(samples_x)
        sgn_y = torch.sign(samples_y)
        sgn_diff = torch.sign(diff)

        grad_x = (sgn_x[:, None, :] - sgn_diff) / self.h
        grad_y = (sgn_y[None, :, :] + sgn_diff) / self.h

        grad_xy_trace = torch.zeros_like(kxy)

        return kxy, grad_x, grad_y, grad_xy_trace


Kernels: dict[str, type[BaseKernel]] = {
    'gaussian': GaussianKernel,
    'imq': IMQKernel,
    'laplace': LaplaceKernel,
    'riesz': RieszKernel,
}
