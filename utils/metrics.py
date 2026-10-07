import torch
import torch.nn.functional as F
from utils.kernels import GaussianKernel, BaseKernel
from typing import Optional, Callable


def _compute_gaussian_ksd_chunked(
    x: torch.Tensor,
    scores: torch.Tensor,
    kernel: GaussianKernel,
    chunk_size: int = 256,
) -> float:
    """
    Compute Gaussian-kernel KSD without materializing [N, N, D] tensors.
    """
    n, dim = x.shape
    if n < 2:
        raise ValueError("KSD requires at least two samples.")

    if kernel.h < 0:
        kernel.fit_h(x)

    h = torch.as_tensor(kernel.h, device=x.device, dtype=x.dtype)
    h2 = h.square()
    h4 = h2.square()

    x_norm = (x**2).sum(dim=1)
    score_x = (scores * x).sum(dim=1)
    total = x.new_tensor(0.0)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = x[start:end]
        scores_chunk = scores[start:end]

        x_dot = x_chunk @ x.transpose(0, 1)
        score_dot = scores_chunk @ scores.transpose(0, 1)
        score_x_chunk = (scores_chunk * x_chunk).sum(dim=1, keepdim=True)
        score_x_cross = scores_chunk @ x.transpose(0, 1)
        x_score_cross = x_chunk @ scores.transpose(0, 1)

        dist2 = (
            (x_chunk**2).sum(dim=1, keepdim=True)
            + x_norm.unsqueeze(0)
            - 2 * x_dot
        ).clamp_min_(0.0)
        kxy = torch.exp(-dist2 / (2 * h2))

        score_diff = (
            score_x_chunk
            - score_x_cross
            - x_score_cross
            + score_x.unsqueeze(0)
        )
        u_matrix = kxy * (
            score_dot
            + score_diff / h2
            + (dim / h2 - dist2 / h4)
        )

        diag_cols = torch.arange(start, end, device=x.device)
        u_matrix[torch.arange(end - start, device=x.device), diag_cols] = 0.0
        total = total + u_matrix.sum()

    ksd = total / (n * (n - 1))
    return ksd.item()


def compute_sliced_wasserstein(
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_projections: int = 1000,
        device: torch.device = torch.device("cpu"),
        p: int = 2,
) -> float:
    """
    Compute the Sliced Wasserstein Distance between two sets of samples.

    Args:
        x1 (torch.Tensor): Samples from the first distribution, shape (N, D).
        x2 (torch.Tensor): Samples from the second distribution, shape (M, D).
        num_projections (int): Number of random projections.
        device (torch.device): Device to perform computations on.
        p (int): The order of the Wasserstein distance.

    Returns:
        float: The estimated Sliced Wasserstein Distance.
    """

    # Ensure data is on the correct device
    x1 = x1.to(device)
    x2 = x2.to(device)

    dim = x1.shape[1]

    # Generate random directions
    projections = torch.randn(dim, num_projections).to(device)
    # L2 normalization
    projections = projections / torch.norm(projections, dim=0, keepdim=True)

    # Project the samples
    x1_projections = torch.matmul(x1, projections)  # (N, num_projections)
    x2_projections = torch.matmul(x2, projections)  # (M, num_projections)

    # Sort the projections
    x1_sorted, _ = torch.sort(x1_projections, dim=0)
    x2_sorted, _ = torch.sort(x2_projections, dim=0)

    # --- CORRECTION: Handle N != M using Interpolation ---
    n = x1_sorted.shape[0]
    m = x2_sorted.shape[0]

    if n != m:
        # We need to interpolate the quantile functions to the same size.
        # We assume x1 and x2 are independent samples from distributions.

        # Determine target size (usually the larger one preserves more info)
        target_size = max(n, m)

        # Helper to interpolate: (Batch, Channels, Length)
        def interpolate_projections(tensor, target_len):
            # Tensor shape: (Len, Projections) -> (1, Projections, Len)
            t = tensor.permute(1, 0).unsqueeze(0)
            # Interpolate
            t = F.interpolate(t,
                              size=target_len,
                              mode='linear',
                              align_corners=True)
            # Reshape back: (Target_Len, Projections)
            return t.squeeze(0).permute(1, 0)

        if n != target_size:
            x1_sorted = interpolate_projections(x1_sorted, target_size)

        if m != target_size:
            x2_sorted = interpolate_projections(x2_sorted, target_size)

    # Calculate L_p distance between sorted projections
    # Now shapes are guaranteed to match
    diff = torch.abs(x1_sorted - x2_sorted)
    distance = torch.pow(diff, p)

    # Average over projections and samples, then take p-th root
    sw_distance = torch.pow(torch.mean(distance), 1.0 / p)

    return sw_distance.item()


def compute_mmd(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel: Optional[BaseKernel] = None,
) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples. Defaults to using a Gaussian kernel with median heuristic bandwidth of combined samples.

    Args:
        x1 (torch.Tensor): Samples from the first distribution, shape (N, D).
        x2 (torch.Tensor): Samples from the second distribution, shape (M, D).
        kernel (Optional[BaseKernel]): Kernel object to use.

    Returns:
        float: The estimated MMD.
    """

    if kernel is None:
        kernel = GaussianKernel()
        combined_samples = torch.cat([x1, x2], dim=0)
        h = kernel.fit_h(combined_samples)
        kernel._h = h

    # Compute pairwise kernels
    xx = kernel.pair_eval(x1, x1).mean()
    yy = kernel.pair_eval(x2, x2).mean()
    xy = kernel.pair_eval(x1, x2).mean()

    mmd = xx + yy - 2 * xy

    return mmd.item()


def compute_ksd(
    x: torch.Tensor,
    scores: torch.Tensor,
    kernel: Optional[BaseKernel] = None,
) -> float:
    """
    Compute the Kernelized Stein Discrepancy (KSD) for a set of samples given a score function. Defaults to using a Gaussian kernel with median heuristic bandwidth.
    Args:
        x (torch.Tensor): Samples from the distribution, shape (N, D).
        scores: Scores (gradient of log density) at given samples, shape (N, D).
        kernel (Optional[BaseKernel]): Kernel object to use.
    Returns:
        float: The estimated KSD.
    """
    if kernel is None:
        kernel = GaussianKernel()
        h = kernel.fit_h(x)
        kernel._h = h

    if isinstance(kernel, GaussianKernel):
        return _compute_gaussian_ksd_chunked(x, scores, kernel)

    k, grad_x, grad_y, tr_grad_xy = kernel.grad_all(x, x)

    # Term 1: s(x)^T s(y) k(x,y)
    term1 = torch.matmul(scores, scores.T) * k

    # Term 2: s(x)^T \nabla_y k(x,y)
    term2 = torch.einsum('id,ijd->ij', scores, grad_y)

    # Term 3: \nabla_x k(x,y)^T s(y)
    term3 = torch.einsum('ijd,jd->ij', grad_x, scores)

    # Term 4: tr(\nabla_x \nabla_y^T k(x,y))
    term4 = tr_grad_xy

    # Sum all terms
    u_matrix = term1 + term2 + term3 + term4

    # set diagonal to zero to use unbiased estimator
    u_matrix.fill_diagonal_(0.0)

    n = x.shape[0]
    ksd = u_matrix.sum() / (n * (n - 1))
    return ksd.item()
