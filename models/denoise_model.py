import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from utils.logging import get_logger

logger = get_logger()


class BaseDenoiseModel(nn.Module):
    """
    Base class for denoising nabla_z q_phi(z|epsilon), estimates psi(z).

    Required config parameters:
        - z_dim: Dimension of latent z.
        - device: Device for computation.

    Args:
        config (DictConfig): Configuration object.
        name (str): Name of the model.
    """

    def __init__(
        self,
        config: DictConfig,
        name: str = '',
    ):
        super().__init__()
        assert name != '', "Please provide a name for the model."
        self.config: DictConfig = config
        self.name: str = name
        self.z_dim: int = config.z_dim
        self.device: torch.device = torch.device(config.device)
        self.jitter: float = 1e-4

    @torch.no_grad()
    def fit(
        self,
        z: torch.Tensor,
        joint_score: torch.Tensor,
        initialize: bool = False,
    ) -> tuple[float, int]:
        """
        Fit model parameters from provided samples.

        Args:
            z (torch.Tensor): Samples of z with shape [..., Dz].
            joint_score (torch.Tensor): Samples of joint score nabla_z q_phi(z|epsilon) with shape [..., Dz].
            initialize (bool): If True, reset parameters before fitting.
        Returns:
            (mse, num_steps) (float, int): average mean squared error and number of optimization steps.
        """
        raise NotImplementedError

    def score(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the estimated score psi(z).

        Args:
            z (torch.Tensor): shape [..., Dz]
        Returns:
            score (torch.Tensor): shape [..., Dz]
        """
        raise NotImplementedError


class GaussianDenoiseModel(BaseDenoiseModel):
    """
    Gaussian-based denoising model. Minimizes E[||psi(z) - joint_score||^2] assuming z is Gaussian.
    
    psi(z) = -Sigma^{-1}(z - mu).
    """

    def __init__(
        self,
        config: DictConfig,
        name: str = 'GaussianDenoiseModel',
    ):
        super().__init__(config, name)
        # Parameters of the Gaussian distribution
        # Register as buffers so they are saved with state_dict but not optimized by optimizer
        self.mu: torch.Tensor
        self.precision: torch.Tensor
        self.register_buffer('mu', torch.zeros(self.z_dim))
        self.register_buffer('precision', torch.eye(self.z_dim))

    @torch.no_grad()
    def fit(
        self,
        z: torch.Tensor,
        joint_score: torch.Tensor,
        initialize: bool = False,
    ) -> tuple[float, int]:
        """
        Estimate Gaussian parameters (mean and precision) from z.
        """
        # z: [Batch, z_dim]
        N = z.shape[0]

        # Estimate Mean
        mu = z.mean(dim=0)

        # Estimate Covariance
        z_centered = z - mu
        cov = (z_centered.t() @ z_centered) / (N - 1)

        # Regularization for numerical stability
        cov = cov + self.jitter * torch.eye(self.z_dim, device=self.device)

        # Precision matrix
        try:
            precision = torch.linalg.inv(cov)
        except RuntimeError:
            logger.warning(
                "Covariance matrix is singular, using pseudo-inverse for precision estimation."
            )
            precision = torch.linalg.pinv(cov)

        self.mu.copy_(mu)
        self.precision.copy_(precision)

        # Calculate MSE for monitoring
        estimated_score = self.score(z)
        mse = ((estimated_score - joint_score)**2).sum(dim=-1).mean().item()

        return mse, 1

    def score(self, z: torch.Tensor) -> torch.Tensor:
        # psi(z) = -Sigma^{-1}(z - mu) = -(z-mu) @ Sigma^{-1}.T
        # precision is symmetric
        return -(z - self.mu) @ self.precision


class LinearDenoiseModel(BaseDenoiseModel):
    """
    Linear regression model for denoising.
    Fits psi(z) = zW + b to the joint score.

    """

    def __init__(
        self,
        config: DictConfig,
        name: str = 'LinearDenoiseModel',
    ):
        super().__init__(config, name)
        self.weight: torch.Tensor
        self.bias: torch.Tensor
        self.register_buffer('weight', torch.zeros(self.z_dim, self.z_dim))
        self.register_buffer('bias', torch.zeros(self.z_dim))

    @torch.no_grad()
    def fit(
        self,
        z: torch.Tensor,
        joint_score: torch.Tensor,
        initialize: bool = False,
    ) -> tuple[float, int]:
        """
        Fit linear model using Least Squares.
        Target Y = joint_score. Input X = z.
        """
        # Solve argmin_W || X W_aug^T - Y ||^2 where X_aug = [z, 1], W_aug = [W, b]
        # Or simply use torch.linalg.lstsq

        N = z.shape[0]
        device = self.device

        # Augment z with ones for bias
        ones = torch.ones(N, 1, device=device, dtype=z.dtype)
        X = torch.cat([z, ones], dim=1)  # [N, D+1]
        Y = joint_score  # [N, D]

        # Normal Equations: (X^T X) W_aug^T = X^T Y
        XTX = X.t() @ X
        XTY = X.t() @ Y

        # Regularization
        XTX = XTX + self.jitter * torch.eye(XTX.shape[0], device=device)

        try:
            W_aug_T = torch.linalg.solve(XTX, XTY)
        except RuntimeError:
            logger.warning(
                "X^T X is singular, using least squares solution for linear denoise model."
            )
            W_aug_T = torch.linalg.lstsq(XTX, XTY).solution

        # W_aug_T is [D+1, D]
        # Weights: [D, D] (first D rows)
        # Bias: [1, D] (last row)

        W_T: torch.Tensor = W_aug_T[:self.z_dim, :]
        b: torch.Tensor = W_aug_T[self.z_dim, :]

        # Update parameters
        # nn.Linear weights are [Out, In], so we need transpose of W_T
        self.weight.copy_(W_T.t())
        self.bias.copy_(b)

        pred = self.score(z)
        mse = ((pred - joint_score)**2).sum(dim=-1).mean().item()
        return mse, 1

    def score(self, z: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(z, self.weight, self.bias)


class MLPDenoiseModel(BaseDenoiseModel):
    """
    MLP-based denoising model using SiLU activation.

    Required config parameters:
        - hidden_dim (int): Dimension of hidden layers.
        - num_layers (int): Number of hidden layers.
    """

    def __init__(
        self,
        config: DictConfig,
        name: str = 'MLPDenoiseModel',
    ):
        super().__init__(config, name)
        self.hidden_dim = self.config['hidden_dim']
        self.num_layers = self.config['num_layers']

        layers = []
        in_dim = self.z_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.z_dim))

        self.net = nn.Sequential(*layers)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResidueBlock(nn.Module):
    """
    Residual block with SiLU activation.

    Args:
        dim (int): Dimension of input and output.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ResidueMLPDenoiseModel(BaseDenoiseModel):
    """
    MLP utilizing residue flow (residual blocks) with SiLU activation.

    Config parameters:
        - hidden_dim (int): Dimension of hidden layers. Default: 64.
        - num_layers (int): Number of residual blocks. Note that
            num_layers is actually the number of Residue Blocks. Default: 2.
    """

    def __init__(
        self,
        config: DictConfig,
        name: str = 'ResidueMLPDenoiseModel',
    ):
        super().__init__(config, name)
        self.hidden_dim: int = self.config["hidden_dim"]
        self.num_layers: int = self.config["num_layers"]

        self.in_proj = nn.Linear(self.z_dim, self.hidden_dim)
        self.act = nn.SiLU()

        blocks = []
        for _ in range(self.num_layers):
            blocks.append(ResidueBlock(self.hidden_dim))
        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Linear(self.hidden_dim, self.z_dim)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(z))
        h = self.blocks(h)
        return self.out_proj(h)


DenoiseModel: dict[str, type[BaseDenoiseModel]] = {
    "GaussianDenoiseModel": GaussianDenoiseModel,
    "LinearDenoiseModel": LinearDenoiseModel,
    "MLPDenoiseModel": MLPDenoiseModel,
    "ResidueMLPDenoiseModel": ResidueMLPDenoiseModel,
}
