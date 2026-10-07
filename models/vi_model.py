import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
import normflows as nf
from utils.logging import get_logger
from omegaconf.dictconfig import DictConfig

logger = get_logger()


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'")


def _inverse_softplus(value: float) -> float:
    return torch.log(torch.expm1(torch.tensor(value))).item()


class BaseVIModel(nn.Module):
    """
    Base class for variational inference models q_phi(z|epsilon).

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
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
        self.name = name
        self.config: DictConfig = config
        self.epsilon_dim: int = config.epsilon_dim
        self.z_dim: int = config.z_dim
        self.device: torch.device = torch.device(config.device)

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from prior.

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        raise NotImplementedError

    def forward(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing `z ~ q_phi(z|epsilon)` and its negative score term.

        Args:
            epsilon (torch.Tensor): Input noise `epsilon`.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score.
        """
        raise NotImplementedError

    def sampling(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample pairs `(epsilon, z)` from the variational model. No gradients computed.

        Args:
            num (int): Number of samples.
        Returns:
            (epsilon, z) (torch.Tensor, torch.Tensor): `epsilon` and `z` samples.
        """
        raise NotImplementedError

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the score term.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            score (torch.Tensor): Score with shape `[..., D_z]`.
        """
        raise NotImplementedError

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_phi(z|epsilon)` for given `z` and `epsilon`.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            log_prob (torch.Tensor): Log probability `log q_phi(z|epsilon)`.
        """
        raise NotImplementedError

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) of prior.

        Args: 
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        raise NotImplementedError


class ConditionalGaussian(BaseVIModel):
    """
    Conditional Gaussian `q_phi(z|epsilon)` parameterized by an MLP.

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - device: Device for computation.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.

    Optional `variance_init` sets a constant initial variance; omitting it
    preserves the default initialization for the chosen parameterization.

    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config, name="ConditionalGaussian")
        self.hidden_dim: int = config.hidden_dim
        self.num_layers: int = config.num_layers
        self.out_dim = self.z_dim * 2
        self.var_min = float(config.get('var_min', 1e-4))
        self.variance_parameterization = config.get(
            'variance_parameterization', 'softplus_var')
        if self.variance_parameterization not in ('softplus_var', 'logvar'):
            raise ValueError(
                "variance_parameterization must be one of "
                "('softplus_var', 'logvar')"
            )
        self.log_var_min = float(config.get('log_var_min', -20.0))
        self.activation = config.get('activation', 'silu')
        # Shared features feed separately trainable mean and variance heads.
        layers = []
        input_dim = self.epsilon_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(_make_activation(self.activation))
            input_dim = self.hidden_dim
        self.backbone = nn.Sequential(*layers)
        # Preserve the original joint initialization and subsequent RNG state.
        joint_head = nn.Linear(self.hidden_dim, self.out_dim)
        with torch.random.fork_rng(devices=[]):
            self.mean_head = nn.Linear(self.hidden_dim, self.z_dim)
            self.variance_head = nn.Linear(self.hidden_dim, self.z_dim)
        with torch.no_grad():
            self.mean_head.weight.copy_(joint_head.weight[:self.z_dim])
            self.mean_head.bias.copy_(joint_head.bias[:self.z_dim])
            self.variance_head.weight.copy_(joint_head.weight[self.z_dim:])
            self.variance_head.bias.copy_(joint_head.bias[self.z_dim:])
        variance_init = config.get('variance_init', None)
        if variance_init is not None:
            variance_init = float(variance_init)
            if not math.isfinite(variance_init) or variance_init <= 0:
                raise ValueError("variance_init must be finite and positive")
            if self.variance_parameterization == 'softplus_var':
                if variance_init < self.var_min:
                    raise ValueError("variance_init must be at least var_min")
                raw_init = variance_init + math.log(-math.expm1(-variance_init))
            else:
                raw_init = math.log(variance_init)
                if raw_init < self.log_var_min:
                    raise ValueError("log(variance_init) must be at least log_var_min")
            self._init_variance_head(raw_init)
        elif self.variance_parameterization == 'logvar':
            self._init_variance_head(float(config.get('log_var_init', 0.0)))

    def _init_variance_head(self, raw_init: float) -> None:
        """Initialize only the variance head; keep the mean head untouched."""
        with torch.no_grad():
            self.variance_head.weight.zero_()
            self.variance_head.bias.fill_(raw_init)

    def _raw_parameters(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate both heads together to preserve the original linear arithmetic."""
        features = self.backbone(epsilon)
        weight = torch.cat((self.mean_head.weight, self.variance_head.weight))
        bias = torch.cat((self.mean_head.bias, self.variance_head.bias))
        return F.linear(features, weight, bias).chunk(2, dim=-1)

    def optimizer_parameter_groups(
        self,
        mean_lr: float,
        variance_lr: float | None = None,
    ) -> list[dict]:
        """Keep shared features at the mean LR; default the variance LR to it."""
        if variance_lr is None:
            variance_lr = mean_lr
        return [
            {
                'params': (list(self.backbone.parameters())
                           + list(self.mean_head.parameters())),
                'lr': mean_lr,
            },
            {
                'params': list(self.variance_head.parameters()),
                'lr': variance_lr,
            },
        ]

    def _variance_from_raw(
        self,
        var_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variance_parameterization == 'logvar':
            log_var = var_raw.clamp(min=self.log_var_min)
            return torch.exp(log_var), log_var
        var = torch.nn.functional.softplus(var_raw)
        var = var.clamp(min=self.var_min)
        return var, var.log()

    def reparameterize(
        self,
        mu: torch.Tensor,
        var_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for `q_phi(z|epsilon)`.

        Args:
            mu (torch.Tensor): Mean.
            var_raw (torch.Tensor): Raw variance. Will be reparameterized by softplus.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std` where
            `z = mu + std * u` and `u ~ N(0, I)`.
        """
        var, _ = self._variance_from_raw(var_raw)
        std = torch.sqrt(var)
        u = torch.randn_like(mu)
        return mu + std * u, u / std

    def getmu(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `mu(epsilon)` from the network output split."""
        return self._raw_parameters(epsilon)[0]

    def getstd(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `std(epsilon)` by clamping variance and taking square root."""
        var_raw = self._raw_parameters(epsilon)[1]
        var, _ = self._variance_from_raw(var_raw)
        std = torch.sqrt(var)
        return std

    def forward(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing `z ~ q_phi(z|epsilon)` and its negative score term.

        Args:
            epsilon (torch.Tensor): Input noise `epsilon`.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std`.
        """
        mu, var_raw = self._raw_parameters(epsilon)
        z, neg_score_implicit = self.reparameterize(mu, var_raw)
        return z, neg_score_implicit

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from standard normal.

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.randn([num, self.epsilon_dim], ).to(self.device)

    def sampling(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample pairs `(epsilon, z)` from the conditional Gaussian. No gradients computed.

        Args:
            num (int): Number of samples.
        Returns:
            (epsilon, z) (torch.Tensor, torch.Tensor): `epsilon` and `z` samples.
        """
        with torch.no_grad():
            epsilon = self.sample_epsilon(num=num)
            Z, _ = self.forward(epsilon)
        return epsilon.clone().detach(), Z.clone().detach()

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the score term `-(z - mu) / var` used in objectives.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            score (torch.Tensor): Score `-(z - mu(epsilon)) / var(epsilon)`.
        """
        mu, var_raw = self._raw_parameters(epsilon)
        var, _ = self._variance_from_raw(var_raw)
        score = -(z - mu) / (var)
        return score

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_phi(z|epsilon)` for given `z` and `epsilon`.
        Supports broadcasting over leading dimensions.
        Args:
            z (torch.Tensor): shape [..., Dz]
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_prob (torch.Tensor): shape [...]
        """
        mu, var_raw = self._raw_parameters(epsilon)
        var, log_var = self._variance_from_raw(var_raw)
        # Gaussian log-likelihood per sample
        const = -0.5 * z.shape[-1] * math.log(2 * math.pi)
        ll = const - 0.5 * (log_var.sum(dim=-1) +
                            ((z - mu)**2 / var).sum(dim=-1))
        return ll

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) of prior.

        Args: 
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        const = -0.5 * epsilon.shape[-1] * math.log(2 * math.pi)
        return const - 0.5 * (epsilon**2).sum(dim=-1)


class ConditionalGaussianUniform(ConditionalGaussian):
    """
    Conditional Gaussian `q_phi(z|epsilon)` parameterized by an MLP. The prior on epsilon is `U[0,1]`.

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.
        - device: Device for computation.
    
    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config)
        self.name = "ConditionalGaussianUniform"

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from uniform [0,1].

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.rand([num, self.epsilon_dim], ).to(self.device)

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) under uniform [0,1] prior.

        Args:
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        # 1. Check if ALL elements in the last dimension are within [0, 1]
        # This results in a boolean tensor of shape [...]
        in_bounds = (epsilon >= 0) & (epsilon <= 1)
        all_in_bounds = in_bounds.all(dim=-1)

        # 2. Initialize log_prob with zeros (log(1) = 0)
        log_prob = torch.zeros_like(
            all_in_bounds,
            device=self.device,
            dtype=epsilon.dtype,
        )

        # 3. Set out-of-bounds entries to -inf
        log_prob[~all_in_bounds] = float('-inf')

        return log_prob


class ConditionalGaussianGlobalUniform(BaseVIModel):
    """
    Conditional Gaussian `q_phi(z|epsilon)` parameterized by an MLP.

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - device: Device for computation.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.

    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config, name="ConditionalGaussianGlobalUniform")
        self.hidden_dim: int = config.hidden_dim
        self.num_layers: int = config.num_layers
        self.out_dim = self.z_dim
        self.var_min = float(config.get('var_min', 1e-4))
        self.activation = config.get('activation', 'silu')
        self.variance_parameterization = config.get(
            'variance_parameterization', 'softplus_var')
        if self.variance_parameterization not in ('softplus_var', 'logvar'):
            raise ValueError(
                "variance_parameterization must be one of "
                "('softplus_var', 'logvar')"
            )
        self.global_log_var_min = float(config.get('global_log_var_min', -20.0))
        # The network outputs both mean and variance, with variance being global
        layers = []
        input_dim = self.epsilon_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(_make_activation(self.activation))
            input_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        self.net = nn.Sequential(*layers)
        if self.variance_parameterization == 'softplus_var':
            global_variance_init = config.get('global_variance_init', None)
            init_raw = 1.0 if global_variance_init is None else _inverse_softplus(
                float(global_variance_init))
        else:
            init_raw = float(config.get('global_log_var_init', 0.0))
        self.var_raw = nn.Parameter(torch.full((self.z_dim, ), init_raw))

    def _variance(self) -> torch.Tensor:
        if self.variance_parameterization == 'softplus_var':
            return torch.nn.functional.softplus(self.var_raw).clamp(
                min=self.var_min)
        log_var = self.var_raw.clamp(min=self.global_log_var_min)
        return torch.exp(log_var)

    def reparameterize(
        self,
        mu: torch.Tensor,
        var_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for `q_phi(z|epsilon)`.

        Args:
            mu (torch.Tensor): Mean.
            var_raw (torch.Tensor): Raw variance. Will be reparameterized by softplus.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std` where
            `z = mu + std * u` and `u ~ N(0, I)`.
        """
        var = self._variance()
        std = torch.sqrt(var)
        u = torch.randn_like(mu)
        return mu + std * u, u / std

    def getmu(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `mu(epsilon)` from the network output split."""
        return self.net(epsilon)

    def getstd(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `std(epsilon)` by clamping variance and taking square root."""
        return torch.sqrt(self._variance())

    def forward(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing `z ~ q_phi(z|epsilon)` and its negative score term.

        Args:
            epsilon (torch.Tensor): Input noise `epsilon`.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std`.
        """
        mu = self.net(epsilon)
        z, neg_score_implicit = self.reparameterize(mu, self.var_raw)
        return z, neg_score_implicit

    def sampling(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample pairs `(epsilon, z)` from the conditional Gaussian. No gradients computed.

        Args:
            num (int): Number of samples.
        Returns:
            (epsilon, z) (torch.Tensor, torch.Tensor): `epsilon` and `z` samples.
        """
        with torch.no_grad():
            epsilon = self.sample_epsilon(num=num)
            Z, _ = self.forward(epsilon)
        return epsilon.clone().detach(), Z.clone().detach()

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the score term `-(z - mu) / var` used in objectives.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            score (torch.Tensor): Score `-(z - mu(epsilon)) / var(epsilon)`.
        """
        mu = self.net(epsilon)
        return -(z - mu) / self._variance()

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_phi(z|epsilon)` for given `z` and `epsilon`.
        Supports broadcasting over leading dimensions.
        Args:
            z (torch.Tensor): shape [..., Dz]
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_prob (torch.Tensor): shape [...]
        """
        mu = self.net(epsilon)
        var = self._variance()
        # Gaussian log-likelihood per sample
        const = -0.5 * z.shape[-1] * math.log(2 * math.pi)
        ll = const - 0.5 * (var.log().sum(dim=-1) +
                            ((z - mu)**2 / var).sum(dim=-1))
        return ll

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from uniform [0,1].

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.rand([num, self.epsilon_dim], ).to(self.device)

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) under uniform [0,1] prior.

        Args:
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        # 1. Check if ALL elements in the last dimension are within [0, 1]
        # This results in a boolean tensor of shape [...]
        in_bounds = (epsilon >= 0) & (epsilon <= 1)
        all_in_bounds = in_bounds.all(dim=-1)

        # 2. Initialize log_prob with zeros (log(1) = 0)
        log_prob = torch.zeros_like(
            all_in_bounds,
            device=self.device,
            dtype=epsilon.dtype,
        )

        # 3. Set out-of-bounds entries to -inf
        log_prob[~all_in_bounds] = float('-inf')

        return log_prob


class ConditionalGaussianGlobal(ConditionalGaussianGlobalUniform):
    """
    Conditional Gaussian `q_phi(z|epsilon)` with global learnable variance
    and standard Normal prior on epsilon.

    This corresponds to KSIVI's SIMINet architecture: an MLP maps epsilon to
    mu(epsilon), with a shared (global) learnable variance parameter, and
    epsilon is sampled from N(0, I).

    Inherits all architecture and methods from ConditionalGaussianGlobalUniform,
    only overriding the prior distribution on epsilon.

    Required config parameters (same as parent):
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - device: Device for computation.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.name = "ConditionalGaussianGlobal"

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from standard Normal N(0, I).

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.randn([num, self.epsilon_dim]).to(self.device)

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) under standard Normal N(0, I) prior.

        Args:
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        const = -0.5 * epsilon.shape[-1] * math.log(2 * math.pi)
        return const - 0.5 * (epsilon**2).sum(dim=-1)


class _RealNVPCouplingNet(nn.Module):
    """MLP used for a RealNVP scale or translation map."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
        scale_clip: float | None = None,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_make_activation(activation))
            in_dim = hidden_dim
        final = nn.Linear(in_dim, dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.scale_clip = scale_clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.net(x)
        if self.scale_clip is not None:
            value = self.scale_clip * torch.tanh(value / self.scale_clip)
        return value


class RealNVP(BaseVIModel):
    """Explicit RealNVP variational distribution with an exact marginal density.

    Unlike the semi-implicit models above, ``epsilon`` is the flow base
    variable and the output is a deterministic bijection.  The density
    ``log q_phi(z)`` is therefore available exactly by change of variables.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, name="RealNVP")
        if self.epsilon_dim != self.z_dim:
            raise ValueError(
                "RealNVP requires epsilon_dim == z_dim because its transform "
                "is bijective."
            )

        self.hidden_dim = int(config.hidden_dim)
        self.num_flow_layers = int(config.get("num_flow_layers", 4))
        self.num_hidden_layers = int(config.get("num_hidden_layers", 2))
        self.activation = str(config.get("activation", "silu"))
        self.scale_clip = float(config.get("scale_clip", 3.0))
        base_trainable = bool(config.get("base_trainable", True))

        if self.num_flow_layers < 1:
            raise ValueError("num_flow_layers must be at least 1")

        flows: list[nf.flows.Flow] = []
        base_mask = torch.tensor(
            [float(i % 2) for i in range(self.z_dim)],
            dtype=torch.float32,
        )
        for layer_idx in range(self.num_flow_layers):
            mask = base_mask if layer_idx % 2 == 0 else 1.0 - base_mask
            scale_net = _RealNVPCouplingNet(
                dim=self.z_dim,
                hidden_dim=self.hidden_dim,
                num_hidden_layers=self.num_hidden_layers,
                activation=self.activation,
                scale_clip=self.scale_clip,
            )
            translation_net = _RealNVPCouplingNet(
                dim=self.z_dim,
                hidden_dim=self.hidden_dim,
                num_hidden_layers=self.num_hidden_layers,
                activation=self.activation,
                scale_clip=None,
            )
            flows.append(
                nf.flows.MaskedAffineFlow(
                    mask,
                    t=translation_net,
                    s=scale_net,
                )
            )

        base = nf.distributions.base.DiagGaussian(
            self.z_dim,
            trainable=base_trainable,
        )
        self.flow = nf.NormalizingFlow(base, flows)

    def sample_epsilon(self, num: int = 1000) -> torch.Tensor:
        epsilon, _ = self.flow.q0(num)
        return epsilon

    def forward(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = self.flow.forward_and_log_det(epsilon)
        # BaseSIVIRunner expects a second return value.  NFVIRunner does not
        # use the conditional-score term, since log q_phi(z) is exact.
        return z, torch.zeros_like(z)

    def forward_and_log_prob(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform base samples and return their exact flow log density."""
        z, log_det = self.flow.forward_and_log_det(epsilon)
        log_q = self.flow.q0.log_prob(epsilon) - log_det
        return z, log_q

    def sampling(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            epsilon = self.sample_epsilon(num=num)
            z, _ = self.forward_and_log_prob(epsilon)
        return epsilon.detach().clone(), z.detach().clone()

    def sampling_with_log_prob(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            epsilon = self.sample_epsilon(num=num)
            z, log_q = self.forward_and_log_prob(epsilon)
        return (
            epsilon.detach().clone(),
            z.detach().clone(),
            log_q.detach().clone(),
        )

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del epsilon
        leading_shape = z.shape[:-1]
        flat_z = z.reshape(-1, self.z_dim)
        return self.flow.log_prob(flat_z).reshape(leading_shape)

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del epsilon
        leading_shape = z.shape
        with torch.enable_grad():
            flat_z = z.detach().reshape(-1, self.z_dim).requires_grad_(True)
            log_q = self.flow.log_prob(flat_z)
            score = torch.autograd.grad(log_q.sum(), flat_z)[0]
        return score.reshape(leading_shape)

    def log_q_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        leading_shape = epsilon.shape[:-1]
        flat_epsilon = epsilon.reshape(-1, self.epsilon_dim)
        return self.flow.q0.log_prob(flat_epsilon).reshape(leading_shape)


VIModel: dict[str, type[BaseVIModel]] = {
    "ConditionalGaussian": ConditionalGaussian,
    "ConditionalGaussianUniform": ConditionalGaussianUniform,
    "ConditionalGaussianGlobalUniform": ConditionalGaussianGlobalUniform,
    "ConditionalGaussianGlobal": ConditionalGaussianGlobal,
    "RealNVP": RealNVP,
}
