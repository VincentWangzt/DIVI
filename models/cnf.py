# This file is adapted from:
# https://github.com/VincentStimper/normalizing-flows

import normflows as nf
import torch
import numpy as np
from torch.nn.functional import softplus
from torch import nn

class SiLUMLP(nn.Module):
    """
    Simple MLP with SiLU activations. Initializes last layer to zero.

    Args:
      layer_sizes: List of layer sizes, including input and output sizes
    """

    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.SiLU())
        for param in layers[-1].parameters():
            torch.nn.init.zeros_(param)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CondMaskedAffineFlow(nf.flows.Flow):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(
        self,
        b: torch.Tensor,
        t: nn.Module,
        s: nn.Module,
    ):
        """Constructor

        Args:
          b (torch.Tensor): mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t (nn.Module): translation neural network
          s (nn.Module): scale neural network
        """
        super().__init__()
        # self.b_cpu = b.view(1, *b.size())
        self.b_cpu = b
        self.b: torch.Tensor
        self.register_buffer("b", self.b_cpu)
        self.min_s: float = 1e-6

        self.s: nn.Module
        self.add_module("s", s)
        self.t: nn.Module
        self.add_module("t", t)

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
    ):
        """
        Forward pass of the flow

        Args:
          z (torch.Tensor): Input tensor of shape (..., latent_size)
          context (torch.Tensor): Conditioning context of shape (..., context_size)
        """
        z_masked = self.b * z
        scale = self.s(torch.cat([z_masked, context], dim=-1))
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(torch.cat([z_masked, context], dim=-1))
        trans = torch.where(torch.isfinite(trans), trans, nan)

        #z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        trafo_scale = softplus(scale) + self.min_s
        z_ = z_masked + (1 - self.b) * (z * trafo_scale + trans)

        #log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        log_det = torch.sum(
            (1 - self.b) * trafo_scale.log(),
            dim=list(range(-1, -1 - self.b.dim(), -1)),
        )

        return z_, log_det

    def inverse(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
    ):
        """
        Inverse pass of the flow

        Args:
          z (torch.Tensor): Input tensor of shape (..., latent_size)
          context (torch.Tensor): Conditioning context of shape (..., context_size)
        """
        z_masked = self.b * z
        scale = self.s(torch.cat([z_masked, context], dim=-1))
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(torch.cat([z_masked, context], dim=-1))
        trans = torch.where(torch.isfinite(trans), trans, nan)

        #z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        trafo_scale = softplus(scale) + self.min_s
        z_ = z_masked + (1 - self.b) * (z - trans) / trafo_scale

        #log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        log_det = -torch.sum(
            (1 - self.b) * trafo_scale.log(),
            dim=list(range(-1, -1 - self.b.dim(), -1)),
        )
        return z_, log_det


class AffineConstFlow(nf.flows.Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """Constructor

        Args:
          shape: Shape of the coupling layer
          scale: Flag whether to apply scaling
          shift: Flag whether to apply shift
          logscale_factor: Optional factor which can be used to control the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("s", torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1,
                                        as_tuple=False)[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)  # type: ignore
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)  # type: ignore
        return z_, log_det


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)

    def forward(self, z, context=None):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(
                z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) *
                           torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().forward(z)

    def inverse(self, z, context=None):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().inverse(z)


class Logit(nf.flows.Flow):
    """Logit mapping of image tensor, see RealNVP paper

    ```
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    ```
    
    Args:
        alpha: Alpha parameter, see above

    """

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, z, context=None):
        """
        Forward pass of the flow. Convert from pre-logit (R) to logit space ([0, 1]).

        Args:
          z (torch.Tensor): Input tensor of shape (..., latent_size)
          context (torch.Tensor): Conditioning context of shape (..., context_size)
        """
        beta = 1 - 2 * self.alpha
        ls = torch.nn.functional.logsigmoid(z)
        mls = torch.nn.functional.logsigmoid(-z)
        log_det = ls + mls - np.log(beta)
        log_det = torch.sum(log_det, dim=-1)
        z = (torch.sigmoid(z) - self.alpha) / beta
        return z, log_det

    def inverse(self, z, context=None):
        """
        Inverse pass of the flow. Convert from logit ([0, 1]) to pre-logit space (R).
        Args:
          z (torch.Tensor): Input tensor of shape (..., latent_size)
          context (torch.Tensor): Conditioning context of shape (..., context_size)
        """
        beta = 1 - 2 * self.alpha
        z = self.alpha + beta * z
        logz = torch.log(z)
        log1mz = torch.log(1 - z)
        z = logz - log1mz
        log_det = np.log(beta) - logz - log1mz
        log_det = torch.sum(log_det, dim=-1)
        return z, log_det


class DiagGaussian(nf.distributions.BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, trainable=True):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, )
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, context=None):
        eps = torch.randn((num_samples, ) + self.shape,
                          dtype=self.loc.dtype,
                          device=self.loc.device)
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2),
            dim=list(range(-1, -self.n_dim - 1, -1)))
        return z, log_p

    def log_prob(self, z, context=None):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(
                (z - self.loc) / torch.exp(log_scale), 2),
            dim=list(range(-1, -self.n_dim - 1, -1)),
        )
        return log_p


class ConditionalNormalizingFlow(nf.ConditionalNormalizingFlow):
    """
    Subclass of nf.ConditionalNormalizingFlow to allow multiple batch dimensions
    """

    def log_prob(self, x, context):
        """
        Compute log probability of data point x given context

        Args:
          x (torch.Tensor): Input tensor of shape (..., latent_size)
          context (torch.Tensor): Conditioning context of shape (..., context_size)
        Returns:
          log_p (torch.Tensor): Log probability of shape (...)
        """
        log_q = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(
                z,
                context=context,
            )  # type: ignore
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q


def generate_cond_real_nvp(
    K,
    latent_size,
    hidden_size,
    context_size,
    device,
    act_norm=True,
    logit=False,
) -> nf.ConditionalNormalizingFlow:
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = SiLUMLP([
            latent_size + context_size,
            hidden_size,
            latent_size,
        ])
        t = SiLUMLP([
            latent_size + context_size,
            hidden_size,
            latent_size,
        ])
        if i % 2 == 0:
            flows += [CondMaskedAffineFlow(b, t, s)]
        else:
            flows += [CondMaskedAffineFlow(1 - b, t, s)]
        if act_norm:
            flows += [ActNorm(latent_size)]
    if logit:
        flows += [Logit(0.0)]

    # Set q0
    q0 = DiagGaussian(latent_size, trainable=False)

    # Construct flow model
    nfm = ConditionalNormalizingFlow(q0, flows, None)

    nfm = nfm.to(device)

    # Initialize ActNorm
    if act_norm:
        num_samples = 2**7
        z, _ = nfm.sample(num_samples,
                          context=torch.randn([num_samples, context_size],
                                              device=device))

    return nfm
