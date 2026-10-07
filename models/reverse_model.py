import torch
import torch.nn as nn
import math
import torch.distributions as dist
from utils.logging import get_logger
from omegaconf.dictconfig import DictConfig

logger = get_logger()


class BaseReverseConditionalModel(nn.Module):
    """
    Base class for reverse conditional models q_psi(epsilon|z).

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
        self.config: DictConfig = config
        self.name: str = name
        self.epsilon_dim: int = config.epsilon_dim
        self.z_dim: int = config.z_dim
        self.device: torch.device = torch.device(config.device)
        self.jitter: float = 1e-4

    def log_prob(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_psi(epsilon|z)` for given `epsilon` and `z`.

        Args:
            epsilon (torch.Tensor): Input `epsilon` of shape `[..., D_epsilon]`.
            z (torch.Tensor): Conditioning variable of shape `[..., D_z]`.
        Returns:
            log_prob (torch.Tensor): Log probability `log q_psi(epsilon|z)`.
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample `epsilon ~ q_psi(epsilon|z)`. No gradients computed, returns the detached tensors.

        Args:
            z (torch.Tensor): Conditioning batch `[B, D_z]`.
            num_samples (int): Number of samples per `z`.
        Returns:
            (z_aux, epsilon_aux, log_prob) (torch.Tensor, torch.Tensor, torch.Tensor): 
                - tiled conditioning `z_aux` of shape `[B, num_samples, D_z]`,
                - samples `epsilon_aux` of shape `[B, num_samples, D_epsilon]`,
                - log probabilities `log_prob` of shape `[B, num_samples]`.
        """
        raise NotImplementedError

    @torch.no_grad()
    def fit(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
        initialize: bool = False,
    ) -> tuple[float, int]:
        """
        Fit model parameters from provided samples.

        Args:
            epsilon (torch.Tensor): Samples of epsilon with shape [..., De].
            z (torch.Tensor): Samples of z with shape [..., Dz].
            initialize (bool): If True, reset parameters before fitting.
        Returns:
            (avg_nll, num_steps) (float, int): average negative log-likelihood and number of optimization steps.
        """
        raise NotImplementedError


class ConditionalMixtureOfGaussianReverse(BaseReverseConditionalModel):
    """
    Mixture-of-Gaussians reverse model for q_psi(epsilon|z) via a joint GMM.

    We model the joint [epsilon, z] with a K-component Gaussian mixture:
        p([e, z]) = sum_k pi_k N([e, z]; mu_k, Sigma_k)

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - hidden_dim: Number of mixture components K.
        - max_em_iters: Maximum EM iterations for fitting.
        - device: Device for computation.

    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config, name="ConditionalMixtureOfGaussianReverse")
        self.num_components: int = config.hidden_dim
        self.max_iters: int = config.max_em_iters
        self.abs_stop_tol: float = 1e-4

        D = self.epsilon_dim + self.z_dim

        # Mixture parameters stored as buffers (no gradients required)
        self.pi: torch.Tensor
        self.mu: torch.Tensor
        self.Sigma: torch.Tensor
        self.cond_A: torch.Tensor
        self.cond_b: torch.Tensor
        self.cond_cov: torch.Tensor
        self.register_buffer(
            'pi',
            torch.ones(self.num_components).to(self.device) /
            self.num_components,
        )
        self.register_buffer(
            'mu',
            torch.zeros(self.num_components, D).to(self.device),
        )
        self.register_buffer(
            'Sigma',
            torch.stack(
                [torch.eye(D) for _ in range(self.num_components)],
                dim=0,
            ).to(self.device),
        )
        self.register_buffer(
            "cond_A",
            torch.zeros(self.num_components, self.epsilon_dim,
                        self.z_dim).to(self.device),
        )
        self.register_buffer(
            "cond_b",
            torch.zeros(self.num_components, self.epsilon_dim).to(self.device),
        )
        self.register_buffer(
            "cond_cov",
            torch.zeros(self.num_components, self.epsilon_dim,
                        self.epsilon_dim).to(self.device),
        )

    @torch.no_grad()
    def fit(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
        initialize: bool = False,
    ) -> tuple[float, int]:
        """
        Fit GMM parameters to joint samples X = [epsilon, z] via EM in one call.

        Args:
            epsilon (torch.Tensor): [..., De]
            z (torch.Tensor): [..., Dz]
            initialize (bool): If True, reset parameters before fitting.
        Returns:
            (avg_nll, num_steps) (float, int): Average negative log-likelihood and number of iterations.
        """
        assert epsilon.shape[-1] == self.epsilon_dim and z.shape[
            -1] == self.z_dim, "Dimension mismatch in fit inputs."

        epsilon = epsilon.clone().detach().reshape(-1, self.epsilon_dim)
        z = z.clone().detach().reshape(-1, self.z_dim)

        # Concatenate samples
        X = torch.cat([epsilon, z], dim=1).to(self.device)  # [N, D]
        N, D = X.shape
        K = self.num_components
        num_steps = 0

        if initialize:
            rand_idx = torch.randperm(N, device=self.device)[:K]
            self.mu.copy_(X[rand_idx])
            # Initialize cov to global variance + jitter
            global_cov = torch.cov(
                X.T) + self.jitter * torch.eye(D, device=self.device)
            self.Sigma.copy_(global_cov.unsqueeze(0).expand(K, -1, -1))
            self.pi.fill_(1.0 / K)

        prev_ll = -float('inf')
        for _ in range(self.max_iters):

            Sigma_safe = self.Sigma + self.jitter * torch.eye(
                D, device=self.device).unsqueeze(0)  # [K, D, D]

            # E-step: responsibilities r[n, k]
            mvn = dist.MultivariateNormal(self.mu, Sigma_safe)
            log_N = mvn.log_prob(X.unsqueeze(1))  # [N, K]
            log_pi = torch.log(self.pi.clamp_min(self.jitter))
            log_probs = log_pi + log_N  # [N, K]

            # log-sum-exp normalize
            # log_denom: [N,1]
            log_denom = torch.logsumexp(log_probs, dim=1, keepdim=True)
            r = torch.exp(log_probs - log_denom)  # [N, K]

            # M-step
            Nk = r.sum(dim=0).clamp_min(self.jitter)  # [K]
            pi_new = (Nk / N)  #[K]
            mu_new = (r.transpose(0, 1) @ X) / Nk.view(K, 1)  # [K, D]
            self.pi.copy_(pi_new)
            self.mu.copy_(mu_new)
            Xc = X.unsqueeze(1) - self.mu.unsqueeze(0)  # [N, K, D]
            sigma_new = torch.einsum('nk,nki,nkj->kij', r, Xc, Xc) / Nk.view(
                K, 1, 1)
            sigma_new = sigma_new + self.jitter * torch.eye(
                D,
                device=self.device,
            ).unsqueeze(0)

            self.Sigma.copy_(sigma_new)

            num_steps += 1

            # Check convergence by log-likelihood
            ll = log_denom.mean().item()
            if abs(ll - prev_ll) < self.abs_stop_tol:
                prev_ll = ll
                break
            prev_ll = ll
        cond = torch.linalg.cond(self.Sigma)
        if torch.any(cond > 1e6):
            logger.warning(
                "Detected ill-conditioned Sigma after GMM fit. Fixing...")
            eigvals, eigvecs = torch.linalg.eigh(self.Sigma)
            eigvals = torch.clamp(
                eigvals,
                min=eigvals.max() / 1e6,
            )
            Sigma_fixed = eigvecs @ torch.diag_embed(
                eigvals) @ eigvecs.transpose(1, 2)
            self.Sigma.copy_(Sigma_fixed)
        # After fitting, cache conditional parameters
        self._cache_conditionals()

        avg_nll = -prev_ll
        return avg_nll, num_steps

    def _cache_conditionals(self):
        """Compute and cache per-component conditional parameters."""
        A, b, cond_cov = self._conditional_params_per_component()
        self.cond_A.copy_(A)
        self.cond_b.copy_(b)
        self.cond_cov.copy_(cond_cov)

    def _conditional_params_per_component(self):
        """
        Compute per-component conditional parameters for e|z, epsilon | z ~ N(A_k z + b_k, cond_cov_k) for component k.
        Returns:
            (A, b, cond_cov) (torch.Tensor, torch.Tensor, torch.Tensor): where
                A: [K, De, Dz]
                b: [K, De]
                cond_cov: [K, De, De]
        """
        K = self.num_components
        De, Dz = self.epsilon_dim, self.z_dim
        D = De + Dz
        mu = self.mu  # [K, D]
        Sigma = self.Sigma  # [K, D, D]

        mu_e = mu[:, :De]  # [K, De]
        mu_z = mu[:, De:]  # [K, Dz]
        S_ee = Sigma[:, :De, :De]  # [K, De, De]
        S_ez = Sigma[:, :De, De:]  # [K, De, Dz]
        S_ze = Sigma[:, De:, :De]  # [K, Dz, De]
        S_zz = Sigma[:, De:, De:]  # [K, Dz, Dz]

        eye_z = torch.eye(Dz, device=self.device).expand(K, Dz, Dz)
        S_zz_reg = S_zz + self.jitter * eye_z
        S_ez_T = S_ez.transpose(1, 2)  # [K, Dz, De]
        A_T = torch.linalg.solve(S_zz_reg, S_ez_T)  # [K, Dz, De]
        A = A_T.transpose(1, 2)  # [K, De, Dz]
        # b_k = mu_e - A mu_z
        b = mu_e - torch.matmul(A, mu_z.unsqueeze(-1)).squeeze(-1)  # [K, De]
        cond_cov = S_ee - torch.matmul(A, S_ze)  # [K, De, De]

        # Ensure SPD per component
        try:
            torch.linalg.cholesky(cond_cov)
        except RuntimeError:
            logger.warning(
                "Detected not SPD conditional covariance. Fixing...")
            eigvals, eigvecs = torch.linalg.eigh(cond_cov)
            eigvals = torch.clamp(eigvals, min=self.jitter)
            cond_cov = torch.matmul(eigvecs * eigvals, eigvecs.transpose(1, 2))

        return A, b, cond_cov

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample epsilon' ~ q_psi(epsilon|z) using the conditional GMM.

        Args:
            z (torch.Tensor): [B, Dz]
            num_samples (int): samples per z
        Returns:
            (z_aux, epsilon_aux, log_prob): shapes [B, S, Dz], [B, S, De], [B, S]
        """
        assert z.dim() == 2 and z.shape[1] == self.z_dim
        B = z.shape[0]
        S = int(num_samples)
        z = z.to(self.device)

        De, Dz = self.epsilon_dim, self.z_dim
        K = self.num_components

        # 1. Compute Gating Probabilities w_k(z)
        #    log w_k \propto log pi_k + log N(z; mu_z, Sigma_zz)
        mu_z = self.mu[:, De:]  # [K, Dz]
        S_zz = self.Sigma[:, De:, De:]  # [K, Dz, Dz]

        # Add jitter for stability
        S_zz_safe = S_zz + self.jitter * torch.eye(
            Dz, device=self.device).unsqueeze(0)

        # Calculate log_prob for all z against all K components
        # mvn_z batch shape: [K], event shape: [Dz]
        mvn_z = dist.MultivariateNormal(mu_z, S_zz_safe)

        # z: [B, Dz] -> [B, 1, Dz] to broadcast against K
        log_Nz = mvn_z.log_prob(z.unsqueeze(1))  # [B, K]

        # [1, K]
        log_pi = torch.log(self.pi.clamp_min(self.jitter)).unsqueeze(0)
        log_w: torch.Tensor = log_pi + log_Nz  # [B, K]

        # Normalize to probabilities
        log_w_max = log_w.max(dim=1, keepdim=True).values
        probs = torch.exp(log_w - log_w_max)  # [B, K]
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.jitter)

        # 2. Sample Component Indices
        # We need S samples per batch item B.
        # comp_idx: [B, S]
        comp_idx = torch.multinomial(probs, num_samples=S, replacement=True)

        # Flatten indices to [B*S] for efficient indexing
        comp_idx_flat = comp_idx.view(-1)

        # 3. Compute Conditional Means for ALL components
        # mean_k(z) = A_k @ z + b_k
        # A: [K, De, Dz], z: [B, Dz]
        # We use einsum to compute all K means for all B inputs at once.
        # Result means_all: [B, K, De]
        means_all = torch.einsum(
            'kez, bz -> bke',
            self.cond_A,
            z,
        ) + self.cond_b.unsqueeze(0)

        # 4. Gather Parameters for the specific sampled components
        # We need to extract the specific mean and covariance for each of the (B*S) samples.

        # A. Gather Means
        # Expand means_all to [B, S, K, De] to match sampling structure, then flatten B*S
        means_expanded = means_all.unsqueeze(1).expand(B, S, K, De).reshape(
            B * S, K, De)

        # Gather: select the k dimension based on comp_idx_flat
        # indices must match dims: [B*S, 1, De]
        gather_idx = comp_idx_flat.view(B * S, 1, 1).expand(B * S, 1, De)
        selected_means = torch.gather(
            means_expanded,
            1,
            gather_idx,
        ).squeeze(1)  # [B*S, De]

        # B. Gather Covariances
        # cond_cov is [K, De, De]. We just index into it directly using the flat indices.
        # selected_covs: [B*S, De, De]
        selected_covs = self.cond_cov[comp_idx_flat]

        # Add tiny jitter to ensure positive definiteness for the sampler
        selected_covs = selected_covs + self.jitter * torch.eye(
            De, device=self.device).unsqueeze(0)

        # 5. Create Batched Distribution and Sample
        # This creates a batch of (B*S) independent Multivariate Normals
        mvn_conditional = dist.MultivariateNormal(
            selected_means,
            selected_covs,
        )

        # Sample (no arguments needed, shapes are in the distribution parameters)
        epsilon_flat = mvn_conditional.sample()  # [B*S, De]

        # 6. Reshape and Return
        epsilon_aux = epsilon_flat.view(B, S, De)
        z_aux = z.unsqueeze(1).expand(B, S, Dz)

        # Calculate log_prob
        # Log probability of epsilon relative to all components, weighted by component probabilities

        # log_w [B, K] -> [B, 1, K]
        log_probs_w = torch.log(probs.clamp_min(self.jitter)).unsqueeze(1)

        # Prepare means and covariances for all components
        # means_all: [B, K, De] -> [B, 1, K, De]
        means_all_expanded = means_all.unsqueeze(1)

        # cond_cov: [K, De, De]. Add jitter.
        cond_cov_safe = self.cond_cov + self.jitter * torch.eye(
            De, device=self.device).unsqueeze(0)

        # Construct Multivariate Normal for all components
        # Batch shape: [B, 1, K] due to broadcasting of means_all_expanded and cond_cov_safe
        mvn_cond_all = dist.MultivariateNormal(means_all_expanded,
                                               cond_cov_safe)

        # Evaluate epsilon_aux against all components
        # epsilon_aux: [B, S, De] -> [B, S, 1, De]
        # Resulting log_probs_components: [B, S, K]
        log_probs_components = mvn_cond_all.log_prob(epsilon_aux.unsqueeze(2))

        # Compute log_sum_exp over K components
        # log p(e|z) = log sum_k p(e|z,k) * w_k(z)
        log_prob = torch.logsumexp(log_probs_components + log_probs_w, dim=2)

        return z_aux.clone().detach(), epsilon_aux.clone().detach(
        ), log_prob.clone().detach()

    def log_prob(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_psi(epsilon|z)` for given `epsilon` and `z`.
        
        Args:
            epsilon (torch.Tensor): shape [..., De]
            z (torch.Tensor): shape [..., Dz]
        Returns:
            log_prob (torch.Tensor): shape [...]
        """
        De, Dz = self.epsilon_dim, self.z_dim
        D = De + Dz

        # p(z)
        mu_z = self.mu[:, De:]
        S_zz = self.Sigma[:, De:, De:]
        S_zz_safe = S_zz + self.jitter * torch.eye(
            Dz, device=self.device).unsqueeze(0)
        mvn_z = dist.MultivariateNormal(mu_z, S_zz_safe)
        # z: [..., Dz] -> [..., 1, Dz]
        log_Nz = mvn_z.log_prob(z.unsqueeze(-2))  # [..., K]

        log_pi = torch.log(self.pi.clamp_min(self.jitter))  # [K]
        log_p_z_joint = log_Nz + log_pi  # [..., K]
        log_p_z = torch.logsumexp(log_p_z_joint, dim=-1)  # [...]

        # p(epsilon, z)
        X = torch.cat([epsilon, z], dim=-1)  # [..., D]
        S_safe = self.Sigma + self.jitter * torch.eye(
            D, device=self.device).unsqueeze(0)
        mvn_joint = dist.MultivariateNormal(self.mu, S_safe)
        log_N_joint = mvn_joint.log_prob(X.unsqueeze(-2))  # [..., K]
        log_p_joint_k = log_N_joint + log_pi
        log_p_joint = torch.logsumexp(log_p_joint_k, dim=-1)  # [...]

        return log_p_joint - log_p_z

    def score(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score `\\nabla_epsilon log q_psi(epsilon|z)`.
        Args:
            epsilon (torch.Tensor): shape [..., De]
            z (torch.Tensor): shape [..., Dz]
        Returns:
            score (torch.Tensor): shape [..., De]
        """
        De, Dz = self.epsilon_dim, self.z_dim
        D = De + Dz

        X = torch.cat([epsilon, z], dim=-1)  # [..., D]
        S_safe = self.Sigma + self.jitter * torch.eye(
            D, device=self.device).unsqueeze(0)
        mvn_joint = dist.MultivariateNormal(self.mu, S_safe)

        # log responsibilities
        log_pi = torch.log(self.pi.clamp_min(self.jitter))
        log_N_joint = mvn_joint.log_prob(X.unsqueeze(-2))  # [..., K]
        log_unnorm_r = log_N_joint + log_pi  # [..., K]
        log_p_X = torch.logsumexp(log_unnorm_r, dim=-1, keepdim=True)
        r = torch.exp(log_unnorm_r - log_p_X)  # [..., K]

        # Gradients per component
        # -Sigma_k^{-1} (X - mu_k)
        # X: [..., D]. mu_k: [K, D]
        diff = X.unsqueeze(-2) - self.mu  # [..., K, D]
        # S_safe: [K, D, D]

        grad_k = -torch.linalg.solve(S_safe, diff)  # [..., K, D]

        # Weighted sum: sum_k r_k * grad_k
        grad = (r.unsqueeze(-1) * grad_k).sum(dim=-2)  # [..., D]

        # Extract epsilon grad
        grad_epsilon = grad[..., :De]
        return grad_epsilon


class ConditionalRealNVP(BaseReverseConditionalModel):
    """
    RealNVP for conditional density estimation. Estimates q_psi(epsilon|z) via normalizing flow. 

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - hidden_dim: Hidden size of coupling layers.
        - num_layers: Number of coupling layers.
        - logit: Whether to use logit transform to clamp epsilon to (0, 1).
        - device: Device for computation.

    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config, name="ConditionalRealNVP")
        self.num_layers: int = config.num_layers
        self.hidden_dim: int = config.hidden_dim
        self.logit: bool = config.get("logit", False)

        from models.cnf import generate_cond_real_nvp

        self.net = generate_cond_real_nvp(
            K=self.num_layers,
            hidden_size=self.hidden_dim,
            latent_size=self.epsilon_dim,
            context_size=self.z_dim,
            device=self.device,
            logit=self.logit,
        )

    def log_prob(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_psi(epsilon|z)` for given `epsilon` and `z`.

        Args:
            epsilon (torch.Tensor): Input `epsilon` of shape `[..., D_epsilon]`.
            z (torch.Tensor): Conditioning variable of shape `[..., D_z]`.
        Returns:
            log_prob (torch.Tensor): Log probability `log q_psi(epsilon|z)`.
        """
        log_prob = self.net.log_prob(epsilon, context=z)
        return log_prob

    def sample(
        self,
        z: torch.Tensor,
        num_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample `epsilon ~ q_psi(epsilon|z)` using the inverse of the flow. No gradients computed. 

        Args:
            z (torch.Tensor): Conditioning batch `[B, D_z]`.
            num_samples (int): Number of samples per `z`.
        Returns:
            (z_aux, epsilon_aux, log_prob) (torch.Tensor, torch.Tensor, torch.Tensor): 
                - tiled conditioning `z_aux` of shape `[B, num_samples, D_z]`,
                - samples `epsilon_aux` of shape `[B, num_samples, D_epsilon]`,
                - log probabilities `log_prob` of shape `[B, num_samples]`.
        """
        # Sample from base distribution
        with torch.no_grad():
            z_aux = z.clone().detach().unsqueeze(1).repeat(1, num_samples, 1)
            z_aux = z_aux.reshape(-1, self.z_dim).to(self.device)
            for i in range(3):
                # Using sample with log prob from the net
                eps_aux, log_prob = self.net.sample(
                    num_samples * z.size(0),
                    context=z_aux,
                )

                if torch.isfinite(eps_aux).all():
                    eps_aux: torch.Tensor = eps_aux.reshape(
                        -1,
                        num_samples,
                        self.epsilon_dim,
                    ).clone().detach()
                    z_aux = z_aux.reshape(
                        -1,
                        num_samples,
                        self.z_dim,
                    ).clone().detach()
                    log_prob = log_prob.reshape(
                        -1,
                        num_samples,
                    ).clone().detach()
                    return z_aux, eps_aux, log_prob
                else:
                    logger.warning(
                        f"Non-finite samples detected in RealNVP sampling attempt {i+1}. Retrying..."
                    )
            raise RuntimeError(
                "Failed to obtain finite samples from RealNVP after 3 attempts."
            )


class ConditionalGaussianReverse(BaseReverseConditionalModel):
    """
    Gaussian reverse model for q_psi(epsilon|z) via joint Gaussian. Assume the joint distribution is Gaussian.

    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config, name="ConditionalGaussianReverse")

        # Parameters of joint Gaussian over [epsilon, z]
        self.mu: torch.Tensor
        self.Sigma: torch.Tensor
        self.register_buffer(
            'mu',
            torch.zeros(self.epsilon_dim + self.z_dim, dtype=torch.float32),
        )
        self.register_buffer(
            'Sigma',
            torch.eye(self.epsilon_dim + self.z_dim, dtype=torch.float32),
        )

    @torch.no_grad()
    def fit(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
        initialize: bool = False,
    ) -> tuple[float, int]:
        """
        Fit joint Gaussian parameters from provided samples.
        epsilon: [..., De], z: [..., Dz]

        Args:
            epsilon (torch.Tensor): Samples of epsilon with shape [..., De].
            z (torch.Tensor): Samples of z with shape [..., Dz].
            initialize (bool): If True, reset parameters before fitting. Not used here.
        Returns:
            (avg_nll, num_steps) (float, int): Average negative log-likelihood and number of steps (always 1 here).
        """
        assert epsilon.shape[-1] == self.epsilon_dim and z.shape[
            -1] == self.z_dim, "Dimension mismatch in fit inputs."
        epsilon = epsilon.clone().detach().reshape(-1, self.epsilon_dim)
        z = z.clone().detach().reshape(-1, self.z_dim)
        # Concatenate samples
        X = torch.cat([epsilon, z], dim=1)  # [N, z_dim + epsilon_dim]
        mu = X.mean(dim=0)  # (z_dim + epsilon_dim,)
        Xc = X - mu
        # Covariance with variables in columns: Sigma = (Xc^T Xc)/(N-1)
        N = X.shape[0]
        Sigma = (Xc.t() @ Xc) / max(1, N - 1)
        Sigma = Sigma + self.jitter * torch.eye(
            self.epsilon_dim + self.z_dim,
            device=self.device,
            dtype=Sigma.dtype,
        )
        self.mu.copy_(mu)
        self.Sigma.copy_(Sigma)
        # Compute average negative log-likelihood
        mvn = dist.MultivariateNormal(self.mu, self.Sigma)
        log_probs = mvn.log_prob(X)
        avg_nll = -log_probs.mean().item()
        return avg_nll, 1

    def _conditional_params(
            self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute A, b, cond_cov for epsilon|z from stored mu/Sigma.
        Returns:
            (A, b, cond_cov) (torch.Tensor, torch.Tensor, torch.Tensor): where epsilon|z ~ N(Az + b, cond_cov)
        """
        mu = self.mu
        Sigma = self.Sigma
        mu_e = mu[:self.epsilon_dim]
        mu_z = mu[self.epsilon_dim:]
        S_ee = Sigma[:self.epsilon_dim, :self.epsilon_dim]
        S_ez = Sigma[:self.epsilon_dim, self.epsilon_dim:]
        S_ze = Sigma[self.epsilon_dim:, :self.epsilon_dim]
        S_zz = Sigma[self.epsilon_dim:, self.epsilon_dim:]
        # Regularized inverse for stability
        S_zz_reg = S_zz + self.jitter * torch.eye(
            self.z_dim,
            device=self.device,
            dtype=Sigma.dtype,
        )
        S_zz_inv = torch.linalg.inv(S_zz_reg)
        A = S_ez @ S_zz_inv
        b = mu_e - A @ mu_z
        cond_cov = S_ee - A @ S_ze
        # Ensure SPD
        try:
            torch.linalg.cholesky(cond_cov)
        except RuntimeError:
            logger.warning(
                "Detected not SPD conditional covariance. Fixing...")
            eigvals, eigvecs = torch.linalg.eigh(cond_cov)
            eigvals = torch.clamp(eigvals, min=1e-10)
            cond_cov = (eigvecs * eigvals) @ eigvecs.transpose(0, 1)
        return A, b, cond_cov

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample epsilon' ~ q_psi(epsilon|z) using conditional Gaussian.

        Args:
            z (torch.Tensor): [B, Dz]
            num_samples (int): samples per z
        Returns:
            (z_aux, epsilon_aux, log_prob): shapes [B, S, Dz], [B, S, De], [B, S]
        """
        assert z.dim() == 2 and z.shape[1] == self.z_dim
        B = z.shape[0]
        S = num_samples
        z_aux = z.clone().detach().to(self.device).unsqueeze(1).repeat(1, S, 1)

        A, b, cond_cov = self._conditional_params()
        # Means: [B,S,De] = (z_aux @ A^T) + b
        mean = torch.matmul(z_aux, A.transpose(0, 1)) + b.view(1, 1, -1)
        # Cholesky of cond_cov
        L = torch.linalg.cholesky(cond_cov)
        noise = torch.randn(B, S, self.epsilon_dim, device=self.device)
        eps = mean + torch.matmul(noise, L.transpose(0, 1))

        # Calculate log_prob
        mvn = dist.MultivariateNormal(mean, cond_cov)
        log_prob = mvn.log_prob(eps)

        return z_aux.clone().detach(), eps.clone().detach(), log_prob.clone(
        ).detach()

    def log_prob(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_psi(epsilon|z)` for given `epsilon` and `z`.
        Args:
            epsilon (torch.Tensor): shape [..., De]
            z (torch.Tensor): shape [..., Dz]
        Returns:
            log_prob (torch.Tensor): shape [...]
        """
        A, b, cond_cov = self._conditional_params()
        # mean = z @ A.T + b
        # z: [..., Dz]
        mean = torch.matmul(z, A.transpose(0, 1)) + b

        mvn = dist.MultivariateNormal(mean, cond_cov)
        return mvn.log_prob(epsilon)

    def score(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score `\\nabla_epsilon log q_psi(epsilon|z)`.
        Args:
            epsilon (torch.Tensor): shape [..., De]
            z (torch.Tensor): shape [..., Dz]
        Returns:
            score (torch.Tensor): shape [..., De]
        """
        A, b, cond_cov = self._conditional_params()

        # Broadcast logic handled by matmul
        mean = torch.matmul(z, A.transpose(0, 1)) + b

        diff = epsilon - mean
        # Solve cond_cov * x = diff.
        # cond_cov: [De, De]. diff: [..., De] -> [..., De, 1]
        score = -torch.linalg.solve(cond_cov, diff.unsqueeze(-1)).squeeze(-1)
        return score


ReverseModel: dict[str, type[BaseReverseConditionalModel]] = {
    "ConditionalMixtureOfGaussianReverse": ConditionalMixtureOfGaussianReverse,
    "ConditionalRealNVP": ConditionalRealNVP,
    "ConditionalGaussianReverse": ConditionalGaussianReverse,
}
