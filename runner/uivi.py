import torch
from omegaconf import DictConfig, OmegaConf
from runner.base_runner import BaseSIVIRunner
from utils.logging import get_logger
from utils.metrics import compute_ksd
from utils.kernels import GaussianKernel

logger = get_logger()


class UIVIRunner(BaseSIVIRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "UIVI",
    ):
        super().__init__(config=config, name=name)
        # HMC config (for UIVI)
        self.reverse_model_type: str = "HMC"

        if 'reverse_model_config_path' not in self.config:
            default_reverse_model_config_path = f'configs/reverse_models/{self.reverse_model_type}.yaml'
            logger.warning(
                f"'reverse_model_config_path' not found in main_config; using default: {default_reverse_model_config_path}"
            )
            self.config.reverse_model_config_path = default_reverse_model_config_path
        reverse_model_config_path: str = self.config.reverse_model_config_path
        logger.info(
            f"Using reverse model config path: {reverse_model_config_path}")
        _reverse_model_config = {
            'hmc': OmegaConf.load(reverse_model_config_path)
        }
        self.config = OmegaConf.merge(
            _reverse_model_config,
            self.config,
        )  # type: ignore

        self.hmc_step_size = self.config.hmc.step_size
        self.hmc_leapfrog_steps = self.config.hmc.leapfrog_steps
        self.hmc_burn_in_steps = self.config.hmc.burn_in_steps
        self.training_reverse_sample_num = self.training_cfg.reverse_sample_num

    def _log_q_phi_eps_given_z(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q_phi(epsilon | z) = log q(epsilon) + log q_phi(z | epsilon) - const, where the const = log q_phi(z) is ignored.
        Supports broadcasting over leading dimensions.
        Args:
            epsilon (torch.Tensor): shape [..., De]
            z (torch.Tensor): shape [..., Dz]
        Returns:
            log_q_phi (torch.Tensor): shape [...]
        """
        return self.vi_model.log_q_epsilon(epsilon) + self.vi_model.logp(
            z, epsilon)

    def _grad_log_q_phi(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gradient of log q_phi(epsilon | z) w.r.t. epsilon.
        Args:
            epsilon (torch.Tensor): shape [B, De]
            z (torch.Tensor): shape [B, Dz]
        Returns:
            grad (torch.Tensor): shape [B, De]
        """
        epsilon = epsilon.clone().detach().requires_grad_(True)
        logp = self._log_q_phi_eps_given_z(epsilon, z)
        logp_sum = logp.sum()
        grad = torch.autograd.grad(logp_sum,
                                   epsilon,
                                   retain_graph=False,
                                   create_graph=False)[0]
        return grad.detach()

    def sample_epsilon_hmc(
        self,
        z: torch.Tensor,
        eps_init: torch.Tensor,
        num_samples: int,
        burn_in_steps: int,
        step_size: float,
        leapfrog_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Sequential HMC per (z, epsilon) starting from eps_init.
        Runs `burn_in_steps` transitions, then collects `num_samples` samples.
        Returns tiled z, sampled epsilon, and average acceptance rate.
        
        Args:
            z (torch.Tensor): shape [B, Dz]
            eps_init (torch.Tensor): shape [B, De]
            num_samples (int): number of samples to collect after burn-in
            burn_in_steps (int): number of burn-in HMC steps
            step_size (float): leapfrog step size
            leapfrog_steps (int): number of leapfrog steps per HMC transition
        Returns:
            (z_aux, eps_aux, acc_rate) (torch.Tensor, torch.Tensor, float):
            z_aux has shape [B, S, Dz] (tiled z),
            eps_aux has shape [B, S, De] (sampled epsilons),
            acc_rate is the average acceptance rate (float).
        """
        B, Dz = z.shape
        De = self.epsilon_dim
        S = num_samples
        device = self.device

        z_aux = z.unsqueeze(1).expand(B, S, Dz).clone().detach()
        eps_current = eps_init.clone().detach().to(device)  # [B, De]

        samples = []
        accepts = []

        total_steps = burn_in_steps + S
        for step in range(total_steps):
            # Resample momentum
            p0 = torch.randn(B, De, device=device)

            # Current energies
            logp0 = self._log_q_phi_eps_given_z(eps_current, z)
            K0 = 0.5 * (p0**2).sum(dim=-1)

            # Leapfrog from current state (batched over B)
            p = p0 + 0.5 * step_size * self._grad_log_q_phi(eps_current, z)
            eps_prop = eps_current
            for t in range(leapfrog_steps):
                eps_prop = eps_prop + step_size * p
                grad = self._grad_log_q_phi(eps_prop, z)
                if t != leapfrog_steps - 1:
                    p = p + step_size * grad
            p = p + 0.5 * step_size * grad

            # Proposed energies
            logp_prop = self._log_q_phi_eps_given_z(eps_prop, z)
            K_prop = 0.5 * (p**2).sum(dim=-1)

            dH = (K_prop - logp_prop) - (K0 - logp0)
            accept_prob = torch.exp((-dH).clamp(max=0))
            u = torch.rand_like(accept_prob)
            accept_mask = (u < accept_prob)

            # Update current
            eps_current = torch.where(accept_mask.unsqueeze(-1), eps_prop,
                                      eps_current)

            accepts.append(accept_mask.float().clone().detach())
            # After burn-in, record sample
            if step >= burn_in_steps:
                samples.append(eps_current.clone().detach())

        # Stack samples: list of S tensors [B, De] -> [S, B, De] -> [B, S, De]
        eps_out = torch.stack(samples, dim=0).transpose(0, 1)
        acc_rate = torch.stack(accepts, dim=0).mean().item()
        return z_aux.detach(), eps_out.detach(), acc_rate

    def calculate_rev_KSD(self) -> tuple[float, float]:
        '''
        Calculate the Kernelized Stein Discrepancy (KSD) using the reverse denoising model.

        Returns:
            ksd (float): The estimated KSD.
            h (float): The bandwidth used in the kernel.
        '''
        with torch.no_grad():
            epsilon_samples, z_samples = self.vi_model.sampling(
                num=self.n_ksd_samples)
            self.reverse_ksd_kernel = GaussianKernel()
            h = self.reverse_ksd_kernel.fit_h(z_samples)

        z_aux, epsilon_aux, acc_rate = self.sample_epsilon_hmc(
            z_samples,
            eps_init=epsilon_samples,
            num_samples=self.training_reverse_sample_num,
            burn_in_steps=self.hmc_burn_in_steps,
            step_size=self.hmc_step_size,
            leapfrog_steps=self.hmc_leapfrog_steps,
        )
        score = self.vi_model.score(z_aux, epsilon_aux)
        score = score.mean(dim=1)
        score = score.clone().detach()

        rev_ksd = compute_ksd(
            z_samples,
            scores=score,
            kernel=self.reverse_ksd_kernel,
        )

        return rev_ksd, h

    def eval_ksd(self, epoch: int):
        super().eval_ksd(epoch)
        rev_ksd, rev_h = self.calculate_rev_KSD()
        self.writer.add_scalar("metric/reverse_model/ksd", rev_ksd, epoch)
        self.writer.add_scalar("metric/reverse_model/ksd_h", rev_h, epoch)

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Estimate log q_phi(z) via the gradient.
        ```
        nabla_z log q_phi(z) ~ E_{epsilon^prime ~ q_psi(epsilon|z)} [nabla_z log q_phi(z|epsilon^prime)]
        ```
        here q_psi is approximated via HMC targeting q_phi(z|epsilon) * q(epsilon).

        Args:
            z (torch.Tensor): Samples from q_phi(z|epsilon), shape (batch_size, z_dim).
            epsilon (torch.Tensor): Corresponding epsilon samples, shape (batch_size, epsilon_dim).
        
        Returns:
            log_q_phi_z (torch.Tensor): Estimated log q_phi(z), shape (batch_size,).
        '''
        z_aux, epsilon_aux, acc_rate = self.sample_epsilon_hmc(
            z,
            eps_init=epsilon,
            num_samples=self.training_reverse_sample_num,
            burn_in_steps=self.hmc_burn_in_steps,
            step_size=self.hmc_step_size,
            leapfrog_steps=self.hmc_leapfrog_steps,
        )
        self.writer.add_scalar(
            "train/reverse_model/hmc_accept_rate",
            acc_rate,
            self.curr_epoch,
        )

        with torch.no_grad():
            score = self.vi_model.score(z_aux, epsilon_aux)
            score = score.mean(dim=1)
            score = score.clone().detach()
            self.log_reverse_score_l2_to_target(score, z)

            # Log the average distance from epsilon_aux to original epsilon
            avg_eps_distance = torch.mean(
                torch.norm(
                    epsilon_aux - epsilon.unsqueeze(1),
                    dim=-1,
                )).item()
            self.writer.add_scalar(
                "diagnostic/reverse_model/avg_epsilon_distance",
                avg_eps_distance,
                self.curr_epoch,
            )

        log_q_phi_z = torch.sum(score * z, dim=-1)  # shape (batch_size,)
        return log_q_phi_z, score
