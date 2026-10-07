import math
import time

import torch
from omegaconf import DictConfig

from models.vi_model import RealNVP
from runner.base_runner import BaseSIVIRunner
from utils.annealing import annealing
from utils.logging import get_logger


logger = get_logger()


class NFVIRunner(BaseSIVIRunner):
    """Direct normalizing-flow variational inference with exact log density."""

    def __init__(self, config: DictConfig, name: str = "NFVI") -> None:
        super().__init__(config=config, name=name)
        if not isinstance(self.vi_model, RealNVP):
            raise TypeError("NFVIRunner requires vi_model_type=RealNVP")

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        del epsilon
        return self.vi_model.logp(z)

    def _compute_loss_and_step(self, epoch: int) -> dict:
        """Optimize the exact reverse-KL objective for the flow."""
        t_vi0 = time.perf_counter()
        epsilon = self.vi_model.sample_epsilon(num=self.training_batch_size)
        z, log_q = self.vi_model.forward_and_log_prob(epsilon)
        t_vi1 = time.perf_counter()

        t_density0 = time.perf_counter()
        log_p = self.target_model.logp(z)
        anneal_factor = annealing(
            t=epoch,
            warm_up_interval=self.anneal_steps,
            anneal=self.use_annealing,
            scheme=self.anneal_scheme,
        )
        loss = torch.mean(log_q - anneal_factor * log_p)
        t_density1 = time.perf_counter()

        t_bw0 = time.perf_counter()
        grad_norm = None
        if torch.isfinite(loss):
            self.optimizer_vi.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.vi_model.parameters(),
                    max_norm=self.grad_clip,
                )
            else:
                grad_norm = torch.nn.utils.get_total_norm([
                    parameter.grad
                    for parameter in self.vi_model.parameters()
                    if parameter.grad is not None
                ])
            self.optimizer_vi.step()
            self.scheduler_vi.step()
            if self.ema_enabled:
                self.ema.update_params(self.vi_model.parameters())
        else:
            logger.warning(
                "NaN or Inf detected in NFVI loss at epoch %s; skipping update.",
                epoch,
            )
        t_bw1 = time.perf_counter()

        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "z": z,
            "epsilon": epsilon,
            "score_q": None,
            "time_vi_sample": t_vi1 - t_vi0,
            "time_neg_score": t_density1 - t_density0,
            "time_backward": t_bw1 - t_bw0,
        }

    def evaluate_elbo(self) -> tuple[float, float, float, float]:
        """Estimate ELBO using the flow's exact marginal log density."""
        _, z, log_q = self.vi_model.sampling_with_log_prob(
            num=self.n_elbo_z_samples
        )
        log_p = self.target_model.logp(z)
        elbo_per_sample = log_p - log_q
        elbo_mean = elbo_per_sample.mean()
        elbo_std_total = elbo_per_sample.std()
        elbo_ci_half = (
            1.96 * elbo_std_total / math.sqrt(self.n_elbo_z_samples)
        )
        return (
            elbo_mean.item(),
            elbo_std_total.item(),
            0.0,
            elbo_ci_half.item(),
        )
