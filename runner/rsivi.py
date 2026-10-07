import torch
from omegaconf import DictConfig
from runner.base_reverse_runner import BaseReverseConditionalRunner
from utils.metrics import compute_ksd
from utils.kernels import GaussianKernel
from utils.logging import get_logger

logger = get_logger()


class RSIVIRunner(BaseReverseConditionalRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "RSIVI",
    ):
        super().__init__(config=config, name=name)

    def _estimate_reverse_score(
        self,
        z_aux: torch.Tensor,
        epsilon_aux: torch.Tensor,
        context: str,
    ) -> torch.Tensor | None:
        score_samples = self.vi_model.score(z_aux, epsilon_aux)
        finite_sample_mask = torch.isfinite(score_samples).all(dim=-1)
        valid_counts = finite_sample_mask.sum(dim=1)
        fully_invalid_rows = valid_counts == 0

        if fully_invalid_rows.any():
            logger.warning(
                f"{fully_invalid_rows.sum()} samples had no finite reverse score samples during {context} at epoch {self.curr_epoch}. Skipping score computation."
            )
            return None

        safe_score_samples = torch.where(
            finite_sample_mask.unsqueeze(-1),
            score_samples,
            torch.zeros_like(score_samples),
        )
        valid_counts = valid_counts.to(
            device=safe_score_samples.device,
            dtype=safe_score_samples.dtype,
        ).unsqueeze(-1)
        score = safe_score_samples.sum(dim=1) / valid_counts
        return score.clone().detach()

    def calculate_rev_KSD(self) -> tuple[float, float]:
        '''
        Calculate the Kernelized Stein Discrepancy (KSD) using the reverse denoising model.

        Returns:
            (ksd, h) (float, float): Estimated KSD value and kernel bandwidth.
        '''
        with torch.no_grad():
            _, z_samples = self.vi_model.sampling(num=self.n_ksd_samples)
            self.reverse_ksd_kernel = GaussianKernel()
            h = self.reverse_ksd_kernel.fit_h(z_samples)

            z_aux, epsilon_aux, _ = self.reverse_model.sample(
                z_samples,
                num_samples=self.training_reverse_sample_num,
            )
            score = self._estimate_reverse_score(
                z_aux,
                epsilon_aux,
                context="reverse KSD evaluation",
            )
            if score is None:
                return float("nan"), h
            if self.normalize_reverse_score:
                score = score - score.mean(dim=0, keepdim=True)

        ksd = compute_ksd(
            z_samples,
            scores=score,
            kernel=self.reverse_ksd_kernel,
        )
        return ksd, h

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

        Args:
            z (torch.Tensor): Samples from q_phi(z|epsilon), shape (batch_size, z_dim).
            epsilon (torch.Tensor): Corresponding epsilon samples, shape (batch_size, epsilon_dim).
        
        Returns:
            log_q_phi_z (torch.Tensor): Estimated log q_phi(z), shape (batch_size,).
        '''
        with torch.no_grad():
            self.reverse_model.eval()
            z_aux, epsilon_aux, _ = self.reverse_model.sample(
                z,
                num_samples=self.training_reverse_sample_num,
            )

        with torch.no_grad():
            score = self._estimate_reverse_score(
                z_aux,
                epsilon_aux,
                context="RSIVI score estimation",
            )
            if score is None:
                return torch.full(
                    (z.shape[0],),
                    float('nan'),
                    device=z.device,
                    dtype=z.dtype,
                )
            self.log_reverse_score_l2_to_target(score, z)

            if self.normalize_reverse_score:
                score = score - score.mean(dim=0, keepdim=True)

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

            # Log the average norm of the score function
            avg_score_norm = torch.mean(torch.norm(score, dim=-1)).item()
            self.writer.add_scalar(
                "diagnostic/reverse_model/avg_score_norm",
                avg_score_norm,
                self.curr_epoch,
            )

            # Log the norm of the average of the score function
            avg_of_score_norm = torch.norm(score.mean(dim=0)).item()
            self.writer.add_scalar(
                "diagnostic/reverse_model/norm_of_avg_score",
                avg_of_score_norm,
                self.curr_epoch,
            )

        log_q_phi_z = torch.sum(score * z, dim=-1)  # shape (batch_size,)
        return log_q_phi_z, score
