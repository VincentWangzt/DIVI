import torch
from omegaconf import DictConfig, OmegaConf
from runner.base_runner import BaseSIVIRunner
from utils.logging import get_logger

logger = get_logger()


class SIVIRunner(BaseSIVIRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "SIVI",
    ):
        super().__init__(config=config, name=name)
        self.reverse_model_type: str = 'prior q(epsilon)'
        self.training_reverse_sample_num = self.training_cfg.reverse_sample_num

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Estimate log q_phi(z).
        ```
        log q_phi(z) ~ log E_{epsilon^prime~q(epsilon)}[q_phi(z|epsilon^prime)]
        ```

        Args:
            z (torch.Tensor): Samples from q_phi(z|epsilon), shape (batch_size, z_dim).
            epsilon (torch.Tensor): Corresponding epsilon samples, shape (batch_size, epsilon_dim).
        
        Returns:
            log_q_phi_z (torch.Tensor): Estimated log q_phi(z), shape (batch_size,).
        '''

        # shape (training_reverse_sample_num, epsilon_dim)
        epsilon_new = self.vi_model.sample_epsilon(
            num=self.training_reverse_sample_num)

        # shape (batch_size, training_reverse_sample_num, z_dim)
        epsilon_aux = epsilon_new.repeat(z.shape[0], 1, 1)

        # shape (batch_size, training_reverse_sample_num + 1, z_dim)
        epsilon_aux = torch.cat([epsilon_aux, epsilon.unsqueeze(1)], dim=1)

        # shape (batch_size, training_reverse_sample_num + 1, z_dim)
        z_aux = z.unsqueeze(1).repeat(
            1,
            self.training_reverse_sample_num + 1,
            1,
        )

        # shape (batch_size, training_reverse_sample_num + 1)
        log_q_phi_z_given_epsilon = self.vi_model.logp(z_aux, epsilon_aux)

        # shape (batch_size,)
        log_q_phi_z = torch.logsumexp(
            log_q_phi_z_given_epsilon,
            dim=1,
        ) - torch.log(
            torch.tensor(
                self.training_reverse_sample_num + 1,
                device=z.device,
            ))

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

        return log_q_phi_z  # shape (batch_size,)
