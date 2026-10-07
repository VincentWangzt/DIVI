import torch
import time
from omegaconf import DictConfig
from runner.base_runner import BaseSIVIRunner
from utils.kernels import Kernels
from utils.annealing import annealing
from utils.logging import get_logger
from models.data_bound_target import DataBoundTarget

logger = get_logger()


class KPGRunner(BaseSIVIRunner):
    """
    Kernel Pathwise Gradient (KPG) runner for Semi-Implicit Variational Inference.

    Uses a kernel-smoothed score difference multiplied element-wise by the
    differentiable sample z to form a pathwise gradient estimator:

        loss = E_{z ~ q_phi} [ (K @ score_gap) * z ].sum(dim=1).mean()

    where:
        K[i,j] = k(z_i, z_j^aux)      — kernel matrix (both inputs detached)
        score_gap = -neg_score_implicit^aux - target_score^aux
                  = nabla_z log q(z^aux | eps^aux) - nabla_z log p(z^aux)

    The gradient nabla_phi loss flows only through z in the element-wise
    multiplication, via the reparameterization trick.  The auxiliary batch
    and kernel are fully detached.

    Unlike KSIVI (which minimises KSD^2 — a quadratic function of score
    differences), KPG forms a *linear* objective whose pathwise gradient has
    provably lower variance under bounded-gradient assumptions.

    Config keys under `train.kpg`:
        kernel (str): Kernel type.  One of 'gaussian', 'imq', 'laplace',
                      'riesz'.  Default: 'gaussian'.
        detach_kernel (bool): If True, detach z from computation graph when
                              computing kernel matrix.  Default: True.
        log_p_reg (float): Coefficient for optional log p(z) regularisation.
                           If > 0, subtracts log_p_reg * E[log p(z)] from loss.
                           Default: 0.0.
        log_p_reg_mode (str): 'warmup_only' or 'always'.  Default: 'warmup_only'.

    Reference:
        Pielok et al., "Kernel Semi-Implicit Variational Inference with
        Pathwise Gradients", 2025.  arXiv:2506.05088.
    """

    def __init__(self, config: DictConfig, name: str = "KPG"):
        super().__init__(config=config, name=name)

        # Parse KPG-specific config
        kpg_cfg = self.training_cfg.get('kpg', {})

        kernel_type: str = kpg_cfg.get('kernel', 'gaussian')
        assert kernel_type in Kernels, \
            f"kernel must be one of {list(Kernels.keys())}, got '{kernel_type}'"
        self.kernel = Kernels[kernel_type]()
        self.kernel_type = kernel_type

        self.detach_kernel: bool = kpg_cfg.get('detach_kernel', True)
        self.log_p_reg: float = kpg_cfg.get('log_p_reg', 0.0)
        self.log_p_reg_mode: str = kpg_cfg.get('log_p_reg_mode', 'warmup_only')
        assert self.log_p_reg_mode in ('warmup_only', 'always'), \
            "log_p_reg_mode must be one of ('warmup_only', 'always')"

        self.pretrain_cfg = self.training_cfg.get('pretrain', {})
        self.pretrain_enabled: bool = self.pretrain_cfg.get('enabled', False)
        self.pretrain_steps: int = int(self.pretrain_cfg.get('steps', 0))
        self.pretrain_lr: float = float(
            self.pretrain_cfg.get('lr', self.vi_lr))
        self.pretrain_batch_size: int = int(
            self.pretrain_cfg.get('batch_size', self.training_batch_size))

        # KPG has no reverse model
        self.reverse_train = False

        logger.info(
            f"KPGRunner initialized: kernel={kernel_type}, "
            f"detach_kernel={self.detach_kernel}, "
            f"log_p_reg={self.log_p_reg}, "
            f"log_p_reg_mode={self.log_p_reg_mode}"
        )

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Not used by KPG. Raises if accidentally called."""
        raise NotImplementedError(
            "KPG does not estimate log q(z). "
            "This method should not be called in the KPG training loop."
        )

    def train_reverse_model(self, epoch_outer: int):
        """No-op: KPG has no reverse model."""
        pass

    def pretrain_vi(self):
        """Optional pretraining on dev data (for BNN targets)."""
        if (not self.pretrain_enabled or self.pretrain_steps <= 0 or
                not isinstance(self.target_model, DataBoundTarget) or
                self.target_model.dev_data is None or
                not hasattr(self.target_model.inner, 'predict_y')):
            return

        X_dev, y_dev, mean_y, std_y = self.target_model.dev_data
        optimizer = torch.optim.Adam(
            self.vi_model.parameters(),
            lr=self.pretrain_lr,
            betas=self.vi_opt_betas,
        )
        log_freq = max(1, self.pretrain_steps // 10)
        logger.info(
            "Starting KPG VI pretraining on dev split: "
            f"steps={self.pretrain_steps}, lr={self.pretrain_lr}"
        )
        self.vi_model.train()
        for step in range(1, self.pretrain_steps + 1):
            epsilon = self.vi_model.sample_epsilon(num=self.pretrain_batch_size)
            z, _ = self.vi_model.forward(epsilon)
            pred_y = self.target_model.inner.predict_y(
                z, X_dev, mean_y, std_y)
            loss = ((pred_y.mean(0) - y_dev)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.vi_model.parameters(), max_norm=self.grad_clip)
            optimizer.step()
            if step % log_freq == 0 or step == self.pretrain_steps:
                self.writer.add_scalar(
                    "pretrain/vi_model/loss", loss.item(), step)
                logger.info(
                    f"KPG pretrain step {step}/{self.pretrain_steps}: "
                    f"loss={loss.item():.6f}"
                )

        if self.ema_enabled:
            from utils.ema import EMA
            self.ema = EMA(
                beta=self.ema_beta,
                model_params=self.vi_model.parameters(),
            )

    def _compute_loss_and_step(self, epoch: int) -> dict:
        """
        Compute the KPG loss and perform an optimizer step.

        Uses two independent batches (V-statistic style):
            - Primary batch z1 (differentiable — gradient flows through here)
            - Auxiliary batch z2_aux (detached — provides score gap)

        The loss is:
            loss = mean_i [ sum_d ( (K @ score_gap)[i,d] * z1[i,d] ) ]

        where K[i,j] = k(z1_i, z2_j^aux), and
              score_gap_j = -neg_score_j^aux - target_score_j^aux.

        Returns:
            dict with diagnostic keys matching BaseSIVIRunner interface.
        """
        t_vi0 = time.perf_counter()

        # Primary batch — differentiable path
        eps1 = self.vi_model.sample_epsilon(num=self.training_batch_size)
        z1, neg_score1 = self.vi_model.forward(eps1)

        # Auxiliary batch — fully detached
        with torch.no_grad():
            eps2 = self.vi_model.sample_epsilon(num=self.training_batch_size)
            z2_aux, neg_score2_aux = self.vi_model.forward(eps2)

        # Target score on auxiliary samples
        if hasattr(self.target_model, 'sample_batch') and hasattr(
                self.target_model, 'score_on_batch'):
            batch_data, batch_labels = self.target_model.sample_batch()
            target_score_aux = self.target_model.score_on_batch(
                z2_aux, batch_data, batch_labels)
        else:
            target_score_aux = self.target_model.score(z2_aux)

        # Apply annealing
        anneal_factor = annealing(
            t=epoch,
            warm_up_interval=self.anneal_steps,
            anneal=self.use_annealing,
            scheme=self.anneal_scheme,
        )

        t_vi1 = time.perf_counter()

        # Kernel matrix — both arguments detached from VI graph
        t_ns0 = time.perf_counter()

        K = self.kernel.pair_eval(
            z1.detach(),
            z2_aux,
            fit_h=True,
            detach_h=self.detach_kernel,
        )

        # Score gap: nabla_z log q(z|eps) - nabla_z log p(z)
        # neg_score_implicit = eps/std = -nabla_z log q(z|eps)
        # target_score = nabla_z log p(z)
        # So: -neg_score_implicit - target_score = nabla_z log q - nabla_z log p
        score_gap = -neg_score2_aux - anneal_factor * target_score_aux  # [N, D]

        # KPG loss: kernel-smoothed score gap, multiplied by z1 (pathwise gradient)
        # K: [N, N],  score_gap: [N, D]  =>  K @ score_gap: [N, D]
        # Element-wise multiplication with z1: [N, D]
        loss = (K.matmul(score_gap) * z1).sum(dim=1).mean()

        # Optional log-p regularisation
        apply_log_p_reg = (
            self.log_p_reg > 0 and (
                self.log_p_reg_mode == 'always' or anneal_factor < 1.0
            )
        )
        if apply_log_p_reg:
            if hasattr(self.target_model, 'logp_on_batch') and 'batch_data' in locals():
                log_p = self.target_model.logp_on_batch(
                    z1, batch_data, batch_labels)
            else:
                log_p = self.target_model.logp(z1)
            reg_scale = anneal_factor if self.log_p_reg_mode == 'warmup_only' else 1.0
            loss = loss - self.log_p_reg * log_p.mean() * reg_scale

        t_ns1 = time.perf_counter()

        # Optimizer step
        t_bw0 = time.perf_counter()

        grad_norm = None

        if torch.isfinite(loss):
            self.optimizer_vi.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.vi_model.parameters(), max_norm=self.grad_clip)
            else:
                grad_norm = torch.nn.utils.get_total_norm(
                    self.vi_model.parameters())
            self.optimizer_vi.step()
            self.scheduler_vi.step()
            if self.ema_enabled:
                self.ema.update_params(self.vi_model.parameters())
        else:
            logger.warning(
                f"NaN or Inf detected in KPG loss at epoch {epoch}. "
                f"Skipping update."
            )

        t_bw1 = time.perf_counter()

        # Log KPG-specific diagnostics
        self.writer.add_scalar(
            "kpg/kernel_bandwidth", self.kernel.h, epoch)
        self.writer.add_scalar(
            "kpg/anneal_factor", anneal_factor, epoch)

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'z': z1,
            'epsilon': eps1,
            'time_vi_sample': t_vi1 - t_vi0,
            'time_neg_score': t_ns1 - t_ns0,
            'time_backward': t_bw1 - t_bw0,
        }
