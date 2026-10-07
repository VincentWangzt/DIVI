import torch
import time
from omegaconf import DictConfig
from runner.base_runner import BaseSIVIRunner
from utils.kernels import Kernels
from utils.annealing import annealing
from utils.logging import get_logger
from models.data_bound_target import DataBoundTarget

logger = get_logger()


class KSIVIRunner(BaseSIVIRunner):
    """
    Kernel Semi-Implicit Variational Inference (KSIVI) runner.

    Minimizes the squared Kernel Stein Discrepancy (KSD²) between the
    variational approximation q_phi and the target distribution p:

        KSD²(q_phi || p) = E_{q(x,z), q(x',z')} [
            k(x,x') * <s_p(x) + s_q(x|z),  s_p(x') + s_q(x'|z')>
        ]

    where k is a kernel function (Gaussian, IMQ, Laplace, or Riesz),
    s_p = ∇ log p is the target score, and s_q(x|z) = -ε/σ is the implicit
    score from the reparameterization trick.

    Unlike ELBO-based SIVI variants, KSIVI does not require estimating the
    intractable marginal density q(z), nor does it need a reverse model.

    Config keys under `train.ksivi`:
        statistic (str): 'v' for V-statistic (two independent batches, unbiased)
                         or 'u' for U-statistic (single batch, diagonal zeroed).
                         Default: 'v'.
        kernel (str): Kernel type. One of 'gaussian', 'imq', 'laplace', 'riesz'.
                      Default: 'gaussian'.
        detach_kernel (bool): If True, detach z from computation graph when
                              computing kernel matrix. Default: True.
        detach_bandwidth (bool): Detach the fitted median bandwidth while
                                 retaining kernel gradients through z when
                                 detach_kernel=False. Default: False.
        log_p_reg (float): Coefficient for optional log p(z) regularization.
                           If > 0, subtracts log_p_reg * E[log p(z)] from loss.
                           Default: 0.0.

    Reference:
        Cheng et al., "Kernel Semi-Implicit Variational Inference", ICML 2024.
    """

    def __init__(self, config: DictConfig, name: str = "KSIVI"):
        super().__init__(config=config, name=name)

        # Parse KSIVI-specific config
        ksivi_cfg = self.training_cfg.get('ksivi', {})
        self.statistic_type: str = ksivi_cfg.get('statistic', 'v')
        assert self.statistic_type in ('v', 'u'), \
            f"statistic must be 'v' or 'u', got '{self.statistic_type}'"

        kernel_type: str = ksivi_cfg.get('kernel', 'gaussian')
        assert kernel_type in Kernels, \
            f"kernel must be one of {list(Kernels.keys())}, got '{kernel_type}'"
        self.kernel = Kernels[kernel_type]()
        self.kernel_type = kernel_type

        self.detach_kernel: bool = ksivi_cfg.get('detach_kernel', True)
        self.detach_bandwidth: bool = ksivi_cfg.get('detach_bandwidth', False)
        self.log_p_reg: float = ksivi_cfg.get('log_p_reg', 0.0)
        self.log_p_reg_mode: str = ksivi_cfg.get(
            'log_p_reg_mode', 'warmup_only')
        self.affine_invariant: bool = ksivi_cfg.get('affine_invariant', False)
        assert self.log_p_reg_mode in ('warmup_only', 'always'), \
            "log_p_reg_mode must be one of ('warmup_only', 'always')"
        self.pretrain_cfg = self.training_cfg.get('pretrain', {})
        self.pretrain_enabled: bool = self.pretrain_cfg.get('enabled', False)
        self.pretrain_steps: int = int(self.pretrain_cfg.get('steps', 0))
        self.pretrain_lr: float = float(
            self.pretrain_cfg.get('lr', self.vi_lr))
        self.pretrain_batch_size: int = int(
            self.pretrain_cfg.get('batch_size', self.training_batch_size))

        # KSIVI has no reverse model
        self.reverse_train = False

        logger.info(
            f"KSIVIRunner initialized: statistic={self.statistic_type}, "
            f"kernel={kernel_type}, detach_kernel={self.detach_kernel}, "
            f"detach_bandwidth={self.detach_bandwidth}, "
            f"log_p_reg={self.log_p_reg}, "
            f"log_p_reg_mode={self.log_p_reg_mode}, "
            f"affine_invariant={self.affine_invariant}"
        )

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Not used by KSIVI. Raises if accidentally called."""
        raise NotImplementedError(
            "KSIVI does not estimate log q(z). "
            "This method should not be called in the KSIVI training loop."
        )

    def train_reverse_model(self, epoch_outer: int):
        """No-op: KSIVI has no reverse model."""
        pass

    def pretrain_vi(self):
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
            "Starting KSIVI VI pretraining on dev split: "
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
                self.writer.add_scalar("pretrain/vi_model/loss", loss.item(), step)
                logger.info(
                    f"KSIVI pretrain step {step}/{self.pretrain_steps}: "
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
        Compute the KSD² loss and perform an optimizer step.

        For V-statistic: uses two independent batches for unbiased estimation.
        For U-statistic: uses a single batch with diagonal zeroed out.

        Returns:
            dict with diagnostic keys matching BaseSIVIRunner interface.
        """
        t_vi0 = time.perf_counter()

        # Sample and forward pass
        if self.statistic_type == 'v':
            # V-statistic: two independent batches
            eps1 = self.vi_model.sample_epsilon(num=self.training_batch_size)
            eps2 = self.vi_model.sample_epsilon(num=self.training_batch_size)
            z1, neg_score1 = self.vi_model.forward(eps1)
            z2, neg_score2 = self.vi_model.forward(eps2)
        else:
            # U-statistic: single batch
            eps1 = self.vi_model.sample_epsilon(num=self.training_batch_size)
            z1, neg_score1 = self.vi_model.forward(eps1)
            z2, neg_score2 = z1, neg_score1
            eps2 = eps1

        # Target scores — for stochastic data-dependent targets, both batches
        # must use the same data minibatch within a KSIVI step.
        if hasattr(self.target_model, 'sample_batch') and hasattr(
                self.target_model, 'score_on_batch'):
            batch_data, batch_labels = self.target_model.sample_batch()
            target_score1 = self.target_model.score_on_batch(
                z1, batch_data, batch_labels)
            target_score2 = self.target_model.score_on_batch(
                z2, batch_data, batch_labels)
        else:
            target_score1 = self.target_model.score(z1)
            target_score2 = self.target_model.score(z2)

        # Apply annealing to target scores
        anneal_factor = annealing(
            t=epoch,
            warm_up_interval=self.anneal_steps,
            anneal=self.use_annealing,
            scheme=self.anneal_scheme,
        )

        # f = annealed_target_score + neg_score_implicit
        # neg_score_implicit = ε/σ = -∇_z log q(z|ε), so
        # f = ∇_z log p(z) * anneal - ∇_z log q(z|ε) ≈ ∇_z log(p/q)
        f1 = anneal_factor * target_score1 + neg_score1
        f2 = anneal_factor * target_score2 + neg_score2

        t_vi1 = time.perf_counter()

        # Kernel matrix computation
        t_ns0 = time.perf_counter()

        if self.detach_kernel:
            K = self.kernel.pair_eval(
                z1.detach(),
                z2.detach(),
                fit_h=True,
                detach_h=True,
            )
        else:
            K = self.kernel.pair_eval(
                z1,
                z2,
                fit_h=True,
                detach_h=self.detach_bandwidth,
            )

        # Score product matrix: [N, N]
        if self.affine_invariant:
            with torch.no_grad():
                cov = torch.cov(z1.T)
                inv_cov = torch.inverse(
                    cov + 1e-8 * torch.eye(z1.shape[-1], device=z1.device))
            score_product = (f1 @ inv_cov) @ f2.T
        else:
            score_product = f1 @ f2.T

        if self.statistic_type == 'u':
            score_product = score_product.fill_diagonal_(0)

        # KSD² loss
        loss = (score_product * K).mean()

        # Optional log-p regularization (default: warmup-only).
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
                f"NaN or Inf detected in KSIVI loss at epoch {epoch}. "
                f"Skipping update."
            )

        t_bw1 = time.perf_counter()

        # Log KSIVI-specific diagnostics
        self.writer.add_scalar(
            "ksivi/kernel_bandwidth", self.kernel.h, epoch)
        self.writer.add_scalar(
            "ksivi/score_product_mean",
            score_product.detach().mean().item(), epoch)

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'z': z1,
            'epsilon': eps1,
            'time_vi_sample': t_vi1 - t_vi0,
            'time_neg_score': t_ns1 - t_ns0,
            'time_backward': t_bw1 - t_bw0,
        }
