import torch
from models.target_models import target_distribution
from models.vi_model import VIModel
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from utils.logging import get_logger, set_file_handler
import ite
import time
import numpy as np
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict
from utils.annealing import annealing
from utils.elm import kde_expected_log_marginal
from utils.metrics import compute_sliced_wasserstein, compute_ksd, compute_mmd

logger = get_logger()


class BaseSIVIRunner():
    '''
    The base Reverse SIVI class that encapsulates the model, training, and evaluation.
    
    Key components:
    - Target model: provides `logp` and plotting utilities.
    - VI model: parameterizes q_phi(z|epsilon)
    - Reverse model [Optional]: parameterizes q_psi(epsilon|z) via normalizing flow
    - Logging and artifact paths: under `results/` and `tb_logs/`.
    
    Args:
        config (DictConfig): Configuration for the experiment.
        name(str): Name of the Runner.
    '''

    def __init__(
        self,
        config: DictConfig,
        name: str = "BaseSIVIRunner",
    ):
        assert name != "BaseSIVIRunner", "Please use a subclass of BaseSIVIRunner."

        self.name: str = name
        self.config: DictConfig = config
        self.config_path: str = config.config_path
        self.device: torch.device = config.device

        # target type
        self.target_type: str = self.config.target_type
        logger.info(f"Target type: {self.target_type}")

        # target config
        default_target_config_path = f'configs/targets/{self.target_type}.yaml'
        if 'target_config_path' not in self.config:
            logger.warning(
                f"'target_config_path' not found in main_config; using default: {default_target_config_path}"
            )
            self.config.target_config_path = default_target_config_path
        target_config_path: str = self.config.target_config_path
        logger.info(f"Using target config path: {target_config_path}")
        _target_config = {'target': OmegaConf.load(target_config_path)}
        self.config = OmegaConf.merge(
            _target_config,
            self.config,
        )  # type: ignore

        # save path
        self.config.setdefault('output', {})
        results_dir = self.config.output.get('results_dir', 'results')
        tb_dir = self.config.output.get('tb_dir', 'tb_logs')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(results_dir, self.name, self.target_type,
                                      timestamp)
        os.makedirs(self.save_path, exist_ok=True)

        # attach file logger under save path
        set_file_handler(self.save_path, filename="run.log")
        logger.info(f"Artifacts will be saved to: {self.save_path}")

        # Determine resume behaviors
        self.resume_config: DictConfig = self.config.get(
            'resume', {'enabled': False})
        self.resume: bool = self.resume_config['enabled']

        # target
        self.target_model = self._build_target_model()

        # Override z_dim/epsilon_dim in target config if the target model
        # reports a different dimensionality (data-dependent targets).
        model_z_dim = getattr(self.target_model, 'z_dim', None)
        if model_z_dim is not None:
            cfg_z_dim = self.config.target.get('z_dim', None)
            if cfg_z_dim is not None and cfg_z_dim != model_z_dim:
                logger.info(
                    f"Overriding target z_dim from config ({cfg_z_dim}) "
                    f"with model z_dim ({model_z_dim})")
            self.config.target.z_dim = model_z_dim
            self.config.target.epsilon_dim = model_z_dim

        # baseline sample
        self.baseline_samples = self._load_baseline_samples()

        # kl ite samples
        self.metric_kl_enabled = self.config['metric']['kl_ite'].setdefault(
            'enabled', True)
        if self.metric_kl_enabled and self.baseline_samples is None:
            logger.warning(
                "No baseline samples available; disabling KL metric.")
            self.metric_kl_enabled = False
        self.n_ite_samples = self.config['metric']['kl_ite']['num_samples']

        # w2 samples
        self.metric_w2_enabled = self.config['metric']['w2'].setdefault(
            'enabled', True)
        if self.metric_w2_enabled and self.baseline_samples is None:
            logger.warning(
                "No baseline samples available; disabling W2 metric.")
            self.metric_w2_enabled = False
        self.n_w2_samples = self.config['metric']['w2']['num_samples']
        self.n_w2_projections = self.config['metric']['w2']['num_projections']

        # ksd config (KSD uses target_model.score(), not baseline samples)
        self.config.metric.setdefault('ksd', {})
        self.metric_ksd_enabled = self.config['metric']['ksd'].setdefault(
            'enabled', False)
        self.n_ksd_samples = self.config['metric']['ksd'].setdefault(
            'num_samples', 1000)

        # mmd config
        self.config.metric.setdefault('mmd', {})
        self.metric_mmd_enabled = self.config['metric']['mmd'].setdefault(
            'enabled', False)
        if self.metric_mmd_enabled and self.baseline_samples is None:
            logger.warning(
                "No baseline samples available; disabling MMD metric.")
            self.metric_mmd_enabled = False
        self.n_mmd_samples = self.config['metric']['mmd'].setdefault(
            'num_samples', 1000)
        self._init_mmd_baseline_samples()

        # fisher divergence config
        self.config.metric.setdefault('fisher', {})
        self.metric_fisher_enabled = self.config['metric'][
            'fisher'].setdefault('enabled', False)
        self.n_fisher_samples = self.config['metric']['fisher'].setdefault(
            'num_samples', 1000)
        self.n_fisher_is_samples = self.config['metric']['fisher'].setdefault(
            'num_is_samples', 512)

        # elbo samples
        self.metric_elbo_enabled = self.config['metric']['elbo'].setdefault(
            'enabled', True)
        self.n_elbo_z_samples = self.config['metric']['elbo']['num_z_samples']
        self.n_elbo_batches = self.config['metric']['elbo']['num_batches']
        self.n_elbo_batch_size = self.config['metric']['elbo']['batch_size']

        # KDE expected log marginal samples
        self.config.metric.setdefault('expected_log_marginal', {})
        self.metric_expected_log_marginal_enabled = self.config['metric'][
            'expected_log_marginal'].setdefault('enabled', True)
        if (self.metric_expected_log_marginal_enabled
                and self.baseline_samples is None):
            logger.warning(
                "No baseline samples available; disabling expected log marginal metric."
            )
            self.metric_expected_log_marginal_enabled = False
        self.n_expected_log_marginal_ref_samples = self.config['metric'][
            'expected_log_marginal'].setdefault('num_ref_samples', 1000)
        self.n_expected_log_marginal_model_samples = self.config['metric'][
            'expected_log_marginal'].setdefault('num_model_samples', 5000)
        self.n_expected_log_marginal_sample_batch_size = self.config['metric'][
            'expected_log_marginal'].setdefault('sample_batch_size', 5000)
        self.n_expected_log_marginal_dim_chunk = self.config['metric'][
            'expected_log_marginal'].setdefault('dim_chunk', 25)
        self.n_expected_log_marginal_ref_chunk = self.config['metric'][
            'expected_log_marginal'].setdefault('ref_chunk', 500)
        self.n_expected_log_marginal_model_chunk = self.config['metric'][
            'expected_log_marginal'].setdefault('model_chunk', 20000)
        self.expected_log_marginal_min_bandwidth = self.config['metric'][
            'expected_log_marginal'].setdefault('min_bandwidth', 1.0e-6)
        self.expected_log_marginal_dtype = self.config['metric'][
            'expected_log_marginal'].setdefault('dtype', 'float32')
        self._expected_log_marginal_reference_samples = None

        # bnn metrics config (RMSE + test log-likelihood; BNN targets only)
        # Auto-detect: enable if target is a DataBoundTarget wrapping Bnn,
        # unless explicitly configured via metric.bnn.enabled in the runner config.
        self.config.metric.setdefault('bnn', {})
        from models.data_bound_target import DataBoundTarget
        from models.target_models import Bnn
        _is_bnn_target = (isinstance(self.target_model, DataBoundTarget)
                          and isinstance(self.target_model.inner, Bnn))
        self.metric_bnn_enabled = self.config['metric']['bnn'].setdefault(
            'enabled', _is_bnn_target)
        if self.metric_bnn_enabled and not _is_bnn_target:
            logger.warning(
                "BNN metrics enabled but target is not a BNN; disabling.")
            self.metric_bnn_enabled = False
        self.n_bnn_samples = self.config['metric']['bnn'].setdefault(
            'num_samples', 500)

        # Jacobian spectral norm (Assumption 1 validation)
        self.config.metric.setdefault('jacobian_spectral', {})
        self.metric_jacobian_spectral_enabled = self.config['metric'][
            'jacobian_spectral'].setdefault('enabled', False)
        self.n_jacobian_spectral_samples = self.config['metric'][
            'jacobian_spectral'].setdefault('num_samples', 128)

        # vi model config
        self.vi_model_type: str = self.config.vi_model_type
        logger.info(f"VI model type: {self.vi_model_type}")

        if 'vi_model_config_path' not in self.config:
            default_vi_model_config_path = f'configs/vi_models/{self.vi_model_type}.yaml'
            logger.warning(
                f"'vi_model_config_path' not found in main_config; using default: {default_vi_model_config_path}"
            )
            self.config.vi_model_config_path = default_vi_model_config_path
        vi_model_config_path: str = self.config.vi_model_config_path
        logger.info(f"Using VI model config path: {vi_model_config_path}")
        _vi_model_config = {'vi_model': OmegaConf.load(vi_model_config_path)}
        self.config = OmegaConf.merge(
            _vi_model_config,
            self.config,
        )  # type: ignore

        self.epsilon_dim = self.config.vi_model['epsilon_dim']
        self.z_dim = self.config.vi_model['z_dim']

        self.vi_model = VIModel[self.vi_model_type](
            config=self.config.vi_model)
        self.vi_model.to(self.device)

        # Default no reverse model training, altered in subclasses
        self.reverse_train = False

        # TensorBoard writer
        self.tb_path = os.path.join(tb_dir, self.name, self.target_type,
                                    timestamp)
        os.makedirs(self.tb_path, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tb_path)

        # --------- Training/Experiment configuration ---------
        self.training_cfg: DictConfig = self.config['train']
        # epochs and batch sizes
        self.training_num_epochs = self.training_cfg['epochs']
        self.training_batch_size = self.training_cfg['batch_size']

        # Annealing config
        self.use_annealing: bool = self.training_cfg['annealing']['enabled']
        self.anneal_steps: int = self.training_cfg['annealing']['steps']
        self.anneal_scheme: str = self.training_cfg['annealing']['scheme']

        # VI optimizer/scheduler config
        self.vi_opt_cfg = self.training_cfg['vi']
        self.vi_lr = self.vi_opt_cfg['lr']
        self.vi_scheduler_cfg = self.vi_opt_cfg['scheduler']
        self.vi_opt_betas = tuple(self.vi_opt_cfg.get('betas', (0.9, 0.999)))
        self.vi_var_lr = self.vi_opt_cfg.get('var_lr', None)
        assert self.vi_scheduler_cfg['type'] == 'StepLR', \
            "Only StepLR scheduler is supported for VI optimizer."

        # Create VI optimizer and scheduler
        if hasattr(self.vi_model, 'optimizer_parameter_groups'):
            self.optimizer_vi = torch.optim.Adam(
                self.vi_model.optimizer_parameter_groups(self.vi_lr, self.vi_var_lr),
                lr=self.vi_lr,
                betas=self.vi_opt_betas,
            )
        elif self.vi_var_lr is not None and hasattr(self.vi_model, 'var_raw'):
            other_params = [
                p for n, p in self.vi_model.named_parameters()
                if n != 'var_raw'
            ]
            self.optimizer_vi = torch.optim.Adam(
                [
                    {
                        'params': other_params,
                        'lr': self.vi_lr
                    },
                    {
                        'params': [self.vi_model.var_raw],
                        'lr': self.vi_var_lr
                    },
                ],
                betas=self.vi_opt_betas,
            )
        else:
            self.optimizer_vi = torch.optim.Adam(
                self.vi_model.parameters(),
                lr=self.vi_lr,
                betas=self.vi_opt_betas,
            )
        self.scheduler_vi = torch.optim.lr_scheduler.StepLR(
            self.optimizer_vi,
            step_size=self.vi_scheduler_cfg['step_size'],
            gamma=self.vi_scheduler_cfg['gamma'],
        )

        # Sampling config
        self.training_sample_cfg = self.training_cfg['sample']
        self.training_sample_freq = self.training_sample_cfg['freq']
        self.training_sample_num = self.training_sample_cfg['num']
        self.training_sample_save_path = os.path.join(
            self.save_path,
            "samples",
        )
        os.makedirs(self.training_sample_save_path, exist_ok=True)

        # Logging config
        self.training_log_cfg = self.training_cfg['log']
        self.training_metric_log_freq = self.training_log_cfg[
            'metric_log_freq']
        self.training_loss_log_freq = self.training_log_cfg['loss_log_freq']

        # running accumulators
        self.training_sample_loss = 0.0
        self.training_steps = 0

        # Timing
        self.train_start_time: float = 0.0
        self.train_time_avg_window: int = self.training_log_cfg.get(
            'time_avg_window',
            100,
        )
        self.time_history = defaultdict(list[tuple[float, int]])

        # Starting epoch
        self.train_start_epoch: int = 1
        self.curr_epoch: int = 1

        # Checkpoint config
        self.ckpt_cfg = self.training_cfg['checkpoint']
        self.ckpt_enabled = self.ckpt_cfg['enabled']
        self.ckpt_freq = self.ckpt_cfg['freq']
        self.ckpt_base_path = os.path.join(self.save_path, "checkpoints")
        if self.ckpt_enabled:
            os.makedirs(self.ckpt_base_path, exist_ok=True)

        # Plotting config
        self.plot_cfg = self.training_cfg['plot']
        self.plot_freq = self.plot_cfg['freq']
        self.plot_num = self.plot_cfg['num']
        self.plot_save_path = os.path.join(self.save_path, "plots")
        os.makedirs(self.plot_save_path, exist_ok=True)

        # Gradient clipping (None = disabled)
        self.grad_clip = self.training_cfg.get('grad_clip', None)
        if self.grad_clip is not None:
            logger.info(
                f"Gradient clipping enabled with max_norm={self.grad_clip}")

        # Optional VI pretraining on BNN dev splits.
        self.pretrain_cfg = self.training_cfg.get('pretrain', {})
        self.pretrain_enabled: bool = self.pretrain_cfg.get('enabled', False)
        self.pretrain_steps: int = int(self.pretrain_cfg.get('steps', 0))
        self.pretrain_lr: float = float(
            self.pretrain_cfg.get('lr', self.vi_lr))
        self.pretrain_batch_size: int = int(
            self.pretrain_cfg.get('batch_size', self.training_batch_size))
        self._vi_pretrained_done = False

        # EMA (Exponential Moving Average) for stable evaluation
        ema_cfg = self.training_cfg.get('ema', {})
        self.ema_enabled = ema_cfg.get('enabled', False)
        if self.ema_enabled:
            from utils.ema import EMA
            self.ema_beta = ema_cfg.get('beta', 0.999)
            self.ema = EMA(
                beta=self.ema_beta,
                model_params=self.vi_model.parameters(),
            )
            logger.info(f"EMA enabled with beta={self.ema_beta}")

    # Data-dependent target types that require the DataBoundTarget wrapper
    _DATA_DEPENDENT_TARGETS = frozenset({
        "LRwaveform",
        "Bnn_boston",
        "Bnn_concrete",
        "Bnn_power",
        "Bnn_protein",
        "Bnn_winered",
        "Bnn_yacht",
    })

    def _build_target_model(self):
        """Instantiate the target model.

        For standard targets, delegates to the ``target_distribution`` registry.
        For data-dependent targets (``LRwaveform``, ``Bnn_boston``), uses the
        :func:`~models.data_bound_target.build_data_bound_target` factory which
        loads data and wraps the inner model.
        """
        if self.target_type in self._DATA_DEPENDENT_TARGETS:
            from models.data_bound_target import build_data_bound_target

            target_cfg = OmegaConf.to_container(self.config.get('target', {}),
                                                resolve=True)
            return build_data_bound_target(
                target_type=self.target_type,
                target_cfg=target_cfg,
                device=self.device,
            )
        return target_distribution[self.target_type](device=self.device)

    def log_config(self):
        '''
        Log the full configuration to TensorBoard and save as YAML file.
        '''
        # Log to TensorBoard as text
        config_str = OmegaConf.to_yaml(self.config, resolve=True)
        self.writer.add_text(
            "config/full_config",
            f"```yaml\n{config_str}\n```",
            0,
        )

        # Save to YAML file
        config_save_path = os.path.join(self.save_path, "full_config.yaml")
        with open(config_save_path, 'w') as f:
            f.write(config_str)
        logger.info(f"Saved full configuration to {config_save_path}.")

    def _load_baseline_samples(self) -> np.ndarray | None:
        """
        Load baseline MCMC samples from a configured path (`self.config.target.baseline_path`) for the current target. If not available, use a default path `baselines/mcmc/{target_dist}.pt`.

        Returns:
            samples (np.ndarray | None): Loaded baseline samples on cpu, or None if unavailable.

        """
        baseline_path = self.config.target.get('baseline_path', None)

        if not baseline_path:
            baseline_path = f'baselines/mcmc/{self.target_type}.pt'
            logger.warning(
                f"baseline_path not found, using default: {baseline_path}")
        try:
            samples = torch.load(baseline_path, map_location='cpu')
            if isinstance(samples, dict):
                samples = samples['samples']
            samples = torch.as_tensor(samples, dtype=torch.float32)
            logger.info(
                f"Loaded baseline samples from {baseline_path}, shape: {samples.shape}"
            )
            return samples.cpu().numpy()
        except Exception as e:
            logger.warning(
                f"Failed to load baseline samples from {baseline_path}: {e}. "
                f"KL and W2 metrics will be disabled.")
            return None

    def evaluate_vi_to_baseline_kl(self) -> float:
        """
        Estimate KL divergence KL(q_phi(z) || q_baseline(z)) using `ite.cost.BDKL_KnnK`.
        Returns:
            kl_div (float): Estimated KL divergence value.
        """
        if self.baseline_samples is None:
            raise RuntimeError(
                "Baseline samples not loaded; cannot compute KL divergence.")

        _, z = self.vi_model.sampling(num=self.n_ite_samples)
        z_np = z.cpu().numpy()

        cost_obj = ite.cost.BDKL_KnnK()
        try:
            kl_div = cost_obj.estimation(z_np, self.baseline_samples)
            return float(kl_div)
        except Exception as e:
            logger.error(f"KL estimation failed: {e}")
            raise e

    def evaluate_vi_to_baseline_w2(self) -> float:
        """
        Estimate Sliced Wasserstein-2 distance W2(q_phi(z), q_baseline(z)).
        Returns:
            w2 (float): Estimated W2 distance.
        """
        if self.baseline_samples is None:
            raise RuntimeError(
                "Baseline samples not loaded; cannot compute W2 distance.")

        _, z = self.vi_model.sampling(num=self.n_w2_samples)

        try:
            # baseline_samples is numpy on cpu. z is torch on device usually.
            # compute_sliced_wasserstein expects torch tensors.
            # We can run W2 on CPU or GPU. Let's send baseline to device to run on GPU if available for speed.

            baseline_tensor = torch.as_tensor(self.baseline_samples,
                                              device=self.device)
            # Use self.n_w2_projections
            w2 = compute_sliced_wasserstein(
                z,
                baseline_tensor,
                num_projections=self.n_w2_projections,
                device=self.device,
                p=2)
            return float(w2)
        except Exception as e:
            logger.error(f"W2 estimation failed: {e}")
            raise e

    def eval_kl_ite(self, epoch: int):
        '''
        Evaluate KL divergence between VI and baseline using ITE and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        '''
        kl_div = self.evaluate_vi_to_baseline_kl()
        self.writer.add_scalar("metric/vi_model/kl_ite", kl_div, epoch)
        logger.debug(f"Epoch {epoch}, VI KL to baseline: {kl_div:.4f}")

    def eval_w2(self, epoch: int):
        '''
        Evaluate W2 distance between VI and baseline and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        '''
        w2_dist = self.evaluate_vi_to_baseline_w2()
        self.writer.add_scalar("metric/vi_model/w2", w2_dist, epoch)
        logger.debug(f"Epoch {epoch}, VI W2 to baseline: {w2_dist:.4f}")

    def evaluate_elbo(self) -> tuple[float, float, float, float]:
        """
        Estimate ELBO using importance sampling for q_phi(z).
        ELBO = E_{z ~ q_phi} [log p(z) - log q_phi(z)]
        
        To estimate log q_phi(z), we use:
        q_phi(z) = E_{epsilon' ~ p(epsilon)} [q_phi(z|epsilon')]
        approximated by Monte Carlo integration over epsilon'.
        
        We use multiple batches to also estimate the standard error of q_phi(z) estimation.
        
        Returns:
            (elbo_mean, elbo_std_total, elbo_std_q, elbo_ci_half) (float, float, float, float): Estimated ELBO mean, total std, std from q(z) estimation, and 1/2 width of 0.95 CI.
        """
        # 1. Sample z from q_phi(z) and keep the generating epsilon so it can
        # be included in the Monte Carlo estimate for each sampled z.
        epsilon_samples, z_samples = self.vi_model.sampling(
            num=self.n_elbo_z_samples)
        # z_samples: [N_z, Dz]

        # 2. Estimate log q_phi(z) for each z sample
        # q_phi(z) \approx (1/K) \sum_{k=1}^K q_phi(z|epsilon'_k)
        # log q_phi(z) \approx logsumexp(log q_phi(z|epsilon'_k)) - log K

        # We perform this for multiple batches of epsilon' to get variance estimate
        # Batches: B batches of size S

        # Accumulate q(z) estimate from auxiliary epsilon draws.
        # We work in log space for stability and add the generating epsilon
        # once after aggregating all auxiliary batches.
        batch_log_q_z_sums = []
        total_aux_samples = self.n_elbo_batches * self.n_elbo_batch_size

        with torch.no_grad():
            for _ in range(self.n_elbo_batches):
                # Sample epsilon' batch
                # [S, De]
                epsilon_prime = self.vi_model.sample_epsilon(
                    num=self.n_elbo_batch_size)

                # Expand to match shapes explicitly as requested using repeat
                # z: [N_z, 1, Dz] -> [N_z, S, Dz]
                # epsilon': [1, S, De] -> [N_z, S, De]
                z_expanded = z_samples.unsqueeze(1).expand(
                    -1,
                    self.n_elbo_batch_size,
                    -1,
                )
                eps_expanded = epsilon_prime.unsqueeze(0).expand(
                    self.n_elbo_z_samples,
                    -1,
                    -1,
                )

                # [N_z, S]
                log_q_z_given_eps = self.vi_model.logp(
                    z_expanded,
                    eps_expanded,
                )

                # Sum over S (in log domain) for this batch
                batch_log_sum = torch.logsumexp(
                    log_q_z_given_eps,
                    dim=1,
                )  # [N_z]
                batch_log_q_z_sums.append(batch_log_sum)

        # Stack: [N_z, B]
        log_sums_tensor = torch.stack(batch_log_q_z_sums, dim=1)

        # Add the generating epsilon contribution once per sampled z.
        log_q_z_given_generating_eps = self.vi_model.logp(
            z_samples,
            epsilon_samples,
        )

        # --- Total Estimate (using B*S auxiliary samples plus one generating
        # epsilon contribution) ---
        log_aux_total_sum = torch.logsumexp(log_sums_tensor, dim=1)  # [N_z]
        log_total_sum = torch.logaddexp(
            log_aux_total_sum,
            log_q_z_given_generating_eps,
        )
        total_samples = total_aux_samples + 1
        log_q_z_mean = log_total_sum - torch.log(
            torch.tensor(total_samples, device=self.device))

        # --- Variance Estimation ---
        # Estimator_b = (1/S) * exp(batch_log_sum_b)
        # We want variance of the mean estimator.
        # Var(Mean) = Var(Estimator_b) / B

        log_estimators_b = log_sums_tensor - torch.log(
            torch.tensor(self.n_elbo_batch_size, device=self.device))

        # Using Delta method for variance of log q(z): Var(log X) \approx Var(X) / E[X]^2

        estimators_b = torch.exp(log_estimators_b)  # [N_z, B]
        var_estimators = torch.var(estimators_b, dim=1)  # [N_z]
        mean_estimators = torch.exp(log_q_z_mean)  # [N_z]

        # Squared standard error of mean estimator (of q(z)).
        # The one-time generating-epsilon term is fixed conditional on z, so
        # only the auxiliary-average term contributes Monte Carlo variance.
        aux_weight = total_aux_samples / total_samples
        sq_se_mean_q = (aux_weight**2) * var_estimators / self.n_elbo_batches

        # Squared standard error of log q(z)
        sq_se_log_q = sq_se_mean_q / (mean_estimators**2 + 1e-10)

        # 3. Compute log p(z)
        log_p_z = self.target_model.logp(z_samples)  # [N_z]

        # 4. Compute ELBO per sample
        # elbo_i = log p(z_i) - log q(z_i)
        elbo_per_sample = log_p_z - log_q_z_mean

        # Mean ELBO
        elbo_mean = torch.mean(elbo_per_sample)

        # Total ELBO Std (direct std of calculated ELBO)
        elbo_std_total = torch.std(elbo_per_sample)

        # Std arising from estimating q_phi(z) (Average std of the log q estimator)
        elbo_std_q = torch.sqrt(torch.mean(sq_se_log_q))

        # 1/2 width of 0.95 confidence interval
        elbo_ci_half = 1.96 * elbo_std_total / (self.n_elbo_z_samples**0.5)

        return elbo_mean.item(), elbo_std_total.item(), elbo_std_q.item(
        ), elbo_ci_half.item()

    def eval_elbo(self, epoch: int):
        '''
        Evaluate ELBO metric and log to TensorBoard.
        '''
        elbo_val, elbo_std_total, elbo_std_q, elbo_ci_half = self.evaluate_elbo(
        )
        self.writer.add_scalar("metric/vi_model/elbo", elbo_val, epoch)
        self.writer.add_scalar("metric/vi_model/elbo_std_total",
                               elbo_std_total, epoch)
        self.writer.add_scalar("metric/vi_model/elbo_std_q", elbo_std_q, epoch)
        self.writer.add_scalar("metric/vi_model/elbo_ci_half", elbo_ci_half,
                               epoch)

        logger.debug(
            f"Epoch {epoch}, ELBO: {elbo_val:.4f}, Std Total: {elbo_std_total:.4f}, Std Q: {elbo_std_q:.4f}, CI Half: {elbo_ci_half:.4f}"
        )

    def _sample_reference_baseline_samples(self,
                                           num_samples: int) -> torch.Tensor:
        """Sample reference points from the baseline store without replacement."""
        if self.baseline_samples is None:
            raise RuntimeError(
                "Baseline samples not loaded; cannot compute expected log marginal."
            )

        if self.baseline_samples.shape[0] > num_samples:
            indices = np.random.choice(
                self.baseline_samples.shape[0],
                num_samples,
                replace=False,
            )
            reference_samples = self.baseline_samples[indices]
        else:
            reference_samples = self.baseline_samples

        return torch.as_tensor(reference_samples, device=self.device)

    def _expected_log_marginal_reference_set(self) -> torch.Tensor:
        """Return the fixed reference set used for training-time KDE ELM."""
        if self._expected_log_marginal_reference_samples is None:
            self._expected_log_marginal_reference_samples = (
                self._sample_reference_baseline_samples(
                    self.n_expected_log_marginal_ref_samples))
        return self._expected_log_marginal_reference_samples

    @torch.no_grad()
    def _sample_vi_model_for_kde(self, num_samples: int,
                                 batch_size: int) -> torch.Tensor:
        """Draw VI samples for the coordinate-wise KDE ELM metric."""
        if num_samples < 1:
            raise ValueError("num_model_samples must be at least 1.")
        if batch_size < 1:
            raise ValueError("sample_batch_size must be at least 1.")

        samples = torch.empty(
            (num_samples, self.z_dim),
            device=self.device,
            dtype=next(self.vi_model.parameters()).dtype,
        )
        was_training = self.vi_model.training
        self.vi_model.eval()
        try:
            for start in range(0, num_samples, batch_size):
                current = min(batch_size, num_samples - start)
                _, z = self.vi_model.sampling(num=current)
                samples[start:start + current].copy_(z.detach())
        finally:
            if was_training:
                self.vi_model.train()
        return samples

    def evaluate_expected_log_marginal(self):
        r"""Estimate paper-style KDE expected log marginal.

        The metric is

            E_{z ~ r}[sum_j log q_hat_phi,j(z_j)],

        where ``r`` is represented by reference baseline samples and each
        ``q_hat_phi,j`` is a one-dimensional Gaussian KDE fit from VI samples.

        Returns:
            KDEELMEstimate: Scalar metric, per-reference values, and
                diagnostics from the chunked KDE evaluator.
        """
        reference_samples = self._expected_log_marginal_reference_set()
        model_samples = self._sample_vi_model_for_kde(
            self.n_expected_log_marginal_model_samples,
            self.n_expected_log_marginal_sample_batch_size,
        )
        return kde_expected_log_marginal(
            reference_samples,
            model_samples,
            dim_chunk=self.n_expected_log_marginal_dim_chunk,
            ref_chunk=self.n_expected_log_marginal_ref_chunk,
            model_chunk=self.n_expected_log_marginal_model_chunk,
            min_bandwidth=self.expected_log_marginal_min_bandwidth,
            dtype=self.expected_log_marginal_dtype,
            device=self.device,
        )

    def eval_expected_log_marginal(self, epoch: int):
        """Evaluate KDE expected log marginal and log to TensorBoard."""
        estimate = self.evaluate_expected_log_marginal()
        metric_mean = estimate.value
        diagnostics = estimate.diagnostics
        self.writer.add_scalar(
            "metric/vi_model/expected_log_marginal",
            metric_mean,
            epoch,
        )
        self.writer.add_scalar(
            "metric/vi_model/kde_expected_log_marginal",
            metric_mean,
            epoch,
        )
        self.writer.add_scalar(
            "diagnostic/vi_model/kde_expected_log_marginal_std",
            diagnostics["std_across_refs"],
            epoch,
        )
        self.writer.add_scalar(
            "diagnostic/vi_model/kde_expected_log_marginal_clamped_dims",
            diagnostics["num_bandwidth_clamped_dims"],
            epoch,
        )
        logger.debug(
            f"Epoch {epoch}, KDE Expected Log Marginal: {metric_mean:.4f}"
        )

    def evaluate_ksd(self) -> float:
        '''
        Evaluate Kernelized Stein Discrepancy (KSD) for VI samples.
        Returns:
            ksd (float): Estimated KSD value.
        '''
        _, z = self.vi_model.sampling(num=self.n_ksd_samples)
        scores = self.target_model.score(z)
        try:
            ksd = compute_ksd(
                z,
                scores=scores,
            )
            return float(ksd)
        except Exception as e:
            logger.error(f"KSD estimation failed: {e}")
            raise e

    def eval_ksd(self, epoch: int):
        '''
        Evaluate KSD metric and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        '''
        ksd_val = self.evaluate_ksd()
        self.writer.add_scalar("metric/vi_model/ksd", ksd_val, epoch)
        logger.debug(f"Epoch {epoch}, VI KSD: {ksd_val:.4f}")

    def _init_mmd_baseline_samples(self):
        '''
        Initialize baseline samples and kernel for MMD evaluation.
        '''
        if not self.metric_mmd_enabled:
            return

        if self.baseline_samples.shape[0] > self.n_mmd_samples:
            self._mmd_baseline_subset = self.baseline_samples[np.random.choice(
                self.baseline_samples.shape[0],
                self.n_mmd_samples,
                replace=False,
            )]
        else:
            self._mmd_baseline_subset = self.baseline_samples

        self.mmd_baseline_subset = torch.as_tensor(
            self._mmd_baseline_subset).to(self.device)

        from utils.kernels import GaussianKernel
        self.mmd_baseline_kernel = GaussianKernel()
        self.mmd_baseline_kernel.fit_h(self.mmd_baseline_subset)

    def evaluate_mmd(self) -> float:
        '''
        Evaluate Maximum Mean Discrepancy (MMD) for VI samples.
        Returns:
            mmd (float): Estimated MMD value.
        '''
        _, z = self.vi_model.sampling(num=self.n_mmd_samples)

        try:
            mmd = compute_mmd(
                z,
                self.mmd_baseline_subset,
                self.mmd_baseline_kernel,
            )
            return float(mmd)
        except Exception as e:
            logger.error(f"MMD estimation failed: {e}")
            raise e

    def eval_mmd(self, epoch: int):
        '''
        Evaluate MMD metric and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        '''
        mmd_val = self.evaluate_mmd()
        self.writer.add_scalar("metric/vi_model/mmd", mmd_val, epoch)
        logger.debug(f"Epoch {epoch}, VI MMD: {mmd_val:.4f}")

    def evaluate_bnn_metrics(self) -> tuple[float, float]:
        """Sample from VI and compute BNN test RMSE and test log-likelihood.

        Returns:
            (rmse, test_llk): RMSE of ensemble mean predictions, and Monte Carlo
            marginalised test log-likelihood. NLL = -test_llk.
        """
        with torch.no_grad():
            _, z = self.vi_model.sampling(num=self.n_bnn_samples)
        return self.target_model.rmse_llk(z)

    def eval_bnn(self, epoch: int):
        """Evaluate BNN RMSE / test log-likelihood and log to TensorBoard.

        Args:
            epoch (int): Current epoch number.
        """
        rmse, test_llk = self.evaluate_bnn_metrics()
        self.writer.add_scalar("metric/vi_model/rmse", rmse, epoch)
        self.writer.add_scalar("metric/vi_model/test_llk", test_llk, epoch)
        self.writer.add_scalar("metric/vi_model/nll", -test_llk, epoch)
        logger.debug(
            f"Epoch {epoch}, BNN RMSE: {rmse:.4f}, Test LLK: {test_llk:.4f}, NLL: {-test_llk:.4f}"
        )

    def evaluate_fisher_divergence(self) -> float:
        """
        Estimate the Fisher divergence between q_phi(z) and the target p(z):

            FD(p || q) = E_{z ~ q_phi} [ || score_p(z) - score_q(z) ||^2 ]

        where:
            score_p(z) = nabla_z log p(z)  -- from target_model.score()
            score_q(z) = nabla_z log q_phi(z)
                       ~ sum_k softmax(log q(z|eps_k)) * nabla_z log q(z|eps_k)
                       = E_{eps ~ p(eps)} [ nabla_z log q(z|eps) ]  (IS estimate)

        The IS estimate of score_q uses the same logsumexp-softmax weighting as
        the ELBO estimator. vi_model.score(z, eps) = -(z - mu(eps)) / var(eps)
        gives nabla_z log q(z|eps) analytically.

        Returns:
            fisher_div (float): Estimated Fisher divergence.
        """
        with torch.no_grad():
            # 1. Sample z ~ q_phi
            _, z_samples = self.vi_model.sampling(num=self.n_fisher_samples)
            # z_samples: [N, Dz]

            # 2. Sample epsilon' ~ p(epsilon) for IS
            eps = self.vi_model.sample_epsilon(num=self.n_fisher_is_samples)
            # eps: [K, De]

            # Expand for joint computation: z [N,1,Dz], eps [1,K,De]
            z_exp = z_samples.unsqueeze(1).expand(-1, self.n_fisher_is_samples,
                                                  -1)
            eps_exp = eps.unsqueeze(0).expand(self.n_fisher_samples, -1, -1)

            # 3. log q(z|eps_k): [N, K]
            log_q_z_given_eps = self.vi_model.logp(z_exp, eps_exp)

            # 4. Softmax weights over K: [N, K]
            log_w = log_q_z_given_eps - torch.logsumexp(
                log_q_z_given_eps, dim=1, keepdim=True)
            w = torch.exp(log_w)  # [N, K]

            # 5. nabla_z log q(z|eps_k): [N, K, Dz]
            score_q_given_eps = self.vi_model.score(z_exp, eps_exp)

            # 6. Weighted sum -> score_q(z): [N, Dz]
            score_q = (w.unsqueeze(-1) * score_q_given_eps).sum(dim=1)

            # 7. Ground truth score: [N, Dz]
            score_p = self.target_model.score(z_samples)

            # 8. Fisher divergence: E[ || score_p - score_q ||^2 ]
            fisher_div = torch.mean(torch.sum((score_p - score_q)**2, dim=-1))

        return fisher_div.item()

    def eval_fisher(self, epoch: int):
        '''
        Evaluate Fisher divergence and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        '''
        fisher_val = self.evaluate_fisher_divergence()
        self.writer.add_scalar("metric/vi_model/fisher_div", fisher_val, epoch)
        logger.debug(f"Epoch {epoch}, Fisher Divergence: {fisher_val:.4f}")

    def eval_jacobian_spectral(self, epoch: int):
        '''
        Evaluate Jacobian spectral norms (Bounded Reparameterization Assumption)
        and log to TensorBoard.

        Computes E_ε[‖∇_φ μ_φ(ε)‖₂⁴] and E_ε[‖∇_φ σ_φ(ε)‖₂⁴] where the
        norm is the matrix 2-norm (spectral norm) of the d_z × d_φ Jacobian.

        Args:
            epoch (int): Current epoch number.
        '''
        from utils.jacobian_spectral import evaluate_assumption_bound

        was_training = self.vi_model.training
        self.vi_model.eval()

        epsilon = self.vi_model.sample_epsilon(
            num=self.n_jacobian_spectral_samples)
        bound = evaluate_assumption_bound(self.vi_model, epsilon)

        self.writer.add_scalar(
            "metric/vi_model/jacobian_spectral_mu",
            bound.mean_sq_spectral_mu, epoch)
        self.writer.add_scalar(
            "metric/vi_model/jacobian_spectral_std",
            bound.mean_sq_spectral_std, epoch)
        self.writer.add_scalar(
            "metric/vi_model/vi_derivative_fourth_moment",
            bound.M_eps, epoch)
        logger.debug(
            f"Epoch {epoch}, VI_D4M: {bound.M_eps:.4f} "
            f"(mu: {bound.mean_sq_spectral_mu:.4f}, "
            f"std: {bound.mean_sq_spectral_std:.4f})")

        if was_training:
            self.vi_model.train()

    def log_reverse_score_l2_to_target(
        self,
        score_eval: torch.Tensor,
        z_eval: torch.Tensor,
    ) -> None:
        '''
        Log the mean squared L2 gap between a reverse-estimated score and the
        target score at the same latent samples.

        Args:
            score_eval (torch.Tensor): Reverse-estimated score with shape [B, Dz].
            z_eval (torch.Tensor): Latent samples with shape [B, Dz].
        '''
        with torch.no_grad():
            target_score = self.target_model.score(z_eval.detach())
            score_l2 = torch.mean(
                torch.sum((score_eval.detach() - target_score)**2, dim=-1))
            self.writer.add_scalar(
                "diagnostic/reverse_model/score_l2_to_target",
                score_l2.item(),
                self.curr_epoch,
            )

    def save_samples(self, epoch: int):
        '''
        Save samples from the VI model at the given epoch.
        Args:
            epoch (int): Current epoch number.
        '''
        current_sample_time = time.perf_counter()
        epsilon_sample, z_sample = self.vi_model.sampling(
            num=self.training_sample_num)

        sample_dict = {
            'z': z_sample,
            'epsilon': epsilon_sample,
            'epoch': epoch,
            'time': current_sample_time - self.train_start_time,
            'exp_name': self.name,
            'target_type': self.target_type,
            'vi_model_type': self.vi_model_type,
        }

        torch.save(
            sample_dict,
            os.path.join(
                self.training_sample_save_path,
                f"samples_epoch_{epoch}.pt",
            ))

        logger.debug(
            f"Saved {self.training_sample_num} samples at epoch {epoch}.")

    def save_checkpoint(self, epoch: int):
        '''
        Save the state dict of model and optimizer to checkpoints at the given epoch.
        Args:
            epoch (int): Current epoch number.
        '''
        epoch_ckpt_dir = os.path.join(self.ckpt_base_path, f"epoch_{epoch}")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        # Save VI model
        vi_ckpt_path = os.path.join(epoch_ckpt_dir, "vi_model.pt")
        torch.save(self.vi_model.state_dict(), vi_ckpt_path)
        # Save VI optimizer and scheduler
        vi_opt_path = os.path.join(epoch_ckpt_dir, "vi_optim.pt")
        vi_sched_path = os.path.join(epoch_ckpt_dir, "vi_sched.pt")
        torch.save(self.optimizer_vi.state_dict(), vi_opt_path)
        torch.save(self.scheduler_vi.state_dict(), vi_sched_path)
        logger.debug(
            f"Saved VIModel checkpoints at epoch {epoch} to {epoch_ckpt_dir}.")

    def load_checkpoints(self):
        '''
        Load model state dicts from checkpoint directory when resuming training. Use default initialization if checkpoint files are missing. Will try to load optimizer and scheduler states if available.
        '''
        ckpt_dir = self.config.resume.ckpt_dir
        if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
            raise RuntimeError(
                f"Checkpoint directory {ckpt_dir} does not exist or is empty.")
        logger.info(f"Resume requested. Checkpoint dir: {ckpt_dir}")
        try:
            # VI model checkpoint
            vi_ckpt_path = os.path.join(
                ckpt_dir,
                'vi_model.pt',
            )
            if os.path.isfile(vi_ckpt_path):
                state = torch.load(
                    vi_ckpt_path,
                    map_location=self.device,
                )
                self.vi_model.load_state_dict(state)
                logger.info(f"Loaded VI model checkpoint from {vi_ckpt_path}")
            else:
                logger.warning(
                    f"VI checkpoint not found at {vi_ckpt_path}; using default initialization."
                )
        except Exception as e:
            logger.error(f"Failed to load checkpoints from {ckpt_dir}: {e}.")
            raise e

        if not self.config.resume.get('load_optimizer', False):
            return

        logger.debug("Trying to load optimizer and scheduler states...")

        try:
            vi_opt_path = os.path.join(ckpt_dir, 'vi_optim.pt')
            vi_sched_path = os.path.join(
                ckpt_dir,
                'vi_sched.pt',
            )
            if os.path.isfile(vi_opt_path):
                opt_state = torch.load(
                    vi_opt_path,
                    map_location=self.device,
                )
                self.optimizer_vi.load_state_dict(opt_state)
                logger.info(f"Loaded VI optimizer from {vi_opt_path}")
            else:
                logger.warning(
                    f"VI optimizer checkpoint not found at {vi_opt_path}; using fresh optimizer."
                )
            if os.path.isfile(vi_sched_path):
                sched_state = torch.load(
                    vi_sched_path,
                    map_location=self.device,
                )
                self.scheduler_vi.load_state_dict(sched_state)
                logger.info(f"Loaded VI scheduler from {vi_sched_path}")
            else:
                logger.warning(
                    f"VI scheduler checkpoint not found at {vi_sched_path}; using fresh scheduler."
                )
        except Exception as e:
            logger.error(f"Failed to load VI optimizer/scheduler: {e}.")
            raise e

        if self.config.resume.get("no_override_epoch", False):
            return

        # Set starting epoch
        try:
            base = os.path.basename(ckpt_dir.rstrip('/'))
            if base.startswith('epoch_'):
                parsed = int(base.split('_')[1])
                self.train_start_epoch = parsed + 1
                logger.info(
                    f"Starting training from epoch {self.train_start_epoch} due to resume."
                )
            else:
                logger.warning(
                    f"Resume dir '{ckpt_dir}' does not end with 'epoch_<n>'; starting from epoch 1."
                )
        except Exception as e:
            logger.warning(
                f"Failed to parse starting epoch from resume dir '{ckpt_dir}': {e}. Starting from epoch 1."
            )

    def train_reverse_model(self, epoch_outer: int):
        '''
        Train the reverse model for several inner epochs using samples from the current VI.
        Args:
            epoch_outer (int): Current outer epoch number.
        '''
        raise NotImplementedError(
            "train_reverse_model must be implemented in subclasses.")

    def pretrain_vi(self):
        """Optional VI pretraining hook for BNN targets with dev splits."""
        from models.data_bound_target import DataBoundTarget

        if (not self.pretrain_enabled or self.pretrain_steps <= 0 or
                not isinstance(self.target_model, DataBoundTarget) or
                self.target_model.dev_data is None or
                not hasattr(self.target_model.inner, 'predict_y')):
            return

        X_dev, y_dev, mean_y, std_y = self.target_model.dev_data
        if (X_dev is None or y_dev is None or mean_y is None or std_y is None):
            return

        optimizer = torch.optim.Adam(
            self.vi_model.parameters(),
            lr=self.pretrain_lr,
            betas=self.vi_opt_betas,
        )
        log_freq = max(1, self.pretrain_steps // 10)
        logger.info(
            "Starting VI pretraining on dev split: "
            f"steps={self.pretrain_steps}, lr={self.pretrain_lr}"
        )

        self.vi_model.train()
        for step in range(1, self.pretrain_steps + 1):
            epsilon = self.vi_model.sample_epsilon(num=self.pretrain_batch_size)
            z, _ = self.vi_model.forward(epsilon)
            pred_y = self.target_model.inner.predict_y(z, X_dev, mean_y, std_y)
            loss = ((pred_y.mean(0) - y_dev)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.vi_model.parameters(),
                    max_norm=self.grad_clip,
                )
            optimizer.step()

            if step % log_freq == 0 or step == self.pretrain_steps:
                self.writer.add_scalar("pretrain/vi_model/loss", loss.item(),
                                       step)
                logger.info(
                    f"VI pretrain step {step}/{self.pretrain_steps}: "
                    f"loss={loss.item():.6f}"
                )

        # Reset EMA after pretraining so evaluation starts from the pretrained
        # weights rather than the random initialization snapshot.
        if self.ema_enabled:
            from utils.ema import EMA
            self.ema = EMA(
                beta=self.ema_beta,
                model_params=self.vi_model.parameters(),
            )
        self._vi_pretrained_done = True

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Estimate log q_phi(z), especially the gradient.

        Args:
            z (torch.Tensor): Samples from q_phi(z|epsilon), shape (batch_size, z_dim).
            epsilon (torch.Tensor): Corresponding epsilon samples, shape (batch_size, epsilon_dim).
        
        Returns:
            log_q_phi_z (torch.Tensor): Estimated log q_phi(z), shape (batch_size,).
        '''
        raise NotImplementedError(
            "calc_log_q_phi_z must be implemented in subclasses.")

    def _compute_loss_and_step(self, epoch: int) -> dict:
        """
        Compute the training loss, perform the optimizer step, and return diagnostics.

        This default implementation computes the ELBO-based loss used by standard SIVI
        and its reverse-model variants. Subclasses (e.g., KSIVIRunner) may override this
        to implement alternative objectives such as KSD².

        Args:
            epoch (int): Current training epoch.

        Returns:
            dict with keys:
                - 'loss' (torch.Tensor): Scalar loss value.
                - 'grad_norm' (float): Gradient norm before step.
                - 'z' (torch.Tensor): Sampled z for diagnostic logging.
                - 'epsilon' (torch.Tensor): Sampled epsilon for diagnostic logging.
                - 'time_vi_sample' (float): Time for sampling + forward pass.
                - 'time_neg_score' (float): Time for log q estimation.
                - 'time_backward' (float): Time for backward pass + optimizer step.
        """
        # Sample epsilon
        t_vi0 = time.perf_counter()
        epsilon = self.vi_model.sample_epsilon(num=self.training_batch_size)

        # Sample z from variational distribution
        z, neg_score_implicit = self.vi_model.forward(epsilon)

        # Compute log prob under target distribution using score-gradient trick
        log_prob_target: torch.Tensor = self.target_model.score(
            z.clone().detach()) * z
        log_prob_target = log_prob_target.sum(dim=-1)

        # Apply annealing if enabled
        anneal_factor = annealing(
            t=epoch,
            warm_up_interval=self.anneal_steps,
            anneal=self.use_annealing,
            scheme=self.anneal_scheme,
        )
        log_prob_target = log_prob_target * anneal_factor

        t_vi1 = time.perf_counter()

        t_ns0 = time.perf_counter()

        # Estimate log q_phi(z)
        result = self.calc_log_q_phi_z(z, epsilon)
        if isinstance(result, tuple):
            log_q_phi_z, score_q = result
        else:
            log_q_phi_z = result
            score_q = None

        t_ns1 = time.perf_counter()

        t_bw0 = time.perf_counter()

        # Compute ELBO loss
        loss = -torch.mean(log_prob_target - log_q_phi_z)
        grad_norm = None

        if torch.isfinite(loss):
            self.optimizer_vi.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.vi_model.parameters(), max_norm=self.grad_clip)
            else:
                grad_norm = torch.nn.utils.get_total_norm(
                    [p.grad for p in self.vi_model.parameters() if p.grad is not None])
            self.optimizer_vi.step()
            self.scheduler_vi.step()
            if self.ema_enabled:
                self.ema.update_params(self.vi_model.parameters())
        else:
            logger.warning(
                f"NaN or Inf detected in VI loss at epoch {epoch}. Skipping update."
            )
            logger.debug(
                f"Detected {(~torch.isfinite(log_prob_target)).sum()} non-finite values in log_prob_target."
            )
            logger.debug(
                f"Detected {(~torch.isfinite(log_q_phi_z)).sum()} non-finite values in log_q_phi_z."
            )

        t_bw1 = time.perf_counter()

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'z': z,
            'epsilon': epsilon,
            'score_q': score_q,
            'time_vi_sample': t_vi1 - t_vi0,
            'time_neg_score': t_ns1 - t_ns0,
            'time_backward': t_bw1 - t_bw0,
        }

    def learn(self):
        '''
        Run the full training procedure for UIVI.

        Algorithm overview:
        1. Warm up the reverse model [Optional].
        2. For each VI step:
            - Sample z from VI q_phi(z|epsilon).
            - Compute target log-density `log p(z)` from the target model.
            - Estimate `nabla_z log q_phi(z)`
            - loss = - E_q[ log p(z) - nabla_z log q_phi(z) ]
        3. Periodically update the reverse model for several inner epochs using
            samples from the current VI [Optional].
        
        Returns:
            None
        '''

        # If resuming, optionally load optimizer & scheduler states
        if self.resume:
            self.load_checkpoints()
        elif not self._vi_pretrained_done:
            self.pretrain_vi()

        # Main training loop
        self.vi_model.train()
        # Timing accumulators
        last_time = time.perf_counter()
        self.train_start_time = time.perf_counter()
        time_scalars = {}

        for epoch in tqdm(
                range(self.train_start_epoch, self.training_num_epochs + 1),
                desc="Main Training",
                initial=self.train_start_epoch - 1,
                total=self.training_num_epochs,
        ):
            epoch_start_time = time.perf_counter()
            self.curr_epoch = epoch
            time_scalars.clear()

            # Compute loss and perform optimizer step
            diagnostics = self._compute_loss_and_step(epoch)
            loss = diagnostics['loss']
            z = diagnostics['z']
            epsilon = diagnostics['epsilon']
            grad_norm = diagnostics['grad_norm']
            score_q = diagnostics.get('score_q', None)
            time_scalars['vi_sample'] = diagnostics['time_vi_sample']
            time_scalars['neg_score'] = diagnostics['time_neg_score']
            time_scalars['backward'] = diagnostics['time_backward']

            # TensorBoard scalars
            self.writer.add_scalar("train/vi_model/loss", loss.item(), epoch)
            if grad_norm is not None:
                self.writer.add_scalar(
                    "diagnostic/vi_model/grad_norm",
                    grad_norm.item(),
                    epoch,
                )
            weight_norm = torch.nn.utils.get_total_norm(
                self.vi_model.parameters())
            self.writer.add_scalar(
                "diagnostic/vi_model/weight_norm",
                weight_norm.item(),
                epoch,
            )
            z_norm = torch.norm(z, dim=1)
            self.writer.add_scalar(
                "diagnostic/vi_model/z_norm_avg",
                z_norm.mean().item(),
                epoch,
            )
            self.writer.add_scalar(
                "diagnostic/vi_model/z_norm_std",
                z_norm.std().item(),
                epoch,
            )
            epsilon_norm = torch.norm(epsilon, dim=1)
            self.writer.add_scalar(
                "diagnostic/vi_model/epsilon_norm_avg",
                epsilon_norm.mean().item(),
                epoch,
            )
            self.writer.add_scalar(
                "diagnostic/vi_model/epsilon_norm_std",
                epsilon_norm.std().item(),
                epoch,
            )

            if score_q is not None:
                with torch.no_grad():
                    score_conditional = self.vi_model.score(
                        z.detach(), epsilon.detach())
                    score_gap = torch.mean(
                        torch.sum((score_q - score_conditional)**2, dim=-1))
                    self.writer.add_scalar(
                        "diagnostic/vi_model/marginal_conditional_score_l2_gap",
                        score_gap.item(),
                        epoch,
                    )

            self.training_sample_loss += loss.item()
            if epoch % self.training_loss_log_freq == 0:
                avg_loss = self.training_sample_loss / self.training_loss_log_freq
                current_time = time.perf_counter()
                avg_epoch_time = (current_time -
                                  last_time) / self.training_loss_log_freq

                logger.debug(
                    f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Avg Epoch Time: {avg_epoch_time:.4f}s"
                )
                # Reset accumulators
                self.training_sample_loss = 0.0
                last_time = current_time

            # Train reverse model
            if self.reverse_train:
                time_rev0 = time.perf_counter()
                self.train_reverse_model(epoch)
                time_rev1 = time.perf_counter()
                time_reverse_step = time_rev1 - time_rev0
                time_scalars['reverse_train'] = time_reverse_step

            # Generate and save samples
            needs_sample = (epoch % self.training_sample_freq == 0)
            needs_metrics = (self.training_metric_log_freq > 0
                             and epoch % self.training_metric_log_freq == 0)
            needs_plot = (epoch % self.plot_freq == 0)

            # EMA: swap to shadow params for evaluation/sampling/plotting
            _ema_swapped = False
            if self.ema_enabled and (needs_sample or needs_metrics
                                     or needs_plot):
                self.ema.store(self.vi_model.parameters())
                self.ema.apply_shadow(self.vi_model.parameters())
                _ema_swapped = True

            if needs_sample:
                time_sample0 = time.perf_counter()
                self.save_samples(epoch)
                time_sample1 = time.perf_counter()
                time_sample_step = time_sample1 - time_sample0
                time_scalars['sampling'] = time_sample_step

            # Save checkpoints
            if self.ckpt_enabled and (epoch % self.ckpt_freq == 0):
                time_ckpt0 = time.perf_counter()
                self.save_checkpoint(epoch)
                time_ckpt1 = time.perf_counter()
                time_ckpt_step = time_ckpt1 - time_ckpt0
                time_scalars['checkpoint'] = time_ckpt_step

            # Log metrics
            if needs_metrics:

                t_metric0 = time.perf_counter()

                if self.metric_kl_enabled:
                    t_kl0 = time.perf_counter()
                    self.eval_kl_ite(epoch)
                    t_kl1 = time.perf_counter()

                    time_kl_step = t_kl1 - t_kl0
                    time_scalars['kl_estimation'] = time_kl_step

                if self.metric_w2_enabled:
                    t_w2_0 = time.perf_counter()
                    self.eval_w2(epoch)
                    t_w2_1 = time.perf_counter()

                    time_w2_step = t_w2_1 - t_w2_0
                    time_scalars['w2_estimation'] = time_w2_step

                if self.metric_elbo_enabled:
                    # Evaluate ELBO
                    t_elbo0 = time.perf_counter()
                    self.eval_elbo(epoch)
                    t_elbo1 = time.perf_counter()

                    time_scalars['elbo_estimation'] = t_elbo1 - t_elbo0

                if self.metric_expected_log_marginal_enabled:
                    t_elm0 = time.perf_counter()
                    self.eval_expected_log_marginal(epoch)
                    t_elm1 = time.perf_counter()

                    time_scalars[
                        'expected_log_marginal_estimation'] = t_elm1 - t_elm0

                if self.metric_mmd_enabled:
                    t_mmd0 = time.perf_counter()
                    self.eval_mmd(epoch)
                    t_mmd1 = time.perf_counter()

                    time_scalars['mmd_estimation'] = t_mmd1 - t_mmd0

                if self.metric_ksd_enabled:
                    t_ksd0 = time.perf_counter()
                    self.eval_ksd(epoch)
                    t_ksd1 = time.perf_counter()

                    time_scalars['ksd_estimation'] = t_ksd1 - t_ksd0

                if self.metric_bnn_enabled:
                    t_bnn0 = time.perf_counter()
                    self.eval_bnn(epoch)
                    t_bnn1 = time.perf_counter()

                    time_scalars['bnn_estimation'] = t_bnn1 - t_bnn0

                if self.metric_fisher_enabled:
                    t_fisher0 = time.perf_counter()
                    self.eval_fisher(epoch)
                    t_fisher1 = time.perf_counter()

                    time_scalars['fisher_estimation'] = t_fisher1 - t_fisher0

                if self.metric_jacobian_spectral_enabled:
                    t_jspec0 = time.perf_counter()
                    self.eval_jacobian_spectral(epoch)
                    t_jspec1 = time.perf_counter()

                    time_scalars['jacobian_spectral_estimation'] = (
                        t_jspec1 - t_jspec0)

                t_metric1 = time.perf_counter()
                time_metric_step = t_metric1 - t_metric0
                time_scalars['metric_eval_tot'] = time_metric_step

            # Generate and save contour plots
            if needs_plot:
                t_plot0 = time.perf_counter()
                _, z_plot = self.vi_model.sampling(num=self.plot_num)

                try:
                    self.target_model.contour_plot(
                        self.config.target.bbox,
                        fnet=None,
                        samples=z_plot.cpu().numpy(),
                        save_to_path=os.path.join(
                            self.plot_save_path,
                            f"contour_epoch_{epoch}.png",
                        ),
                        quiver=False,
                        t=epoch,
                    )
                    logger.debug(f"Saved contour plot at epoch {epoch}.")
                except Exception as e:
                    self.target_model.trace_plot(
                        z_plot,
                        figpath=self.plot_save_path,
                        figname=f"trace_epoch_{epoch}.png",
                        figtitle=f"Trace Plot at Epoch {epoch}",
                    )
                    logger.debug(f"Saved trace plot at epoch {epoch}.")

                t_plot1 = time.perf_counter()
                time_plot_step = t_plot1 - t_plot0
                time_scalars['plot'] = time_plot_step

            # EMA: restore training params
            if _ema_swapped:
                self.ema.restore(self.vi_model.parameters())

            epoch_end_time = time.perf_counter()
            epoch_time = epoch_end_time - epoch_start_time
            time_scalars['epoch'] = epoch_time

            # Update time history and log
            for key, value in time_scalars.items():
                self.time_history[key].append((value, epoch))
                self.writer.add_scalar(f"time/{key}", value, epoch)

                while self.time_history[key][0][
                        1] <= epoch - self.train_time_avg_window:
                    self.time_history[key].pop(0)

                avg_val = sum(val for val, _ in self.time_history[key]) / min(
                    self.train_time_avg_window, epoch)
                self.writer.add_scalar(f"time_avg/{key}", avg_val, epoch)

        # Close writer at end
        total_time = time.perf_counter() - self.train_start_time
        avg_epoch_time = total_time / max(1, self.training_num_epochs)
        logger.info(
            f"Training completed. Total time: {total_time:.3f}s, Avg epoch time: {avg_epoch_time:.6f}s"
        )
        self.writer.add_scalar("summary/total_training_time", total_time, 0)
        self.writer.add_scalar("summary/avg_epoch_time", avg_epoch_time, 0)
        self.writer.close()
