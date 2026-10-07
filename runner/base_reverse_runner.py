import torch
from models.reverse_model import ReverseModel, BaseReverseConditionalModel
import os
from utils.logging import get_logger
from typing import Callable
import ite
import time
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from runner.base_runner import BaseSIVIRunner
from utils.metrics import compute_sliced_wasserstein

logger = get_logger()


class BaseReverseConditionalRunner(BaseSIVIRunner):
    '''
    Base class for SIVI runners that use a reverse conditional model q_psi(epsilon|z).
    Extends BaseSIVIRunner by adding reverse model training and log q_phi(z) estimation.

    Args:
        config (DictConfig): Configuration for the experiment.
        name(str): Name of the Runner.
    '''

    def __init__(
        self,
        config: DictConfig,
        name: str = "BaseReverseConditionalRunner",
    ):
        assert name != "BaseReverseConditionalRunner", "Please use a subclass of BaseReverseConditionalRunner."
        super().__init__(config=config, name=name)

        # reverse model config
        self.reverse_model_type: str = self.config.reverse_model_type
        logger.info(f"Reverse model type: {self.reverse_model_type}")

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
            'reverse_model': OmegaConf.load(reverse_model_config_path)
        }
        self.config = OmegaConf.merge(
            _reverse_model_config,
            self.config,
        )  # type: ignore

        # The reverse conditional must model q_psi(epsilon | z) for the
        # instantiated VI family.  Some legacy reverse configs interpolate
        # dimensions from the target, which breaks AISIVI variants that use a
        # lower-dimensional semi-implicit epsilon.
        self.config.reverse_model.z_dim = self.z_dim
        self.config.reverse_model.epsilon_dim = self.epsilon_dim

        # construct reverse model seperately to adopt direct gradient estimation
        self.reverse_model = self._get_reverse_model()
        self.reverse_model.to(self.device)

        # Enable reverse model training
        self.reverse_train = True

        # whether to normalize the score output of reverse model to zero mean
        self.normalize_reverse_score = self.training_cfg.setdefault(
            'normalize_reverse_score', False)
        self.training_sample_reverse_loss = 0.0

        # Load warmup config
        self.warmup_cfg = self.config.reverse_model['warmup']
        self.warmup_enabled = self.warmup_cfg['enabled']
        self.warmup_batch_size = self.warmup_cfg['batch_size']
        self.warmup_epochs = self.warmup_cfg['epochs']

        # Load warmup accumulators
        self.warmup_sample_loss = 0.0
        self.warmup_last_time = 0.0
        self.warmup_steps = 0
        self.warmup_start_time = 0.0

        # Load training configs
        self.training_reverse_sample_num = self.training_cfg.setdefault(
            'reverse_sample_num', -1)
        self.training_reverse_log_freq = self.training_cfg['log'][
            'reverse_log_freq']

        # Load reverse training configs
        self.rev_train_cfg = self.training_cfg['reverse']
        self.reverse_lr = self.rev_train_cfg['lr']
        self.reverse_weight_decay = self.rev_train_cfg.get(
            'weight_decay',
            self.config.reverse_model.get('weight_decay', 0.0),
        )
        self.rev_batch_size = self.rev_train_cfg['batch_size']
        self.rev_epochs = self.rev_train_cfg['epochs']
        self.rev_update_freq = self.rev_train_cfg['update_freq']
        self.share_scheduler = self.rev_train_cfg.setdefault(
            'share_scheduler', False)

        # Instantiating reverse model optimizer
        if 'use_optimizer' not in self.config.reverse_model:
            logger.warning(
                "'use_optimizer' not specified in reverse_model config; defaulting to True."
            )
            self.config.reverse_model['use_optimizer'] = True

        if not self.config.reverse_model['use_optimizer']:
            logger.info(
                "Reverse model optimizer disabled. Using `fit()` method instead."
            )
            self.training_reverse_optimizer = None
            self.training_reverse_scheduler = None
        else:
            self.training_reverse_optimizer = torch.optim.Adam(
                self.reverse_model.parameters(),
                lr=self.reverse_lr,
                weight_decay=self.reverse_weight_decay,
            )

            self.training_reverse_scheduler = None
            rev_sched_cfg = self.rev_train_cfg.get('scheduler', None)
            if rev_sched_cfg is not None:
                assert rev_sched_cfg['type'] == 'StepLR', \
                    "Only StepLR scheduler is supported for reverse optimizer."
                self.training_reverse_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.training_reverse_optimizer,
                    step_size=rev_sched_cfg['step_size'],
                    gamma=rev_sched_cfg['gamma'],
                )
            elif self.share_scheduler:
                logger.info(
                    "Sharing VI model scheduler config for reverse model training."
                )
                self.training_reverse_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.training_reverse_optimizer,
                    step_size=(self.vi_scheduler_cfg['step_size'] *
                               self.rev_epochs) // self.rev_update_freq,
                    gamma=self.vi_scheduler_cfg['gamma'],
                )

    def _get_reverse_model(self) -> BaseReverseConditionalModel:
        '''
        Construct the reverse model based on the specified type and configuration.

        Returns:
            reverse_model (BaseReverseConditionalModel): Instantiated reverse model.
        '''
        if self.reverse_model_type not in ReverseModel:
            raise ValueError(
                f"Unsupported reverse model type: {self.reverse_model_type}")
        reverse_model = ReverseModel[self.reverse_model_type](
            config=self.config.reverse_model)
        return reverse_model

    def _training_rev_log_func(
        self,
        loss: float,
        steps: int,
        epoch_inner: int,
        epoch_outer: int,
    ):
        """
        Logging hook for reverse model training across inner/outer epochs.
        Args:
            loss (float): Loss value for the current inner epoch.
            steps (int): Number of steps taken in the current inner epoch.
            epoch_inner (int): Current inner epoch number.
            epoch_outer (int): Current outer epoch number.
        """
        epoch = (epoch_outer - 1) * self.rev_epochs + epoch_inner
        self.training_sample_reverse_loss += loss
        self.training_steps += steps
        if self.training_reverse_log_freq and self.training_reverse_log_freq > 0 and epoch % self.training_reverse_log_freq == 0:
            avg_loss = self.training_sample_reverse_loss / self.training_reverse_log_freq
            self.training_sample_reverse_loss = 0.0
            avg_steps = self.training_steps / self.training_reverse_log_freq
            self.training_steps = 0
            logger.debug(
                f"Epoch {epoch_outer}, Inner epoch {epoch_inner}, Reverse Model Loss: {avg_loss:.4f}, Avg Steps: {avg_steps:.4f}"
            )
        self.writer.add_scalar("train/reverse_model/loss", loss, epoch)
        self.writer.add_scalar("train/reverse_model/steps", steps, epoch)

    def calculate_rev_KL(self) -> float:
        '''
        Calculate the KL divergence between the true joint distribution and the
        joint distribution induced by the reverse model using the ITE package.
        Returns:
            kl_div (float): Estimated KL divergence value.
        '''
        if self.reverse_model is None:
            raise RuntimeError(
                "KL calculation is only available for reverse_uivi where reverse_model exists."
            )

        # Sample from true joint
        true_eps, true_z = self.vi_model.sampling(num=self.n_ite_samples)
        true_joint = torch.cat([true_eps, true_z], dim=1).cpu().numpy()

        # Sample epsilon from reverse model given true z
        _, rev_true_z = self.vi_model.sampling(num=self.n_ite_samples)
        with torch.no_grad():
            generated_z, generated_eps, _ = self.reverse_model.sample(
                rev_true_z, num_samples=1)
            generated_z = generated_z.reshape(-1, self.z_dim)
            generated_eps = generated_eps.reshape(-1, self.epsilon_dim)
        generated_joint = torch.cat([generated_eps, generated_z],
                                    dim=-1).cpu().numpy()

        # Estimate KL divergence using ITE
        cost_obj = ite.cost.BDKL_KnnK()
        kl_div = cost_obj.estimation(generated_joint, true_joint)

        return kl_div

    def calculate_rev_W2(self) -> float:
        '''
        Calculate the W2 distance between the true joint distribution and the
        joint distribution induced by the reverse model.
        Returns:
            w2 (float): Estimated W2 distance.
        '''
        if self.reverse_model is None:
            raise RuntimeError(
                "W2 calculation is only available for reverse_uivi where reverse_model exists."
            )

        # Sample from true joint
        true_eps, true_z = self.vi_model.sampling(num=self.n_w2_samples)
        true_joint = torch.cat([true_eps, true_z], dim=1)

        # Sample epsilon from reverse model given true z
        _, rev_true_z = self.vi_model.sampling(num=self.n_w2_samples)
        with torch.no_grad():
            generated_z, generated_eps, _ = self.reverse_model.sample(
                rev_true_z, num_samples=1)
            generated_z = generated_z.reshape(-1, self.z_dim)
            generated_eps = generated_eps.reshape(-1, self.epsilon_dim)
        generated_joint = torch.cat([generated_eps, generated_z], dim=-1)

        return compute_sliced_wasserstein(
            generated_joint,
            true_joint,
            num_projections=self.n_w2_projections,
            device=self.device)

    def eval_kl_ite(self, epoch: int):
        """
        Evaluate KL divergence between VI and baseline using ITE and log to TensorBoard. Also evaluate the KL divergence between the true joint distribution and the joint distribution induced by the reverse model.
        Args:
            epoch (int): Current epoch number.
        """
        super().eval_kl_ite(epoch)
        try:
            rev_kl_div = self.calculate_rev_KL()
        except RuntimeError as exc:
            logger.warning(
                f"Reverse model KL ITE failed at epoch {epoch}: {exc}. "
                "Logging NaN and continuing."
            )
            rev_kl_div = float("nan")
        self.writer.add_scalar("metric/reverse_model/kl_ite", rev_kl_div, epoch)
        logger.debug(f"Epoch {epoch}, Reverse Model KL ITE: {rev_kl_div:.4f}")

    def eval_w2(self, epoch: int):
        """
        Evaluate W2 distance between VI and baseline using sliced wasserstein and log to TensorBoard. Also evaluate the W2 distance between the true joint distribution and the joint distribution induced by the reverse model.
        Args:
            epoch (int): Current epoch number.
        """
        super().eval_w2(epoch)
        try:
            rev_w2 = self.calculate_rev_W2()
        except RuntimeError as exc:
            logger.warning(
                f"Reverse model W2 failed at epoch {epoch}: {exc}. "
                "Logging NaN and continuing."
            )
            rev_w2 = float("nan")
        self.writer.add_scalar("metric/reverse_model/w2", rev_w2, epoch)
        logger.debug(f"Epoch {epoch}, Reverse Model W2: {rev_w2:.4f}")

    def _train_reverse_model(
        self,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epochs: int,
        batch_size: int,
        initialize: bool = False,
        progress_bar: bool = True,
        log_func: Callable[[float, int, int], None] | None = None,
    ) -> None:
        '''
        Train the reverse model. Generate samples from the VI model and then optimize the reverse model to maximize the log-reverse-probablity of these samples.
        
        Args:
            optimizer (torch.optim.Optimizer | None): Optimizer for the reverse model. If None, the reverse model's own fit() method will be used.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            initialize (bool, optional): Whether to re-initialize parameters before fitting. Defaults to False.
            progress_bar (bool, optional): Whether to display a progress bar. Defaults to True.
            log_func (Callable[[float, int, int], None], optional): Function to log training progress. Defaults to None. The first argument is the loss, the second is the steps, and the third is the epoch.
        Returns:
            None
        '''
        iterator = range(1, epochs + 1)
        if progress_bar:
            iterator = tqdm(iterator, desc="Reverse Model Training")
        # Two modes:
        # - Flow-based (ConditionalRealNVP): optimize log-prob.
        # - GaussianReverse: call fit() using current VI samples.
        if optimizer is not None:
            self.reverse_model.train()
            for epoch in iterator:
                epsilon_samples, z_samples = self.vi_model.sampling(
                    num=batch_size)
                optimizer.zero_grad()
                log_prob = self.reverse_model.log_prob(
                    epsilon_samples,
                    z_samples,
                )
                loss = -torch.mean(log_prob)
                if torch.isfinite(loss):
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.reverse_model.parameters(),
                            max_norm=self.grad_clip)
                    optimizer.step()
                    if self.curr_epoch > 1 and scheduler is not None:
                        scheduler.step()
                else:
                    logger.warning(
                        f"NaN or Inf detected in reverse model loss at epoch {self.curr_epoch}. Skipping update."
                    )
                if log_func is not None:
                    log_func(loss.item(), 1, epoch)
        else:
            epsilon_samples, z_samples = self.vi_model.sampling(num=batch_size)
            nll, steps = self.reverse_model.fit(
                epsilon_samples,
                z_samples,
                initialize=initialize,
            )
            if log_func is not None:
                log_func(nll, steps, epochs)

    def _warmup_log_func(self, loss: float, steps: int, epoch: int) -> None:
        '''
        Logging hook for reverse model warmup.
        Args:
            loss (float): Loss value for the current epoch.
            steps (int): Number of steps taken in the current epoch.
            epoch (int): Current epoch number.
        '''
        self.writer.add_scalar("warmup/reverse_model_loss", loss, epoch)
        self.warmup_sample_loss += loss
        self.warmup_steps += steps
        if (self.training_metric_log_freq
                and self.training_metric_log_freq > 0
                and epoch % self.training_metric_log_freq == 0):

            if self.metric_kl_enabled:
                kl_div = self.calculate_rev_KL()
                self.writer.add_scalar("warmup/kl_div", kl_div, epoch)
                logger.debug(
                    f"Warmup Epoch {epoch}, KL Divergence: {kl_div:.4f}")

            if self.metric_w2_enabled:
                w2_dist = self.calculate_rev_W2()
                self.writer.add_scalar("warmup/w2_dist", w2_dist, epoch)
                logger.debug(f"Warmup Epoch {epoch}, W2 Dist: {w2_dist:.4f}")

            if self.metric_ksd_enabled:
                rev_ksd, _ = self.calculate_rev_KSD()
                self.writer.add_scalar("warmup/rev_model_ksd", rev_ksd, epoch)
                logger.debug(
                    f"Warmup Epoch {epoch}, Rev KSD: {rev_ksd:.4f}")

            if self.metric_fisher_enabled:
                fisher_val = self.evaluate_fisher_divergence()
                self.writer.add_scalar("warmup/fisher_div", fisher_val, epoch)
                logger.debug(
                    f"Warmup Epoch {epoch}, Fisher Div: {fisher_val:.4f}")

        if epoch % self.training_loss_log_freq == 0:
            avg_loss = self.warmup_sample_loss / self.training_loss_log_freq
            current_time = time.perf_counter()
            avg_step_time = (current_time - self.warmup_last_time
                             ) / self.training_loss_log_freq
            self.warmup_last_time = current_time
            self.warmup_sample_loss = 0.0
            avg_steps = self.warmup_steps / self.training_loss_log_freq
            logger.debug(
                f"Warmup Epoch {epoch}, Average Reverse Model Loss: {avg_loss:.4f}, Avg Step Time: {avg_step_time:.4f}s, Avg Steps: {avg_steps:.4f}"
            )

    def warmup(self) -> None:
        '''
        Warm up the reverse model by training it for a specified number of epochs. Logs the KL divergence and reverse model loss during warmup. Configured via the 'warmup' section in the configuration dictionary.
        
        Returns:
            None
        '''
        if self.reverse_model is None:
            return
        if not self.warmup_enabled:
            return
        logger.info("Starting reverse model warmup...")

        # For flow model, use optimizer; for GaussianReverse, no optimizer.
        optimizer = None
        if self.config.reverse_model['use_optimizer']:
            lr = self.warmup_cfg['lr']
            optimizer = torch.optim.Adam(
                self.reverse_model.parameters(),
                lr=lr,
                weight_decay=self.warmup_cfg.get(
                    'weight_decay',
                    self.reverse_weight_decay,
                ),
            )

        self.warmup_sample_loss = 0.0

        self.warmup_start_time = time.perf_counter()
        self.warmup_last_time = time.perf_counter()

        self._train_reverse_model(
            optimizer=optimizer,
            scheduler=None,
            epochs=self.warmup_epochs,
            batch_size=self.warmup_batch_size,
            initialize=True,
            progress_bar=True,
            log_func=self._warmup_log_func,
        )
        warmup_end_time = time.perf_counter()
        warmup_time = warmup_end_time - self.warmup_start_time
        logger.info(
            f"Warmup completed for {self.warmup_epochs} epochs. Total time: {warmup_time:.3f}s, Avg epoch time: {warmup_time/self.warmup_epochs:.6f}s"
        )
        self.writer.add_scalar("summary/warmup_time", warmup_time, 0)
        self.writer.add_scalar(
            "summary/warmup_avg_epoch_time",
            warmup_time / self.warmup_epochs,
            0,
        )

    def save_checkpoint(self, epoch: int):
        '''
        Save the state dict of model and optimizer to checkpoints at the given epoch.
        Args:
            epoch (int): Current epoch number.
        '''
        super().save_checkpoint(epoch)
        epoch_ckpt_dir = os.path.join(self.ckpt_base_path, f"epoch_{epoch}")

        rev_ckpt_path = os.path.join(
            epoch_ckpt_dir,
            "reverse_model.pt",
        )
        torch.save(self.reverse_model.state_dict(), rev_ckpt_path)
        if self.training_reverse_optimizer is not None:
            rev_opt_path = os.path.join(
                epoch_ckpt_dir,
                "reverse_optim.pt",
            )
            torch.save(
                self.training_reverse_optimizer.state_dict(),
                rev_opt_path,
            )
            if self.training_reverse_scheduler is not None:
                rev_sched_path = os.path.join(
                    epoch_ckpt_dir,
                    "reverse_sched.pt",
                )
                torch.save(
                    self.training_reverse_scheduler.state_dict(),
                    rev_sched_path,
                )
        logger.debug(
            f"Saved reverse checkpoints at epoch {epoch} to {epoch_ckpt_dir}.")

    def load_checkpoints(self):
        '''
        Load model state dicts from checkpoint directory when resuming training. Use default initialization if checkpoint files are missing. Will try to load optimizer and scheduler states if available.
        '''
        super().load_checkpoints()
        logger.debug("Trying to load reverse model checkpoints...")
        # Reverse model checkpoint
        ckpt_dir = self.config.resume.ckpt_dir
        rev_ckpt_path = os.path.join(
            ckpt_dir,
            'reverse_model.pt',
        )
        try:
            if os.path.isfile(rev_ckpt_path):
                state = torch.load(
                    rev_ckpt_path,
                    map_location=self.device,
                )
                self.reverse_model.load_state_dict(state)
                logger.info(
                    f"Loaded reverse model checkpoint from {rev_ckpt_path}")
            else:
                logger.warning(
                    f"Reverse model checkpoint not found at {rev_ckpt_path}; using default initialization."
                )
        except Exception as e:
            logger.error(
                f"Failed to load reverse model checkpoints from {ckpt_dir}: {e}."
            )
            raise e

        if not self.config.resume.get('load_optimizer', False):
            return

        # load reverse optimizer only optimizer is not None
        if self.training_reverse_optimizer is not None:
            try:
                rev_opt_path = os.path.join(
                    ckpt_dir,
                    'reverse_optim.pt',
                )
                if os.path.isfile(rev_opt_path):
                    ro_state = torch.load(
                        rev_opt_path,
                        map_location=self.device,
                    )
                    self.training_reverse_optimizer.load_state_dict(ro_state)
                    logger.info(
                        f"Loaded reverse optimizer from {rev_opt_path}")
                else:
                    logger.warning(
                        f"Reverse optimizer checkpoint not found at {rev_opt_path}; using fresh optimizer."
                    )
            except Exception as e:
                logger.error(f"Failed to load reverse optimizer: {e}.")
                raise e

        if self.training_reverse_scheduler is not None:
            try:
                rev_sched_path = os.path.join(
                    ckpt_dir,
                    'reverse_sched.pt',
                )
                if os.path.isfile(rev_sched_path):
                    rs_state = torch.load(
                        rev_sched_path,
                        map_location=self.device,
                    )
                    self.training_reverse_scheduler.load_state_dict(rs_state)
                    logger.info(
                        f"Loaded reverse scheduler from {rev_sched_path}")
                else:
                    logger.warning(
                        f"Reverse scheduler checkpoint not found at {rev_sched_path}; using fresh scheduler."
                    )
            except Exception as e:
                logger.error(f"Failed to load reverse scheduler: {e}.")
                raise e

    def train_reverse_model(self, epoch_outer: int):
        '''
        Train the reverse model for several inner epochs using samples from the current VI.
        Args:
            epoch_outer (int): Current outer epoch number.
        '''
        if epoch_outer % self.rev_update_freq != 0:
            return
        self._train_reverse_model(
            optimizer=self.training_reverse_optimizer,
            scheduler=self.training_reverse_scheduler,
            epochs=self.rev_epochs,
            batch_size=self.rev_batch_size,
            initialize=False,
            progress_bar=False,
            log_func=lambda loss, steps, epoch: self._training_rev_log_func(
                loss,
                steps,
                epoch,
                epoch_outer,
            ),
        )

    def learn(self):
        '''
        Run the full training procedure for UIVI with reverse conditional model.

        Returns:
            None
        '''
        if self.resume:
            self.load_checkpoints()
            self.resume = False  # only load once
        elif not getattr(self, "_vi_pretrained_done", False):
            self.pretrain_vi()

        # Warmup reverse model if enabled
        if self.warmup_enabled:
            self.warmup()

        # Proceed with main UIVI learning
        super().learn()
