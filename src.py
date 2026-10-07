import torch
import os
import argparse
from utils.logging import get_logger
from omegaconf import OmegaConf, DictConfig
from runner.runners import Runners

logger = get_logger()

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VI Runner")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to main YAML config file",
    )

    parser.add_argument(
        'overrides',
        nargs='*',
        help='Override .yaml config options with key=value pairs',
    )

    cli_args = parser.parse_args()

    main_cfg: DictConfig = OmegaConf.load(cli_args.config)  # type: ignore
    overrides_cfg = OmegaConf.from_dotlist(cli_args.overrides)
    main_cfg = OmegaConf.merge(main_cfg, overrides_cfg)  # type: ignore
    main_cfg.config_path = cli_args.config

    seed = main_cfg.get('seed', 42)

    # Set CUDA_VISIBLE_DEVICES before any torch.cuda calls
    if main_cfg.get('use_cuda', False):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(main_cfg.get(
            'cuda_visible_devices',
            '0',
        ))

    use_cuda = main_cfg.use_cuda and torch.cuda.is_available()
    if use_cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
    else:
        device = 'cpu'

    torch.manual_seed(seed)
    main_cfg.device = device

    runner_type = main_cfg.runner_type

    runner = Runners[runner_type](config=main_cfg)
    runner.log_config()
    runner.learn()
