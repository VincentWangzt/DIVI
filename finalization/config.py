from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "finalization" / "default_config_grid.yaml"


def load_config(config_path: Path | str | None, overrides: list[str] | None = None) -> DictConfig:
    path = DEFAULT_CONFIG if config_path is None else Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)

