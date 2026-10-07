"""Data loading utilities for data-dependent target distributions.

Loads pre-processed ``.pt`` files created by ``prepare_data.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_waveform(
    data_dir: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-processed Waveform dataset.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``train.pt`` and ``test.pt``.
        Defaults to ``<project_root>/data/waveform/``.
    device : torch.device
        Target device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(X_train, y_train, X_test, y_test)`` on *device*.
        The prepared files are generated from the official KSIVI
        ``waveform.mat`` split.
    """
    d = Path(data_dir) if data_dir else _DATA_DIR / "waveform"
    train = torch.load(os.path.join(d, "train.pt"), map_location=device, weights_only=True)
    test = torch.load(os.path.join(d, "test.pt"), map_location=device, weights_only=True)
    return train["X"], train["y"], test["X"], test["y"]


def load_boston(
    data_dir: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...] | None,
]:
    """Load pre-processed Boston Housing dataset.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``train.pt`` and ``test.pt``.
        Defaults to ``<project_root>/data/boston/``.
    device : torch.device
        Target device for returned tensors.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test, mean_y_train, std_y_train)``
        on *device*.  ``y_train`` is standardised; ``y_test`` is in original
        scale.
    """
    d = Path(data_dir) if data_dir else _DATA_DIR / "boston"
    train = torch.load(os.path.join(d, "train.pt"), map_location=device, weights_only=True)
    test = torch.load(os.path.join(d, "test.pt"), map_location=device, weights_only=True)
    return (
        train["X"],
        train["y"],
        test["X"],
        test["y"],
        train["mean_y"],
        train["std_y"],
        (
            train["X_dev"],
            train["y_dev"],
            train["mean_y"],
            train["std_y"],
        ) if "X_dev" in train and "y_dev" in train else None,
    )


def load_bnn_regression(
    name: str,
    data_dir: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...] | None,
]:
    """Load a pre-processed UCI BNN regression dataset.

    Parameters
    ----------
    name : str
        Dataset name, e.g. ``"concrete"``, ``"power"``, ``"protein"``,
        ``"winered"``, ``"yacht"``.  Expects ``data/<name>/train.pt`` and
        ``data/<name>/test.pt`` prepared by ``prepare_data.py``.
    data_dir : str or Path, optional
        Override the directory.  Defaults to ``<project_root>/data/<name>/``.
    device : torch.device
        Target device for returned tensors.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test, mean_y_train, std_y_train)``
        on *device*.  ``y_train`` is standardised; ``y_test`` is in original
        scale.
    """
    d = Path(data_dir) if data_dir else _DATA_DIR / name
    train = torch.load(os.path.join(d, "train.pt"), map_location=device, weights_only=True)
    test = torch.load(os.path.join(d, "test.pt"), map_location=device, weights_only=True)
    return (
        train["X"],
        train["y"],
        test["X"],
        test["y"],
        train["mean_y"],
        train["std_y"],
        (
            train["X_dev"],
            train["y_dev"],
            train["mean_y"],
            train["std_y"],
        ) if "X_dev" in train and "y_dev" in train else None,
    )
