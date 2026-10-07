"""One-time data preprocessing script.

Prepares ``.pt`` files under ``data/`` for data-dependent targets:

- **Waveform**: official KSIVI waveform split loaded from ``waveform.mat``.
- **Boston Housing**: 13 features regression.
- **Concrete**: UCI Concrete Compressive Strength (8 features, N=1030).
- **Power**: UCI Combined Cycle Power Plant (4 features, N=9568).
- **Protein**: UCI Physicochemical Properties of Protein (9 features, N=45730).
- **Winered**: UCI Wine Quality Red (11 features, N=1599).
- **Yacht**: UCI Yacht Hydrodynamics (6 features, N=308).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.model_selection import train_test_split


_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

_WAVEFORM_PATH = _DATA_DIR / "waveform.mat"


def _find_waveform_mat() -> str:
    if _WAVEFORM_PATH.is_file():
        return os.fspath(_WAVEFORM_PATH)
    raise FileNotFoundError(
        f"Cannot find waveform.mat at {_WAVEFORM_PATH}"
    )


def prepare_waveform(out_dir: str = "data/waveform") -> None:
    """Load the official waveform split and save it as prepared ``.pt`` files."""
    os.makedirs(out_dir, exist_ok=True)

    src = _find_waveform_mat()
    data = scipy.io.loadmat(src)

    X_train = data["X_train"].astype(np.float64, copy=False)
    y_train = data["y_train"].reshape(-1).astype(np.float64, copy=False)
    X_test = data["X_test"].astype(np.float64, copy=False)
    y_test = data["y_test"].reshape(-1).astype(np.float64, copy=False)

    torch.save(
        {
            "X": torch.tensor(X_train, dtype=torch.float32),
            "y": torch.tensor(y_train, dtype=torch.float32),
        },
        os.path.join(out_dir, "train.pt"),
    )
    torch.save(
        {
            "X": torch.tensor(X_test, dtype=torch.float32),
            "y": torch.tensor(y_test, dtype=torch.float32),
        },
        os.path.join(out_dir, "test.pt"),
    )
    print(
        f"Waveform saved to {out_dir}/ from {src} "
        f"(train: {X_train.shape}, test: {X_test.shape})"
    )


# ---------------------------------------------------------------------------
# Boston Housing
# ---------------------------------------------------------------------------

_BOSTON_PATH = _DATA_DIR / "boston" / "boston_housing.txt"


def _find_boston_file() -> str:
    if _BOSTON_PATH.is_file():
        return os.fspath(_BOSTON_PATH)
    raise FileNotFoundError(
        f"Cannot find boston_housing.txt at {_BOSTON_PATH}"
    )


def prepare_boston(out_dir: str = "data/boston", seed: int = 42) -> None:
    """Load Boston Housing, split with official semantics, and save."""
    os.makedirs(out_dir, exist_ok=True)

    src = _find_boston_file()
    data = np.loadtxt(src)
    X = data[:, :-1]
    y = data[:, -1]
    _prepare_bnn_regression_arrays(
        X=X,
        y=y,
        out_dir=out_dir,
        seed=seed,
        name="Boston",
    )


# ---------------------------------------------------------------------------
# Generic UCI regression datasets
# ---------------------------------------------------------------------------


def _prepare_bnn_regression_arrays(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    seed: int = 42,
    name: str = "",
    dev_fraction: float = 0.1,
    dev_max_size: int = 500,
) -> None:
    """Split and standardize a BNN regression dataset using official semantics."""
    os.makedirs(out_dir, exist_ok=True)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=seed,
    )

    y_train_all = y_train_all[:, None]
    y_test = y_test[:, None]

    dev_size = min(int(np.round(dev_fraction * X_train_all.shape[0])), dev_max_size)
    if dev_size <= 0 or dev_size >= X_train_all.shape[0]:
        raise ValueError(f"Invalid dev_size={dev_size} for dataset {name or out_dir}")

    X_dev = X_train_all[-dev_size:]
    y_dev = y_train_all[-dev_size:]
    X_train = X_train_all[:-dev_size]
    y_train = y_train_all[:-dev_size]

    mu_x = X_train.mean(axis=0)
    std_x = X_train.std(axis=0) + 1e-8
    X_train_std = (X_train - mu_x) / std_x
    X_test_std = (X_test - mu_x) / std_x
    X_dev_std = (X_dev - mu_x) / std_x

    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-8
    y_train_norm = (y_train - mean_y) / std_y

    torch.save(
        {
            "X": torch.tensor(X_train_std, dtype=torch.float32),
            "y": torch.tensor(y_train_norm, dtype=torch.float32),
            "mean_y": torch.tensor(mean_y, dtype=torch.float32),
            "std_y": torch.tensor(std_y, dtype=torch.float32),
            "X_dev": torch.tensor(X_dev_std, dtype=torch.float32),
            "y_dev": torch.tensor(y_dev, dtype=torch.float32),
        },
        os.path.join(out_dir, "train.pt"),
    )
    torch.save(
        {
            "X": torch.tensor(X_test_std, dtype=torch.float32),
            "y": torch.tensor(y_test, dtype=torch.float32),
        },
        os.path.join(out_dir, "test.pt"),
    )

    label = name or out_dir
    print(
        f"{label} saved to {out_dir}/ "
        f"(train: {X_train_std.shape}, dev: {X_dev_std.shape}, test: {X_test_std.shape}, "
        f"mean_y={mean_y:.4f}, std_y={std_y:.4f})"
    )


def _prepare_regression_csv(
    src: str,
    out_dir: str,
    delimiter: str | None,
    feature_cols: list[int],
    target_col: int,
    seed: int = 42,
    name: str = "",
) -> None:
    """Load a UCI regression CSV, split with official semantics, and save .pt files."""
    data = np.loadtxt(src, delimiter=delimiter)
    X = data[:, feature_cols]
    y = data[:, target_col]
    _prepare_bnn_regression_arrays(
        X=X,
        y=y,
        out_dir=out_dir,
        seed=seed,
        name=name,
    )


def prepare_concrete(out_dir: str = "data/concrete", seed: int = 42) -> None:
    """UCI Concrete Compressive Strength: 8 features, target col 8, N=1030."""
    _prepare_regression_csv(
        src=os.fspath(_DATA_DIR / "Concrete_Data.csv"),
        out_dir=out_dir,
        delimiter=",",
        feature_cols=list(range(8)),
        target_col=8,
        seed=seed,
        name="Concrete",
    )


def prepare_power(out_dir: str = "data/power", seed: int = 42) -> None:
    """UCI Combined Cycle Power Plant: 4 features, target col 4, N=9568."""
    _prepare_regression_csv(
        src=os.fspath(_DATA_DIR / "power.csv"),
        out_dir=out_dir,
        delimiter=",",
        feature_cols=list(range(4)),
        target_col=4,
        seed=seed,
        name="Power",
    )


def prepare_protein(out_dir: str = "data/protein", seed: int = 42) -> None:
    """UCI Protein Physicochemical: 9 features, target col 9, N=45730."""
    _prepare_regression_csv(
        src=os.fspath(_DATA_DIR / "protein.csv"),
        out_dir=out_dir,
        delimiter=",",
        feature_cols=list(range(9)),
        target_col=9,
        seed=seed,
        name="Protein",
    )


def prepare_winered(out_dir: str = "data/winered", seed: int = 42) -> None:
    """UCI Wine Quality Red: 11 features, target col 11, N=1599."""
    _prepare_regression_csv(
        src=os.fspath(_DATA_DIR / "winered.csv"),
        out_dir=out_dir,
        delimiter=";",
        feature_cols=list(range(11)),
        target_col=11,
        seed=seed,
        name="Winered",
    )


def prepare_yacht(out_dir: str = "data/yacht", seed: int = 42) -> None:
    """UCI Yacht Hydrodynamics: 6 features, target col 6, N=308."""
    _prepare_regression_csv(
        src=os.fspath(_DATA_DIR / "yacht.csv"),
        out_dir=out_dir,
        delimiter=None,
        feature_cols=list(range(6)),
        target_col=6,
        seed=seed,
        name="Yacht",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_waveform()
    prepare_boston()
    prepare_concrete()
    prepare_power()
    prepare_protein()
    prepare_winered()
    prepare_yacht()
    print("Data preparation complete.")
