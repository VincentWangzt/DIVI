from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import scipy.stats as st
import torch
from omegaconf import OmegaConf

from models.target_models import target_distribution

from .artifacts import (
    RunRecord,
    find_final_samples,
    load_baseline_samples,
    load_grad_norm_series,
    load_kl_ite_series,
    load_sample_z,
    load_weight_norm_series,
    normalize_target,
    run_index,
)
from .config import REPO_ROOT, repo_path
from .eval_assumption_jacobian import (
    evaluate_run as evaluate_m_eps_run,
    write_csv as write_m_eps_csv,
)
from .eval_score_fourth_moment import (
    evaluate_run as evaluate_score_4th_run,
    write_csv as write_score_4th_csv,
)
from .eval_vi_fourth_moment import (
    evaluate_run as evaluate_vi_4th_run,
    write_csv as write_vi_4th_csv,
)


def _display_method(name: str) -> str:
    """Normalize method names for legends/titles."""
    if name.upper() == "DIVI":
        return "DIVI"
    return name


def _target_bbox(target: str) -> list[float] | None:
    path = REPO_ROOT / "configs" / "targets" / f"{target}.yaml"
    if not path.exists():
        return None
    cfg = OmegaConf.load(path)
    bbox = cfg.get("bbox")
    return list(bbox) if bbox is not None else None


def _take_points(samples: torch.Tensor, count: int) -> np.ndarray:
    if samples.shape[0] > count:
        samples = samples[torch.randperm(samples.shape[0])[:count]]
    return samples.detach().cpu().numpy()


def _toy_logp_grid(target: str, bbox: list[float], grid_size: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_model = target_distribution[target](device="cpu")
    xx, yy = np.mgrid[bbox[0]:bbox[1]:complex(grid_size), bbox[2]:bbox[3]:complex(grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    with torch.no_grad():
        logp = target_model.logp(torch.as_tensor(positions.T, dtype=torch.float32))
    logp_grid = logp.detach().cpu().numpy().reshape(xx.shape)
    return xx, yy, logp_grid


def _draw_toy_contours(ax: plt.Axes, target: str, bbox: list[float]) -> None:
    xx, yy, logp_grid = _toy_logp_grid(target, bbox)
    with np.errstate(divide="ignore", invalid="ignore"):
        density_surface = -np.log(-logp_grid)
    if not np.isfinite(density_surface).any():
        density_surface = logp_grid
    ax.contourf(xx, yy, density_surface, cmap="Blues", alpha=0.8, levels=11)
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))


def _draw_target_line_contours(
    ax: plt.Axes,
    target: str,
    bbox: list[float],
    *,
    grid_size: int,
    num_levels: int,
    linewidth: float,
) -> None:
    xx, yy, logp_grid = _toy_logp_grid(target, bbox, grid_size)
    finite = logp_grid[np.isfinite(logp_grid)]
    if finite.size == 0:
        return
    levels = np.unique(np.quantile(finite, np.linspace(0.28, 0.94, num_levels)))
    if levels.size < 2:
        return
    ax.contour(
        xx,
        yy,
        logp_grid,
        levels=levels,
        colors="#2f2f2f",
        linewidths=linewidth,
        linestyles="solid",
    )


def _draw_sample_hist2d(
    ax: plt.Axes,
    points: np.ndarray,
    bbox: list[float],
    *,
    bins: int,
    alpha: float,
) -> None:
    hist, x_edges, y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
        density=True,
    )
    hist = np.ma.masked_where(hist <= 0, hist)
    vmax = float(hist.max()) if hist.count() else 1.0
    ax.pcolormesh(
        x_edges,
        y_edges,
        hist.T,
        cmap="Blues",
        norm=Normalize(vmin=0.0, vmax=vmax),
        alpha=alpha,
        shading="auto",
        rasterized=True,
    )


def _load_plot_samples(column: str, target: str, seed: int, idx: dict[tuple[int, str, str], RunRecord], count: int, cfg: Any, plot_cfg: Any) -> torch.Tensor:
    if _is_truth_column(column):
        return load_baseline_samples(target)

    rec = idx[(seed, column.upper(), target)]
    sample_path, _ = find_final_samples(rec.result_path)
    samples = load_sample_z(sample_path)
    if samples.shape[0] >= count:
        return samples

    try:
        from .runner_eval import _sample_vi, build_runner, prepare_config, remove_file_handlers

        runner_cfg = prepare_config(
            rec,
            device=str(cfg.evaluation.device),
            scratch_results=str(cfg.campaign.scratch_results_dir),
            scratch_tb=str(cfg.campaign.scratch_tb_dir),
        )
        runner, _ckpt_dir, _epoch = build_runner(rec, runner_cfg)
        try:
            return _sample_vi(runner, count, int(plot_cfg.get("sample_batch_size", 10000)))
        finally:
            if hasattr(runner, "writer"):
                runner.writer.close()
            remove_file_handlers()
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception:
        return samples


def _method_seed(cfg: Any, method: str) -> int:
    """Return the seed for *method*, applying any per-method override."""
    default = int(cfg.selection.seed_for_figures)
    overrides = cfg.selection.get("seed_overrides", {})
    return int(overrides.get(method.upper(), default))


def render_scatter_grid(records: list[RunRecord], cfg: Any) -> Path:
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [normalize_target(str(target)) for target in cfg.selection.scatter_targets]
    configured_columns = [str(method) for method in cfg.selection.scatter_methods]
    columns = configured_columns if any(_is_truth_column(column) for column in configured_columns) else configured_columns + ["GroundTruth"]
    idx = run_index(records)
    num_points = int(cfg.plots.scatter.num_points)
    panel_w, panel_h = [float(x) for x in cfg.plots.scatter.figsize_per_panel]
    title_fontsize = int(cfg.plots.scatter.get("title_fontsize", 12))
    label_fontsize = int(cfg.plots.scatter.get("label_fontsize", 12))
    w_pad = float(cfg.plots.scatter.get("w_pad", 0.8))
    h_pad = float(cfg.plots.scatter.get("h_pad", 0.35))

    fig, axes = plt.subplots(
        len(targets),
        len(columns),
        figsize=(panel_w * len(columns), panel_h * len(targets)),
        squeeze=False,
    )
    for row_idx, target in enumerate(targets):
        bbox = _target_bbox(target)
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(_scatter_column_label(column), fontsize=title_fontsize)
            if col_idx == 0:
                ax.set_ylabel(target, fontsize=label_fontsize)
            try:
                if bbox is not None:
                    _draw_toy_contours(ax, target, bbox)
                if _is_truth_column(column):
                    samples = load_baseline_samples(target)
                else:
                    seed = _method_seed(cfg, column)
                    rec = idx[(seed, column.upper(), target)]
                    sample_path, _ = find_final_samples(rec.result_path)
                    samples = load_sample_z(sample_path)
                points = _take_points(samples[:, :2], num_points)
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    ".",
                    markersize=float(cfg.plots.scatter.point_size),
                    color="#ff7f0e",
                    alpha=float(cfg.plots.scatter.alpha),
                )
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"missing\n{type(exc).__name__}", ha="center", va="center", fontsize=7)
            if bbox is not None:
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.tick_params(axis="both", labelsize=7, length=2, width=0.5)
            if col_idx != 0:
                ax.tick_params(labelleft=False)
    fig.tight_layout(pad=0.35, w_pad=w_pad, h_pad=h_pad)
    png_path = out_dir / "toy_scatter_grid.png"
    pdf_path = out_dir / "toy_scatter_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_scatter_hist_grid(records: list[RunRecord], cfg: Any) -> Path:
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [normalize_target(str(target)) for target in cfg.selection.scatter_targets]
    configured_columns = [str(method) for method in cfg.selection.scatter_methods]
    columns = configured_columns if any(_is_truth_column(column) for column in configured_columns) else configured_columns + ["GroundTruth"]
    idx = run_index(records)
    plot_cfg = cfg.plots.get("scatter_hist", cfg.plots.scatter)
    num_points = int(plot_cfg.get("num_points", cfg.plots.scatter.num_points))
    bins = int(plot_cfg.get("bins", 70))
    contour_grid_size = int(plot_cfg.get("contour_grid_size", 160))
    contour_levels = int(plot_cfg.get("contour_levels", 5))
    contour_linewidth = float(plot_cfg.get("contour_linewidth", 0.75))
    hist_alpha = float(plot_cfg.get("hist_alpha", 0.78))
    panel_w, panel_h = [float(x) for x in plot_cfg.get("figsize_per_panel", cfg.plots.scatter.figsize_per_panel)]
    title_fontsize = int(plot_cfg.get("title_fontsize", cfg.plots.scatter.get("title_fontsize", 12)))
    label_fontsize = int(plot_cfg.get("label_fontsize", cfg.plots.scatter.get("label_fontsize", 12)))
    w_pad = float(plot_cfg.get("w_pad", cfg.plots.scatter.get("w_pad", 0.8)))
    h_pad = float(plot_cfg.get("h_pad", cfg.plots.scatter.get("h_pad", 0.35)))

    fig, axes = plt.subplots(
        len(targets),
        len(columns),
        figsize=(panel_w * len(columns), panel_h * len(targets)),
        squeeze=False,
    )
    for row_idx, target in enumerate(targets):
        bbox = _target_bbox(target)
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(_scatter_column_label(column), fontsize=title_fontsize)
            if col_idx == 0:
                ax.set_ylabel(target, fontsize=label_fontsize)
            try:
                if bbox is None:
                    raise ValueError(f"No bbox configured for {target}")
                seed = _method_seed(cfg, column)
                samples = _load_plot_samples(column, target, seed, idx, num_points, cfg, plot_cfg)
                points = _take_points(samples[:, :2], num_points)
                _draw_sample_hist2d(ax, points, bbox, bins=bins, alpha=hist_alpha)
                _draw_target_line_contours(
                    ax,
                    target,
                    bbox,
                    grid_size=contour_grid_size,
                    num_levels=contour_levels,
                    linewidth=contour_linewidth,
                )
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"missing\n{type(exc).__name__}", ha="center", va="center", fontsize=7)
            if bbox is not None:
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
                ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
            ax.tick_params(axis="both", labelsize=7, length=2, width=0.5)
            if col_idx != 0:
                ax.tick_params(labelleft=False)
    fig.tight_layout(pad=0.35, w_pad=w_pad, h_pad=h_pad)
    png_path = out_dir / "toy_scatter_hist_grid.png"
    pdf_path = out_dir / "toy_scatter_hist_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def _is_truth_column(column: str) -> bool:
    normalized = column.replace("_", "").replace(" ", "").lower()
    return normalized in {"truth", "groundtruth", "groundtruthdistribution"}


def _scatter_column_label(column: str) -> str:
    return "GroundTruth" if _is_truth_column(column) else _display_method(column.upper())


def _langevin_target(device: str = "cpu"):
    return target_distribution["Langevin_post"](device=device)


def _trace_stats(samples: torch.Tensor, target_model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_np = samples.detach().cpu().numpy()
    u_mean = u_np.mean(0)
    low, high = st.t.interval(0.95, len(u_mean), loc=u_mean, scale=np.std(u_np, 0))
    true_path = target_model.u.detach().cpu().numpy().flatten()
    obs = target_model.data.detach().cpu().numpy().reshape(-1)
    t = np.arange(target_model.dt, target_model.T + target_model.dt, target_model.dt)
    obs_t = np.arange(target_model.T / target_model.num_obs, target_model.T + target_model.T / target_model.num_obs, target_model.T / target_model.num_obs)
    return t, true_path, u_mean, low, high, np.vstack([obs_t[: obs.shape[0]], obs]).T


def langevin_panel_labels(methods: list[str], available_methods: set[str] | None = None) -> list[str]:
    available = None if available_methods is None else {method.upper() for method in available_methods}
    method_labels = [method.upper() for method in methods if available is None or method.upper() in available]
    labels = ["SGLD"] + [method for method in method_labels if method != "DIVI"]
    if "DIVI" not in method_labels:
        return labels
    cols = math.ceil((len(labels) + 1) / 2)
    labels.insert(cols, "DIVI")
    return labels


def render_langevin_trace_grid(records: list[RunRecord], cfg: Any) -> Path:
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.selection.seed_for_figures)
    methods = [str(method).upper() for method in cfg.selection.methods]
    idx = run_index(records)
    target_model = _langevin_target()
    num_samples = int(cfg.plots.langevin_trace.num_samples)

    available = {rec.method.upper() for rec in records if rec.seed == seed and rec.target == "Langevin_post"}
    panel_labels = langevin_panel_labels(methods, available)

    panels: list[tuple[str, torch.Tensor]] = []
    for method in panel_labels:
        if method == "SGLD":
            panels.append(("SGLD", load_baseline_samples("Langevin_post")[:num_samples]))
            continue
        rec = idx.get((seed, method, "Langevin_post"))
        if rec is None:
            continue
        sample_path, _ = find_final_samples(rec.result_path)
        panels.append((method, load_sample_z(sample_path)[:num_samples]))

    if not panels:
        raise RuntimeError("No Langevin_post panels available.")

    stats = [(label, _trace_stats(samples, target_model)) for label, samples in panels]
    y_values = []
    for _, (_, true_path, mean, low, high, obs_points) in stats:
        y_values.extend([true_path, mean, low, high, obs_points[:, 1]])
    y_min = min(float(np.nanmin(values)) for values in y_values)
    y_max = max(float(np.nanmax(values)) for values in y_values)
    margin = 0.05 * max(y_max - y_min, 1.0)

    cols = math.ceil(len(panels) / 2)
    panel_w, panel_h = [float(x) for x in cfg.plots.langevin_trace.figsize_per_panel]
    fig, axes = plt.subplots(2, cols, figsize=(panel_w * cols, panel_h * 2), squeeze=False, sharey=True)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for i, (ax, (label, (t, true_path, mean, low, high, obs_points))) in enumerate(zip(axes.ravel(), stats)):
        first = i == 0
        ax.plot(t, true_path, color="magenta", linewidth=1.0, label="true path" if first else None)
        ax.plot(t, mean, color="blue", linewidth=1.0, label="sample path" if first else None)
        ax.plot(t, low, color="black", linewidth=0.6)
        ax.plot(t, high, color="black", linewidth=0.6)
        ax.fill_between(t, low, high, facecolor="aqua", alpha=0.3, label="confidence interval" if first else None)
        ax.scatter(obs_points[:, 0], obs_points[:, 1], color="red", marker=".", linewidth=0.5, s=10, label="observation" if first else None)
        ax.set_title(_display_method(label), fontsize=10)
        ax.grid(True, linewidth=0.3)
        ax.set_ylim(y_min - margin, y_max + margin)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = out_dir / "langevin_trace_grid.png"
    pdf_path = out_dir / "langevin_trace_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# KL convergence curve plots
# ---------------------------------------------------------------------------

_TARGET_DISPLAY_NAMES: dict[str, str] = {
    "Langevin_post": "Conditioned Diffusion",
}


def _target_display_name(target: str) -> str:
    """Return a human-friendly display name for *target*, falling back to the raw name."""
    return _TARGET_DISPLAY_NAMES.get(target, target)


_KL_METHOD_COLORS: dict[str, str] = {
    "UIVI": "#1f77b4",   # blue
    "AISIVI": "#ff7f0e", # orange
    "DIVI": "#2ca02c",   # green
    "KSIVI": "#d62728",  # red
    "SIVI": "#9467bd",   # purple
}


def _method_color(method: str, idx: int = 0) -> str:
    """Return a stable color for *method*, falling back to the tab10 cycle."""
    key = method.upper()
    if key in _KL_METHOD_COLORS:
        return _KL_METHOD_COLORS[key]
    display = _display_method(key)
    if display in _KL_METHOD_COLORS:
        return _KL_METHOD_COLORS[display]
    tab10 = plt.cm.tab10.colors  # type: ignore[attr-defined]
    return tab10[idx % len(tab10)]


def _aggregate_curves(
    seed_curves: list[tuple[np.ndarray, np.ndarray]],
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed curves to a common grid, return (grid, mean, se).

    The common x-range is the intersection of all seeds' [min, max] ranges
    so that ``np.interp`` never extrapolates.
    """
    if not seed_curves:
        return None
    global_min = max(float(xs.min()) for xs, _ in seed_curves)
    global_max = min(float(xs.max()) for xs, _ in seed_curves)
    if global_min >= global_max:
        return None
    grid = np.linspace(global_min, global_max, n_grid)
    interpolated = np.empty((len(seed_curves), n_grid), dtype=np.float64)
    for i, (xs, ys) in enumerate(seed_curves):
        interpolated[i] = np.interp(grid, xs, ys)
    mean = interpolated.mean(axis=0)
    se = (
        interpolated.std(axis=0, ddof=1) / np.sqrt(len(seed_curves))
        if len(seed_curves) > 1
        else np.zeros_like(mean)
    )
    return grid, mean, se


def _aggregate_curves_minmax(
    seed_curves: list[tuple[np.ndarray, np.ndarray]],
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed curves to a common grid, return (grid, mean, min, max).

    The common x-range is the intersection of all seeds' [min, max] ranges
    so that ``np.interp`` never extrapolates.
    """
    if not seed_curves:
        return None
    global_min = max(float(xs.min()) for xs, _ in seed_curves)
    global_max = min(float(xs.max()) for xs, _ in seed_curves)
    if global_min >= global_max:
        return None
    grid = np.linspace(global_min, global_max, n_grid)
    interpolated = np.empty((len(seed_curves), n_grid), dtype=np.float64)
    for i, (xs, ys) in enumerate(seed_curves):
        interpolated[i] = np.interp(grid, xs, ys)
    mean = interpolated.mean(axis=0)
    min_vals = interpolated.min(axis=0)
    max_vals = interpolated.max(axis=0)
    return grid, mean, min_vals, max_vals


def _collect_kl_curves(
    records: list[RunRecord],
    targets: list[str],
    methods: list[str],
    x_mode: str,
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load KL_ITE series for all (method, target) pairs grouped by seed.

    Args:
        x_mode: ``"step"`` uses iteration number; ``"time"`` uses seconds elapsed.

    Returns:
        Mapping of ``(method_upper, target)`` to list of ``(x, y)`` seed curves.
    """
    method_set = {m.upper() for m in methods}
    target_set = set(targets)

    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for rec in records:
        mu = rec.method.upper()
        if mu not in method_set or rec.target not in target_set:
            continue
        loaded = load_kl_ite_series(rec)
        if loaded is None:
            continue
        steps, wall_times, values = loaded
        if x_mode == "step":
            x = steps
        else:
            x = wall_times - wall_times[0]
        key = (mu, rec.target)
        curves.setdefault(key, []).append((x, values))
    return curves


def _collect_grad_norm_curves(
    records: list[RunRecord],
    targets: list[str],
    methods: list[str],
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load gradient-norm series for all (method, target) pairs grouped by seed.

    Returns:
        Mapping of ``(method_upper, target)`` to list of ``(steps, values)`` seed curves.
    """
    method_set = {m.upper() for m in methods}
    target_set = set(targets)

    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for rec in records:
        mu = rec.method.upper()
        if mu not in method_set or rec.target not in target_set:
            continue
        loaded = load_grad_norm_series(rec)
        if loaded is None:
            continue
        steps, _wall_times, values = loaded
        key = (mu, rec.target)
        curves.setdefault(key, []).append((steps, values))
    return curves


def render_kl_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render KL(p, q_theta) vs iteration — one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.kl_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.kl_curve_targets]
    methods = [str(m) for m in cfg.selection.kl_curve_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    all_curves = _collect_kl_curves(records, targets, methods, x_mode="step")

    n_cols = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(target, fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(r"$\mathrm{KL}(p,\, q_\theta)$", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, se = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label=_display_method(mu))
            ax.fill_between(grid, mean - se, mean + se, color=color, alpha=band_alpha)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "kl_iteration_grid.png"
    pdf_path = out_dir / "kl_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_kl_time_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render KL(p, q_theta) vs wall-clock time — one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.kl_time
    targets = [normalize_target(str(t)) for t in cfg.selection.kl_curve_targets]
    methods = [str(m) for m in cfg.selection.kl_curve_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    all_curves = _collect_kl_curves(records, targets, methods, x_mode="time")

    n_cols = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(target, fontsize=title_fontsize)
        ax.set_xlabel("Time (s)", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(r"$\mathrm{KL}(p,\, q_\theta)$", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, se = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label=_display_method(mu))
            ax.fill_between(grid, mean - se, mean + se, color=color, alpha=band_alpha)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "kl_time_grid.png"
    pdf_path = out_dir / "kl_time_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_grad_norm_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render gradient norm vs iteration -- one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.grad_norm_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.grad_norm_targets]
    methods = [str(m) for m in cfg.selection.grad_norm_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))
    log_scale = bool(plot_cfg.get("log_scale", True))

    all_curves = _collect_grad_norm_curves(records, targets, methods)

    n_cols = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel("Gradient Norm", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="Mean")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha, label="Min/Max")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
            ax.yaxis.get_minor_formatter().set_scientific(False)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "grad_norm_iteration_grid.png"
    pdf_path = out_dir / "grad_norm_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Weight-norm convergence curve plot
# ---------------------------------------------------------------------------


def _collect_weight_norm_curves(
    records: list[RunRecord],
    targets: list[str],
    methods: list[str],
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load weight-norm series for all (method, target) pairs grouped by seed.

    Returns:
        Mapping of ``(method_upper, target)`` to list of ``(steps, values)`` seed curves.
    """
    method_set = {m.upper() for m in methods}
    target_set = set(targets)

    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for rec in records:
        mu = rec.method.upper()
        if mu not in method_set or rec.target not in target_set:
            continue
        loaded = load_weight_norm_series(rec)
        if loaded is None:
            continue
        steps, _wall_times, values = loaded
        key = (mu, rec.target)
        curves.setdefault(key, []).append((steps, values))
    return curves


def render_weight_norm_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render weight norm vs iteration -- one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.weight_norm_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.weight_norm_targets]
    methods = [str(m) for m in cfg.selection.weight_norm_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))
    log_scale = bool(plot_cfg.get("log_scale", False))

    all_curves = _collect_weight_norm_curves(records, targets, methods)

    n_cols = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel("Weight Norm", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="Mean")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha, label="Min/Max")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
            ax.yaxis.get_minor_formatter().set_scientific(False)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "weight_norm_iteration_grid.png"
    pdf_path = out_dir / "weight_norm_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# VI Derivative Fourth Moment vs iteration plot
# ---------------------------------------------------------------------------


def _evaluate_m_eps(
    records: list[RunRecord],
    targets: list[str],
    cfg: Any,
) -> Path:
    """Compute M_eps at every checkpoint for all matching DIVI runs.

    Writes results to ``{output_dir}/m_eps_results.csv``.  When the CSV
    already exists and ``evaluation.m_eps.overwrite`` is false the
    computation is skipped.

    Returns:
        Path to the results CSV.
    """
    import csv as _csv

    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    csv_path = root / "m_eps_results.csv"

    m_eps_cfg = cfg.evaluation.get("m_eps", {})
    overwrite = bool(m_eps_cfg.get("overwrite", False))
    if csv_path.exists() and not overwrite:
        return csv_path

    n_samples_default = int(m_eps_cfg.get("n_samples", 1024))
    n_samples_override = dict(m_eps_cfg.get("n_samples_override", {}))
    checkpoint_stride = int(m_eps_cfg.get("checkpoint_stride", 1))
    device = str(cfg.evaluation.get("device", "auto"))

    target_set = {normalize_target(t) for t in targets}
    divi_records = [
        rec for rec in records
        if rec.method.upper() == "DIVI" and rec.target in target_set
    ]

    all_results: list[dict[str, Any]] = []
    for rec in divi_records:
        try:
            n_samples = int(n_samples_override.get(rec.target, n_samples_default))
            run_results = evaluate_m_eps_run(
                rec,
                device=device,
                n_samples=n_samples,
                checkpoint_stride=checkpoint_stride,
            )
            all_results.extend(run_results)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Skipping M_eps evaluation for %s: %s", rec.run_id, exc,
            )

    if all_results:
        write_m_eps_csv(all_results, csv_path)
    return csv_path


def _collect_m_eps_curves(
    csv_path: Path,
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load M_eps CSV and build per-seed (epoch, M_eps) curves.

    Returns:
        Mapping of ``("DIVI", target)`` to list of ``(epochs, values)``
        seed curves, matching the shape expected by
        :func:`_aggregate_curves_minmax`.
    """
    import csv as _csv

    if not csv_path.exists():
        return {}

    # Group rows by (target, seed)
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            key = (str(row["target"]), int(row["seed"]))
            grouped.setdefault(key, []).append(
                (int(row["epoch"]), float(row["M_eps"]))
            )

    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for (target, _seed), points in grouped.items():
        points.sort(key=lambda p: p[0])
        epochs = np.array([p[0] for p in points], dtype=np.float64)
        values = np.array([p[1] for p in points], dtype=np.float64)
        curves.setdefault(("DIVI", target), []).append((epochs, values))
    return curves


def render_m_eps_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render M_eps vs iteration -- one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.m_eps_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.m_eps_targets]
    methods = [str(m) for m in cfg.selection.m_eps_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate (or load cached CSV)
    csv_path = _evaluate_m_eps(records, targets, cfg)
    all_curves = _collect_m_eps_curves(csv_path)

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel("Derivative Fourth Moment", fontsize=label_fontsize - 2)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="Mean")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha, label="Min/Max")
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "m_eps_iteration_grid.png"
    pdf_path = out_dir / "m_eps_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# VI Fourth Moment vs iteration plot
# ---------------------------------------------------------------------------


def _evaluate_vi_fourth_moment(
    records: list[RunRecord],
    targets: list[str],
    cfg: Any,
) -> Path:
    """Compute VI output fourth moments at every checkpoint for all matching DIVI runs.

    Writes results to ``{output_dir}/vi_fourth_moment_results.csv``.  When the CSV
    already exists and ``evaluation.vi_fourth_moment.overwrite`` is false the
    computation is skipped.

    Returns:
        Path to the results CSV.
    """
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    csv_path = root / "vi_fourth_moment_results.csv"

    vi_4th_cfg = cfg.evaluation.get("vi_fourth_moment", {})
    overwrite = bool(vi_4th_cfg.get("overwrite", False))
    if csv_path.exists() and not overwrite:
        return csv_path

    n_samples_default = int(vi_4th_cfg.get("n_samples", 1024))
    checkpoint_stride = int(vi_4th_cfg.get("checkpoint_stride", 1))
    device = str(cfg.evaluation.get("device", "auto"))

    target_set = {normalize_target(t) for t in targets}
    divi_records = [
        rec for rec in records
        if rec.method.upper() == "DIVI" and rec.target in target_set
    ]

    all_results: list[dict[str, Any]] = []
    for rec in divi_records:
        try:
            run_results = evaluate_vi_4th_run(
                rec,
                device=device,
                n_samples=n_samples_default,
                checkpoint_stride=checkpoint_stride,
            )
            all_results.extend(run_results)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Skipping VI fourth moment evaluation for %s: %s", rec.run_id, exc,
            )

    if all_results:
        write_vi_4th_csv(all_results, csv_path)
    return csv_path


def _collect_vi_fourth_moment_curves(
    csv_path: Path,
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load VI fourth moment CSV and build per-seed (epoch, vi_fourth_moment) curves.

    Returns:
        Mapping of ``("DIVI", target)`` to list of ``(epochs, values)``
        seed curves, matching the shape expected by
        :func:`_aggregate_curves_minmax`.
    """
    import csv as _csv

    if not csv_path.exists():
        return {}

    # Group rows by (target, seed)
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            key = (str(row["target"]), int(row["seed"]))
            grouped.setdefault(key, []).append(
                (int(row["epoch"]), float(row["vi_fourth_moment"]))
            )

    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for (target, _seed), points in grouped.items():
        points.sort(key=lambda p: p[0])
        epochs = np.array([p[0] for p in points], dtype=np.float64)
        values = np.array([p[1] for p in points], dtype=np.float64)
        curves.setdefault(("DIVI", target), []).append((epochs, values))
    return curves


def render_vi_fourth_moment_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render VI fourth moment vs iteration -- one subplot per target."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.vi_fourth_moment_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.vi_fourth_moment_targets]
    methods = [str(m) for m in cfg.selection.vi_fourth_moment_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate (or load cached CSV)
    csv_path = _evaluate_vi_fourth_moment(records, targets, cfg)
    all_curves = _collect_vi_fourth_moment_curves(csv_path)

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel("VI Fourth Moment", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="VI Fourth Moment")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "vi_fourth_moment_iteration_grid.png"
    pdf_path = out_dir / "vi_fourth_moment_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Score Fourth Moment vs iteration plots
# ---------------------------------------------------------------------------


def _evaluate_score_4th_moment(
    records: list[RunRecord],
    targets: list[str],
    cfg: Any,
) -> Path:
    """Compute score fourth moments at every checkpoint for all matching DIVI runs.

    Writes results to ``{output_dir}/score_4th_moment_results.csv``.  When the CSV
    already exists and ``evaluation.score_4th_moment.overwrite`` is false the
    computation is skipped.

    Returns:
        Path to the results CSV.
    """
    import csv as _csv

    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    csv_path = root / "score_4th_moment_results.csv"

    s4_cfg = cfg.evaluation.get("score_4th_moment", {})
    overwrite = bool(s4_cfg.get("overwrite", False))
    if csv_path.exists() and not overwrite:
        return csv_path

    n_samples = int(s4_cfg.get("n_samples", 10240))
    checkpoint_stride = int(s4_cfg.get("checkpoint_stride", 1))
    device = str(cfg.evaluation.get("device", "auto"))

    target_set = {normalize_target(t) for t in targets}
    divi_records = [
        rec for rec in records
        if rec.method.upper() == "DIVI" and rec.target in target_set
    ]

    all_results: list[dict[str, Any]] = []
    for rec in divi_records:
        try:
            run_results = evaluate_score_4th_run(
                rec,
                device=device,
                n_samples=n_samples,
                checkpoint_stride=checkpoint_stride,
            )
            all_results.extend(run_results)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Skipping score 4th moment evaluation for %s: %s", rec.run_id, exc,
            )

    if all_results:
        write_score_4th_csv(all_results, csv_path)
    return csv_path


def _collect_score_4th_moment_curves(
    csv_path: Path,
) -> tuple[
    dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
    dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
    dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
]:
    """Load score 4th moment CSV and build per-seed curves.

    Returns:
        (score_p_curves, score_q_curves, score_diff_curves) -- each mapping
        ``("DIVI", target)`` to list of ``(epochs, values)`` seed curves,
        matching the shape expected by :func:`_aggregate_curves_minmax`.
    """
    import csv as _csv

    if not csv_path.exists():
        return {}, {}, {}

    # Group rows by (target, seed)
    grouped: dict[tuple[str, int], list[tuple[int, float, float, float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            key = (str(row["target"]), int(row["seed"]))
            score_diff = float(row["score_diff_l2_fourth"]) if "score_diff_l2_fourth" in row else 0.0
            grouped.setdefault(key, []).append(
                (int(row["epoch"]), float(row["score_p_4th_moment"]), float(row["score_q_4th_moment"]), score_diff)
            )

    score_p_curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    score_q_curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    score_diff_curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for (target, _seed), points in grouped.items():
        points.sort(key=lambda p: p[0])
        epochs = np.array([p[0] for p in points], dtype=np.float64)
        p_values = np.array([p[1] for p in points], dtype=np.float64)
        q_values = np.array([p[2] for p in points], dtype=np.float64)
        diff_values = np.array([p[3] for p in points], dtype=np.float64)
        score_p_curves.setdefault(("DIVI", target), []).append((epochs, p_values))
        score_q_curves.setdefault(("DIVI", target), []).append((epochs, q_values))
        score_diff_curves.setdefault(("DIVI", target), []).append((epochs, diff_values))
    return score_p_curves, score_q_curves, score_diff_curves


def render_score_p_4th_moment_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render E[||score_p||^4] vs iteration -- one subplot per target."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.score_4th_moment_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.score_4th_moment_targets]
    methods = [str(m) for m in cfg.selection.score_4th_moment_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate (or load cached CSV)
    csv_path = _evaluate_score_4th_moment(records, targets, cfg)
    score_p_curves, _, _ = _collect_score_4th_moment_curves(csv_path)

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(r"$E[\|\nabla_z \log p(z)\|^4]$", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = score_p_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="Mean")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha, label="Min/Max")
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "score_p_4th_moment_iteration_grid.png"
    pdf_path = out_dir / "score_p_4th_moment_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_score_q_4th_moment_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render E[||score_q||^4] vs iteration -- one subplot per target."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.score_4th_moment_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.score_4th_moment_targets]
    methods = [str(m) for m in cfg.selection.score_4th_moment_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate (or load cached CSV)
    csv_path = _evaluate_score_4th_moment(records, targets, cfg)
    _, score_q_curves, _ = _collect_score_4th_moment_curves(csv_path)

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(r"$E[\|\psi(z)\|^4]$", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = score_q_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="Mean")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha, label="Min/Max")
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "score_q_4th_moment_iteration_grid.png"
    pdf_path = out_dir / "score_q_4th_moment_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_score_diff_l2_fourth_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render E[||score_p - score_q||_2^4] vs iteration -- one subplot per target."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.score_4th_moment_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.score_4th_moment_targets]
    methods = [str(m) for m in cfg.selection.score_4th_moment_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate (or load cached CSV)
    csv_path = _evaluate_score_4th_moment(records, targets, cfg)
    _, _, score_diff_curves = _collect_score_4th_moment_curves(csv_path)

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(r"$E[\|\nabla_z \log p - \psi(z)\|^4]$", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = score_diff_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="Mean")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha, label="Min/Max")
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "score_diff_l2_fourth_iteration_grid.png"
    pdf_path = out_dir / "score_diff_l2_fourth_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Score Linearity scatter plot
# ---------------------------------------------------------------------------

_SCORE_LINEARITY_EPOCH_COLORS: dict[int, str] = {
    2000: "#1f77b4",   # blue
    5000: "#ff7f0e",   # orange
    10000: "#2ca02c",  # green
}


def render_score_linearity_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render score-linearity scatter: log||diff|| - log(||z||+1) vs ||z||.

    One subplot per target, colored by checkpoint epoch.  If the linear bound
    holds, the y-values should form a bounded horizontal band.
    """
    import csv as _csv

    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.score_linearity_grid
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    alpha = float(plot_cfg.get("alpha", 0.2))
    marker_size = float(plot_cfg.get("marker_size", 6))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate if needed
    from .eval_score_linearity import evaluate_score_linearity

    csv_path = evaluate_score_linearity(records, cfg)

    # Load CSV
    if not csv_path.exists():
        raise RuntimeError(f"Score linearity CSV not found at {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))

    if not rows:
        raise RuntimeError("Score linearity CSV is empty.")

    # Organize data by target
    targets = sorted({row["target"] for row in rows})
    epochs = sorted({int(row["epoch"]) for row in rows})

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel(r"$\|z\|$", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(
                r"$\log\|\nabla_z \log p - s_\psi\| - \log(\|z\| + 1)$",
                fontsize=label_fontsize,
            )

        target_rows = [r for r in rows if r["target"] == target]

        for epoch in epochs:
            epoch_rows = [r for r in target_rows if int(r["epoch"]) == epoch]
            if not epoch_rows:
                continue

            z_norms = np.array([float(r["z_norm"]) for r in epoch_rows])
            log_ratios = np.array([float(r["log_ratio"]) for r in epoch_rows])
            color = _SCORE_LINEARITY_EPOCH_COLORS.get(epoch, "gray")

            ax.scatter(
                z_norms,
                log_ratios,
                c=color,
                s=marker_size,
                alpha=alpha,
                marker=".",
                label=f"Epoch {epoch}",
                edgecolors="none",
                rasterized=True,
            )

            # Median horizontal line
            median_val = float(np.median(log_ratios))
            ax.axhline(
                median_val, color=color, linestyle="--",
                linewidth=1.0, alpha=0.7,
            )

        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        ax.legend(fontsize=11, loc="upper right", framealpha=0.8, markerscale=4)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "score_linearity_grid.png"
    pdf_path = out_dir / "score_linearity_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Score Norm Linearity scatter plot
# ---------------------------------------------------------------------------


def render_score_norm_linearity_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render score-norm linearity scatter: log max(||score_gt||,||score_proxy||) - log(||z||+1) vs ||z||.

    One subplot per target, colored by checkpoint epoch.  If both score
    functions have at most linear growth, the y-values should form a bounded
    horizontal band.
    """
    import csv as _csv

    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.score_norm_linearity_grid
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    alpha = float(plot_cfg.get("alpha", 0.2))
    marker_size = float(plot_cfg.get("marker_size", 6))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate if needed
    from .eval_score_norm_linearity import evaluate_score_norm_linearity

    csv_path = evaluate_score_norm_linearity(records, cfg)

    # Load CSV
    if not csv_path.exists():
        raise RuntimeError(f"Score norm linearity CSV not found at {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))

    if not rows:
        raise RuntimeError("Score norm linearity CSV is empty.")

    # Organize data by target
    targets = sorted({row["target"] for row in rows})
    epochs = sorted({int(row["epoch"]) for row in rows})

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel(r"$\|z\|$", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(
                r"$\log\max(\|\nabla_z \log p\|, \|s_\psi\|) - \log(\|z\| + 1)$",
                fontsize=label_fontsize,
            )

        target_rows = [r for r in rows if r["target"] == target]

        for epoch in epochs:
            epoch_rows = [r for r in target_rows if int(r["epoch"]) == epoch]
            if not epoch_rows:
                continue

            z_norms = np.array([float(r["z_norm"]) for r in epoch_rows])
            log_ratios = np.array([float(r["log_ratio"]) for r in epoch_rows])
            color = _SCORE_LINEARITY_EPOCH_COLORS.get(epoch, "gray")

            ax.scatter(
                z_norms,
                log_ratios,
                c=color,
                s=marker_size,
                alpha=alpha,
                marker=".",
                label=f"Epoch {epoch}",
                edgecolors="none",
                rasterized=True,
            )

            # Median horizontal line
            median_val = float(np.median(log_ratios))
            ax.axhline(
                median_val, color=color, linestyle="--",
                linewidth=1.0, alpha=0.7,
            )

        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        ax.legend(fontsize=11, loc="upper right", framealpha=0.8, markerscale=4)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "score_norm_linearity_grid.png"
    pdf_path = out_dir / "score_norm_linearity_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Score Norm Linearity (Uniform z) scatter plot
# ---------------------------------------------------------------------------


def render_score_norm_linearity_uniform_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render score-norm linearity scatter with uniform-sphere z sampling.

    Same metric as render_score_norm_linearity_grid but z is sampled with
    uniform radius and uniform direction on the sphere, giving unbiased
    coverage across all norm values.

    One subplot per target, colored by checkpoint epoch.
    """
    import csv as _csv

    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.score_norm_linearity_uniform_grid
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    alpha = float(plot_cfg.get("alpha", 0.2))
    marker_size = float(plot_cfg.get("marker_size", 6))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    # Evaluate if needed
    from .eval_score_norm_linearity_uniform import evaluate_score_norm_linearity_uniform

    csv_path = evaluate_score_norm_linearity_uniform(records, cfg)

    # Load CSV
    if not csv_path.exists():
        raise RuntimeError(f"Score norm linearity uniform CSV not found at {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))

    if not rows:
        raise RuntimeError("Score norm linearity uniform CSV is empty.")

    # Organize data by target
    targets = sorted({row["target"] for row in rows})
    epochs = sorted({int(row["epoch"]) for row in rows})

    n_cols = max(len(targets), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(_target_display_name(target), fontsize=title_fontsize)
        ax.set_xlabel(r"$\|z\|$", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(
                r"$\log\max(\|\nabla_z \log p\|, \|s_\psi\|) - \log(\|z\| + 1)$",
                fontsize=label_fontsize,
            )

        target_rows = [r for r in rows if r["target"] == target]

        for epoch in epochs:
            epoch_rows = [r for r in target_rows if int(r["epoch"]) == epoch]
            if not epoch_rows:
                continue

            z_norms = np.array([float(r["z_norm"]) for r in epoch_rows])
            log_ratios = np.array([float(r["log_ratio"]) for r in epoch_rows])
            color = _SCORE_LINEARITY_EPOCH_COLORS.get(epoch, "gray")

            ax.scatter(
                z_norms,
                log_ratios,
                c=color,
                s=marker_size,
                alpha=alpha,
                marker=".",
                label=f"Epoch {epoch}",
                edgecolors="none",
                rasterized=True,
            )

            # Median horizontal line
            median_val = float(np.median(log_ratios))
            ax.axhline(
                median_val, color=color, linestyle="--",
                linewidth=1.0, alpha=0.7,
            )

        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        legend_loc = "lower right" if target == "x_shaped" else "upper right"
        ax.legend(fontsize=11, loc=legend_loc, framealpha=0.8, markerscale=4)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "score_norm_linearity_uniform_grid.png"
    pdf_path = out_dir / "score_norm_linearity_uniform_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path
