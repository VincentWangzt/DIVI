from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable
from tqdm import tqdm

import numpy as np
import torch
from omegaconf import OmegaConf

from runner.runners import Runners
from utils.elm import kde_expected_log_marginal, load_baseline_sample_store
from utils.logging import get_logger
from utils.metrics import compute_sliced_wasserstein

from .artifacts import RunRecord, find_final_checkpoint, load_baseline_samples
from .config import repo_path, REPO_ROOT


logger = get_logger()


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_finite(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def truncated_w2_metric_name(width: float) -> str:
    width_float = float(width)
    if width_float.is_integer():
        width_text = str(int(width_float))
    else:
        width_text = f"{width_float:g}".replace(".", "_")
    return f"w2_trunc_abs_{width_text}"


class ConstrainedW2SamplingFailure(RuntimeError):
    def __init__(self, *, source: str, width: float, max_draws: int, draws: int, accepted: int, needed: int):
        super().__init__(
            f"{source} sampling reached max_draws={max_draws} before collecting "
            f"{needed} accepted samples for width={width}; accepted={accepted}, draws={draws}"
        )
        self.source = source
        self.width = float(width)
        self.max_draws = int(max_draws)
        self.draws = int(draws)
        self.accepted = int(accepted)
        self.needed = int(needed)


def _campaign_summary_path(cfg: Any) -> Path:
    manifest_path = repo_path(str(cfg.campaign.manifest_path))
    assert manifest_path is not None
    return manifest_path.parent / "generated_reports" / "summary.csv"


def load_campaign_timing(cfg: Any) -> dict[str, dict[str, float]]:
    summary_path = _campaign_summary_path(cfg)
    if not summary_path.exists():
        return {}
    timing: dict[str, dict[str, float]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            run_id = row.get("run_id")
            if not run_id:
                continue
            values: dict[str, float] = {}
            for key in ("wall_clock_sec", "training_time_sec", "iterations"):
                parsed = _finite_float(row.get(key))
                if parsed is not None:
                    values[key] = parsed
            if values:
                timing[run_id] = values
    return timing


def augment_run_rows_with_campaign_timing(rows: list[dict[str, Any]], cfg: Any) -> list[dict[str, Any]]:
    timing = load_campaign_timing(cfg)
    if not timing:
        return rows
    augmented: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        run_id = str(next_row.get("run_id", ""))
        values = timing.get(run_id)
        if values is not None:
            wall_clock = values.get("wall_clock_sec")
            if wall_clock is not None:
                next_row["wall_clock_sec"] = wall_clock
                next_row["duration_sec"] = wall_clock
            training_time = values.get("training_time_sec")
            if training_time is not None:
                next_row["training_time_sec"] = training_time
            iterations = values.get("iterations")
            if iterations is not None:
                next_row["summary_iterations"] = iterations
                if next_row.get("checkpoint_epoch") in (None, ""):
                    next_row["checkpoint_epoch"] = iterations
        augmented.append(next_row)
    return augmented


def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_file_handlers() -> None:
    root_logger = get_logger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)


def prepare_config(rec: RunRecord, *, device: str, scratch_results: str, scratch_tb: str):
    cfg = OmegaConf.load(rec.config_path)
    extra_overrides = rec.entry.get("extra_overrides") or []
    if isinstance(extra_overrides, str):
        extra_overrides = [extra_overrides]
    if extra_overrides:
        cfg = OmegaConf.merge(
            cfg, OmegaConf.from_dotlist([str(item) for item in extra_overrides]),
        )
    cfg.seed = rec.seed
    cfg.config_path = rec.config_path.as_posix()
    if device == "cpu":
        resolved_device = "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        resolved_device = "cuda"
    else:
        resolved_device = "cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available() else "cpu"
    cfg.device = resolved_device
    cfg.use_cuda = resolved_device == "cuda"
    cfg.output = OmegaConf.merge(
        cfg.get("output", {}),
        {
            "results_dir": f"{scratch_results}/{rec.run_id}",
            "tb_dir": f"{scratch_tb}/{rec.run_id}",
        },
    )
    return cfg


def build_runner(rec: RunRecord, cfg: Any):
    ckpt_dir, epoch = find_final_checkpoint(rec.result_path)
    set_seed(rec.seed, cfg.device == "cuda")
    runner = Runners[rec.runner_type](config=cfg)
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    state = torch.load(ckpt_dir / "vi_model.pt", map_location=runner.device)
    runner.vi_model.load_state_dict(state)
    runner.vi_model.eval()
    return runner, ckpt_dir, epoch


def _sample_vi(runner, count: int, batch_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            current = min(batch_size, count - start)
            _, z = runner.vi_model.sampling(num=current)
            chunks.append(z.detach().cpu())
    return torch.cat(chunks, dim=0)


def _sample_target(runner, count: int, batch_size: int) -> torch.Tensor:
    if hasattr(runner.target_model, "sample"):
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, count, batch_size):
                current = min(batch_size, count - start)
                chunks.append(runner.target_model.sample(current).detach().cpu())
        return torch.cat(chunks, dim=0)
    baseline = load_baseline_samples(runner.target_type)
    if baseline.shape[0] < count:
        indices = torch.randint(0, baseline.shape[0], (count,))
        return baseline[indices]
    return baseline[torch.randperm(baseline.shape[0])[:count]]


def _sampling_failure_warning(
    *,
    failure: ConstrainedW2SamplingFailure,
    fallback_value: float,
    context: dict[str, Any] | None,
) -> str:
    prefix = ""
    if context:
        labels = [
            f"{key}={value}"
            for key, value in context.items()
            if value not in (None, "")
        ]
        if labels:
            prefix = ", ".join(labels) + ": "
    return (
        f"{prefix}sampling process failed for constrained W2 "
        f"(source={failure.source}, width={failure.width:g}, accepted={failure.accepted}/"
        f"{failure.needed}, draws={failure.draws}, max_draws={failure.max_draws}); "
        f"using fallback W2=edge length {fallback_value:g}."
    )


def constrained_w2(
    runner,
    width: float,
    cfg: Any,
    *,
    warning_callback: Callable[[str], None] | None = None,
    warning_context: dict[str, Any] | None = None,
) -> float:
    needed = int(cfg.accepted_samples)
    batch_size = int(cfg.sample_batch_size)
    max_draws = int(cfg.max_draws or 0)

    def collect(source: str) -> torch.Tensor:
        accepted: list[torch.Tensor] = []
        accepted_count = 0
        total = 0
        while accepted_count < needed:
            if max_draws > 0 and total >= max_draws:
                raise ConstrainedW2SamplingFailure(
                    source=source,
                    width=float(width),
                    max_draws=max_draws,
                    draws=total,
                    accepted=accepted_count,
                    needed=needed,
                )
            current_batch = batch_size
            if max_draws > 0:
                current_batch = min(current_batch, max_draws - total)
            if current_batch <= 0:
                raise ConstrainedW2SamplingFailure(
                    source=source,
                    width=float(width),
                    max_draws=max_draws,
                    draws=total,
                    accepted=accepted_count,
                    needed=needed,
                )
            if source == "vi":
                samples = _sample_vi(runner, current_batch, current_batch)
            else:
                samples = _sample_target(runner, current_batch, current_batch)
            total += samples.shape[0]
            mask = (samples.abs() < float(width)).all(dim=1)
            if mask.any():
                selected = samples[mask].cpu()
                accepted.append(selected)
                accepted_count += int(selected.shape[0])
        return torch.cat(accepted, dim=0)[:needed]

    try:
        vi = collect("vi")
        truth = collect("truth")
    except ConstrainedW2SamplingFailure as exc:
        fallback_value = abs(float(width))
        message = _sampling_failure_warning(
            failure=exc,
            fallback_value=fallback_value,
            context=warning_context,
        )
        logger.warning(message)
        if warning_callback is not None:
            warning_callback(message)
        return fallback_value
    return compute_sliced_wasserstein(
        vi,
        truth,
        num_projections=int(cfg.get("num_projections", 5000)),
        device=runner.device,
        p=2,
    )


def evaluate_w2_budgeted(runner, target: str, cfg: Any) -> float:
    num_samples = int(cfg.num_samples)
    vi = _sample_vi(runner, num_samples, num_samples)
    baseline = load_baseline_samples(target)
    if baseline.shape[0] >= num_samples:
        truth = baseline[torch.randperm(baseline.shape[0])[:num_samples]]
    else:
        truth = baseline[torch.randint(0, baseline.shape[0], (num_samples,))]
    return compute_sliced_wasserstein(
        vi,
        truth,
        num_projections=int(cfg.num_projections),
        device=runner.device,
        p=2,
    )


def evaluate_one_run(rec: RunRecord, cfg: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_cfg = cfg.evaluation
    runner_cfg = prepare_config(
        rec,
        device=str(eval_cfg.device),
        scratch_results=str(cfg.campaign.scratch_results_dir),
        scratch_tb=str(cfg.campaign.scratch_tb_dir),
    )
    runner, ckpt_dir, ckpt_epoch = build_runner(rec, runner_cfg)
    metrics: dict[str, float] = {}
    errors: dict[str, str] = {}
    warnings: dict[str, str] = {}
    runtimes: dict[str, float] = {}
    raw_rows: list[dict[str, Any]] = []

    try:
        runner.n_elbo_batch_size = int(eval_cfg.elbo.batch_size)
        runner.n_elbo_batches = int(eval_cfg.elbo.num_batches)
        runner.n_elbo_z_samples = int(eval_cfg.elbo.num_z_samples)
        runner.n_w2_samples = int(eval_cfg.w2.num_samples)
        runner.n_w2_projections = int(eval_cfg.w2.num_projections)
        runner.n_bnn_samples = int(eval_cfg.bnn.num_samples)
        runner.n_expected_log_marginal_ref_samples = int(eval_cfg.langevin_kde_elm.num_ref_samples)
        runner.n_expected_log_marginal_model_samples = int(eval_cfg.langevin_kde_elm.num_model_samples)
        runner.n_expected_log_marginal_sample_batch_size = int(eval_cfg.langevin_kde_elm.sample_batch_size)
        runner.n_expected_log_marginal_dim_chunk = int(eval_cfg.langevin_kde_elm.dim_chunk)
        runner.n_expected_log_marginal_ref_chunk = int(eval_cfg.langevin_kde_elm.ref_chunk)
        runner.n_expected_log_marginal_model_chunk = int(eval_cfg.langevin_kde_elm.model_chunk)
        runner.expected_log_marginal_dtype = str(eval_cfg.langevin_kde_elm.dtype)
        runner._expected_log_marginal_reference_samples = None

        def record_metric_warning(metric_name: str) -> Callable[[str], None]:
            return lambda msg: warnings.__setitem__(metric_name, msg)

        metric_plan: list[tuple[str, Any]] = []
        if rec.target in {"banana", "multimodal", "x_shaped", "student_uc", "8_gaussians", "Langevin_post"}:
            metric_plan.append(("elbo", lambda: runner.evaluate_elbo()[0]))
        if rec.target in {"banana", "multimodal", "x_shaped", "student_uc", "8_gaussians", "Langevin_post"}:
            metric_plan.append(("w2", lambda: evaluate_w2_budgeted(runner, rec.target, eval_cfg.w2)))
        if bool(eval_cfg.get("truncated_w2", {}).get("enabled", False)):
            target_widths = eval_cfg.truncated_w2.get("target_widths", {})
            if rec.target in target_widths:
                width = float(target_widths[rec.target])
                metric_name = truncated_w2_metric_name(width)
                metric_plan.append((
                    metric_name,
                    lambda w=width, name=metric_name: constrained_w2(
                        runner,
                        w,
                        OmegaConf.merge(
                            eval_cfg.truncated_w2,
                            {"num_projections": int(eval_cfg.w2.num_projections)},
                        ),
                        warning_callback=record_metric_warning(name),
                        warning_context={
                            "run_id": rec.run_id,
                            "method": rec.method,
                            "target": rec.target,
                            "metric": name,
                            "checkpoint_epoch": ckpt_epoch,
                        },
                    ),
                ))
        if rec.target == "student_uc" and bool(eval_cfg.student_uc_constrained_w2.enabled):
            for width in eval_cfg.student_uc_constrained_w2.widths:
                metric_name = f"w2_edge_{int(width)}"
                metric_plan.append((
                    metric_name,
                    lambda w=float(width), name=metric_name: constrained_w2(
                        runner,
                        w,
                        OmegaConf.merge(
                            eval_cfg.student_uc_constrained_w2,
                            {"num_projections": int(eval_cfg.w2.num_projections)},
                        ),
                        warning_callback=record_metric_warning(name),
                        warning_context={
                            "run_id": rec.run_id,
                            "method": rec.method,
                            "target": rec.target,
                            "metric": name,
                            "checkpoint_epoch": ckpt_epoch,
                        },
                    ),
                ))
        if rec.target == "Langevin_post" and bool(eval_cfg.langevin_kde_elm.enabled):
            metric_plan.append(("kde_elm", lambda: runner.evaluate_expected_log_marginal().value))
        if rec.target.startswith("Bnn_"):
            metric_plan.append(("rmse_nll", runner.evaluate_bnn_metrics))

        for metric_name, metric_fn in metric_plan:
            start = time.perf_counter()
            try:
                value = metric_fn()
                if metric_name == "rmse_nll":
                    rmse, test_llk = value
                    metrics["rmse"] = float(rmse)
                    metrics["nll"] = float(-test_llk)
                else:
                    metrics[metric_name] = float(value)
            except Exception as exc:
                errors[metric_name] = f"{type(exc).__name__}: {exc}"
                if bool(eval_cfg.fail_fast):
                    raise
            finally:
                runtimes[f"{metric_name}_runtime_sec"] = time.perf_counter() - start

        for name, value in metrics.items():
            raw_rows.append(
                {
                    "run_id": rec.run_id,
                    "seed": rec.seed,
                    "method": rec.method,
                    "target": rec.target,
                    "metric": name,
                    "value": value,
                    "checkpoint_epoch": ckpt_epoch,
                    "duration_sec": rec.duration_sec,
                    "warning": warnings.get(name, ""),
                }
            )

        summary = {
            "run_id": rec.run_id,
            "seed": rec.seed,
            "method": rec.method,
            "target": rec.target,
            "checkpoint_epoch": ckpt_epoch,
            "checkpoint_dir": ckpt_dir.relative_to(REPO_ROOT).as_posix(),
            "duration_sec": rec.duration_sec,
            "errors": json.dumps(errors, sort_keys=True),
            "warnings": json.dumps(warnings, sort_keys=True),
            **metrics,
            **runtimes,
        }
        return summary, raw_rows
    finally:
        if hasattr(runner, "writer"):
            runner.writer.close()
        remove_file_handlers()
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["target"], row["method"]), []).append(row)
    out: list[dict[str, Any]] = []
    excluded = {
        "seed",
        "duration_sec",
        "wall_clock_sec",
        "training_time_sec",
        "checkpoint_epoch",
        "summary_iterations",
    }
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and _finite_float(value) is not None
        }
    )
    for (target, method), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "target": target,
            "method": method,
            "seed_count": len({int(item["seed"]) for item in items}),
        }
        durations = [
            value
            for item in items
            for value in [_first_finite(item.get("wall_clock_sec"), item.get("duration_sec"))]
            if value is not None
        ]
        training_times = [
            value
            for item in items
            for value in [_finite_float(item.get("training_time_sec"))]
            if value is not None
        ]
        epochs = [
            value
            for item in items
            for value in [_first_finite(item.get("checkpoint_epoch"), item.get("summary_iterations"))]
            if value is not None
        ]
        if durations:
            summary["duration_sec_mean"] = float(np.mean(durations))
            summary["duration_sec_se"] = float(np.std(durations, ddof=1) / math.sqrt(len(durations))) if len(durations) > 1 else 0.0
            summary["wall_clock_sec_mean"] = summary["duration_sec_mean"]
            summary["wall_clock_sec_se"] = summary["duration_sec_se"]
        if training_times:
            summary["training_time_sec_mean"] = float(np.mean(training_times))
            summary["training_time_sec_se"] = float(np.std(training_times, ddof=1) / math.sqrt(len(training_times))) if len(training_times) > 1 else 0.0
            summary["training_time_sec_count"] = len(training_times)
        if epochs:
            summary["training_iterations_mean"] = float(np.mean(epochs))
        for metric in metric_names:
            values = [value for item in items for value in [_finite_float(item.get(metric))] if value is not None]
            if not values:
                continue
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_se"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            summary[f"{metric}_count"] = len(values)
        out.append(summary)
    return out


def _sgld_kde_cfg(cfg: Any) -> Any:
    return cfg.evaluation.langevin_kde_elm.get("sgld", {})


def _sgld_enabled(cfg: Any) -> bool:
    sgld_cfg = _sgld_kde_cfg(cfg)
    return bool(sgld_cfg.get("enabled", True)) and bool(cfg.evaluation.langevin_kde_elm.enabled)


def evaluate_langevin_sgld_baseline(cfg: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_cfg = cfg.evaluation.langevin_kde_elm
    sgld_cfg = _sgld_kde_cfg(cfg)
    reference_path = repo_path(str(sgld_cfg.get("reference_path", "baselines/mcmc/Langevin_post.pt")))
    model_path = repo_path(str(sgld_cfg.get("model_path", "baselines/mcmc/Langevin_post_sgld_100k.pt")))
    if reference_path is None or model_path is None:
        raise FileNotFoundError("SGLD KDE reference/model paths must be configured.")
    reference_samples = load_baseline_sample_store(reference_path)
    model_samples = load_baseline_sample_store(model_path)
    if reference_samples.shape[0] > int(eval_cfg.num_ref_samples):
        reference_samples = reference_samples[: int(eval_cfg.num_ref_samples)]
    if model_samples.shape[0] > int(eval_cfg.num_model_samples):
        model_samples = model_samples[: int(eval_cfg.num_model_samples)]

    start = time.perf_counter()
    kde_device = str(cfg.evaluation.device)
    if kde_device == "auto":
        kde_device = "cuda" if torch.cuda.is_available() else "cpu"
    estimate = kde_expected_log_marginal(
        reference_samples,
        model_samples,
        dim_chunk=int(eval_cfg.dim_chunk),
        ref_chunk=int(eval_cfg.ref_chunk),
        model_chunk=int(eval_cfg.model_chunk),
        dtype=str(eval_cfg.dtype),
        device=kde_device,
    )
    runtime_sec = time.perf_counter() - start
    summary = {
        "run_id": "langevin_sgld_kde_baseline",
        "seed": int(sgld_cfg.get("seed", 0)),
        "method": "SGLD",
        "target": "Langevin_post",
        "checkpoint_epoch": "",
        "checkpoint_dir": model_path.relative_to(REPO_ROOT).as_posix(),
        "duration_sec": "",
        "errors": "{}",
        "kde_elm": float(estimate.value),
        "kde_elm_runtime_sec": float(runtime_sec),
    }
    raw = [
        {
            "run_id": summary["run_id"],
            "seed": summary["seed"],
            "method": "SGLD",
            "target": "Langevin_post",
            "metric": "kde_elm",
            "value": float(estimate.value),
            "checkpoint_epoch": "",
            "duration_sec": "",
        }
    ]
    return summary, raw


def _append_langevin_sgld_if_needed(
    run_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    cfg: Any,
) -> list[dict[str, Any]]:
    if not _sgld_enabled(cfg):
        return run_rows
    if not any(row.get("target") == "Langevin_post" for row in run_rows):
        return run_rows
    if any(row.get("target") == "Langevin_post" and str(row.get("method")).upper() == "SGLD" for row in run_rows):
        return run_rows
    try:
        summary, raw = evaluate_langevin_sgld_baseline(cfg)
    except Exception:
        if bool(cfg.evaluation.fail_fast):
            raise
        logger.warning("Skipping Langevin_post SGLD KDE baseline.", exc_info=True)
        return run_rows
    raw_rows.extend(raw)
    return [summary, *run_rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def warning_rows_from_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warning_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_warnings = row.get("warnings", "")
        if not raw_warnings:
            continue
        if isinstance(raw_warnings, dict):
            parsed = raw_warnings
        else:
            try:
                parsed = json.loads(str(raw_warnings))
            except json.JSONDecodeError:
                continue
        if not isinstance(parsed, dict):
            continue
        for metric, message in parsed.items():
            if not message:
                continue
            warning_rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "seed": row.get("seed", ""),
                    "method": row.get("method", ""),
                    "target": row.get("target", ""),
                    "metric": metric,
                    "checkpoint_epoch": row.get("checkpoint_epoch", ""),
                    "warning": message,
                }
            )
    return warning_rows


def summarize_warning_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row.get("target", "")),
                str(row.get("method", "")),
                str(row.get("metric", "")),
            ),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for (target, method, metric), items in sorted(grouped.items()):
        out.append(
            {
                "target": target,
                "method": method,
                "metric": metric,
                "warning": "sampling process failed; edge-length fallback used",
                "count": len(items),
                "run_ids": ";".join(str(item.get("run_id", "")) for item in items),
            }
        )
    return out


def write_warning_outputs(out_dir: Path, run_rows: list[dict[str, Any]]) -> None:
    warning_rows = warning_rows_from_run_rows(run_rows)
    summary_rows = summarize_warning_rows(warning_rows)
    write_csv(out_dir / "reevaluation_warnings.csv", warning_rows)
    write_csv(out_dir / "reevaluation_warning_summary.csv", summary_rows)


def log_warning_summary(run_rows: list[dict[str, Any]]) -> None:
    warning_rows = warning_rows_from_run_rows(run_rows)
    if not warning_rows:
        return
    summary_rows = summarize_warning_rows(warning_rows)
    logger.warning("Sampling process failed for %d constrained W2 metric(s).", len(warning_rows))
    for row in summary_rows:
        logger.warning(
            "Sampling failure summary: target=%s method=%s metric=%s count=%s",
            row["target"],
            row["method"],
            row["metric"],
            row["count"],
        )


def evaluate_runs(records: list[RunRecord], cfg: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = repo_path(str(cfg.campaign.output_dir))
    assert out_dir is not None
    run_csv = out_dir / "reevaluation_runs.csv"
    raw_jsonl = out_dir / "reevaluation_raw.jsonl"
    summary_csv = out_dir / "reevaluation_summary.csv"
    if run_csv.exists() and summary_csv.exists() and not bool(cfg.evaluation.overwrite):
        with run_csv.open("r", encoding="utf-8", newline="") as fh:
            run_rows = list(csv.DictReader(fh))
        raw_rows: list[dict[str, Any]] = []
        run_rows = _append_langevin_sgld_if_needed(run_rows, raw_rows, cfg)
        run_rows = augment_run_rows_with_campaign_timing(run_rows, cfg)
        summary_rows = summarize(run_rows)
        write_csv(run_csv, run_rows)
        write_csv(summary_csv, summary_rows)
        write_warning_outputs(out_dir, run_rows)
        log_warning_summary(run_rows)
        return run_rows, summary_rows

    run_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for rec in tqdm(records, desc="Evaluating runs"):
        logger.info(f"Re-evaluating {rec.run_id}")
        summary, raw = evaluate_one_run(rec, cfg)
        run_rows.append(summary)
        run_rows = augment_run_rows_with_campaign_timing(run_rows, cfg)
        raw_rows.extend(raw)
        write_csv(run_csv, run_rows)
        write_jsonl(raw_jsonl, raw_rows)
        write_csv(summary_csv, summarize(run_rows))
        write_warning_outputs(out_dir, run_rows)
    run_rows = _append_langevin_sgld_if_needed(run_rows, raw_rows, cfg)
    summary_rows = summarize(run_rows)
    write_csv(run_csv, run_rows)
    write_jsonl(raw_jsonl, raw_rows)
    write_csv(summary_csv, summary_rows)
    write_warning_outputs(out_dir, run_rows)
    log_warning_summary(run_rows)
    return run_rows, summary_rows
