from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _rel(path: Path) -> str:
    """Return repo-relative POSIX path string."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()

from finalization.artifacts import completed_runs, load_manifest, select_runs  # noqa: E402
from finalization.config import load_config, repo_path  # noqa: E402
from finalization.plots import render_grad_norm_iteration_grid, render_kl_iteration_grid, render_kl_time_grid, render_langevin_trace_grid, render_m_eps_iteration_grid, render_scatter_grid, render_scatter_hist_grid, render_score_diff_l2_fourth_iteration_grid, render_score_linearity_grid, render_score_norm_linearity_grid, render_score_norm_linearity_uniform_grid, render_score_p_4th_moment_iteration_grid, render_score_q_4th_moment_iteration_grid, render_vi_fourth_moment_iteration_grid, render_weight_norm_iteration_grid  # noqa: E402
from finalization.runner_eval import augment_run_rows_with_campaign_timing, evaluate_runs, summarize, write_csv  # noqa: E402
from finalization.tables import render_tables  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _selection_seeds(value):
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return [int(seed) for seed in value]


def _enabled(cfg, name: str) -> bool:
    return bool(cfg.modules.get(name, False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final evaluation and visualization for a campaign.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. --set evaluation.overwrite=true",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=[
            "evaluate",
            "scatter_grid",
            "scatter_hist_grid",
            "toy_tables",
            "toy_method_grid",
            "langevin_table",
            "student_edge_table",
            "langevin_trace_grid",
            "bnn_table",
            "kl_iteration_grid",
            "kl_time_grid",
            "grad_norm_iteration_grid",
            "weight_norm_iteration_grid",
            "m_eps_iteration_grid",
            "vi_fourth_moment_iteration_grid",
            "score_4th_moment_iteration_grid",
            "score_diff_l2_fourth_iteration_grid",
            "score_linearity_grid",
            "score_norm_linearity_grid",
            "score_norm_linearity_uniform_grid",
        ],
        help="Run only selected module(s). May be passed multiple times.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.only:
        for key in cfg.modules.keys():
            cfg.modules[key] = key in set(args.only)

    manifest = load_manifest(cfg.campaign.manifest_path)
    all_records = completed_runs(manifest)
    methods = [str(method) for method in cfg.selection.methods]
    eval_targets = [str(target) for target in cfg.selection.evaluation_targets]
    if _enabled(cfg, "bnn_table"):
        eval_targets = sorted(set(eval_targets) | {str(target) for target in cfg.selection.bnn_targets})
    eval_records = select_runs(
        all_records,
        methods=methods,
        targets=eval_targets,
        seeds=_selection_seeds(cfg.selection.seeds),
    )

    out_dir = repo_path(cfg.campaign.output_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict]
    summary_rows: list[dict]
    if _enabled(cfg, "evaluate"):
        run_rows, summary_rows = evaluate_runs(eval_records, cfg)
    else:
        run_rows = _read_csv(out_dir / "reevaluation_runs.csv")
        run_rows = augment_run_rows_with_campaign_timing(run_rows, cfg)
        summary_rows = summarize(run_rows) if run_rows else _read_csv(out_dir / "reevaluation_summary.csv")
        if run_rows:
            write_csv(out_dir / "reevaluation_runs.csv", run_rows)
            write_csv(out_dir / "reevaluation_summary.csv", summary_rows)

    figure_seeds = {int(cfg.selection.seed_for_figures)}
    for _method, override_seed in cfg.selection.get("seed_overrides", {}).items():
        figure_seeds.add(int(override_seed))
    figure_records = select_runs(
        all_records,
        methods=sorted(set(methods) | {str(method) for method in cfg.selection.scatter_methods}),
        targets=sorted(set(str(target) for target in cfg.selection.scatter_targets) | {"Langevin_post"}),
        seeds=sorted(figure_seeds),
    )

    generated: list[str] = []
    if _enabled(cfg, "scatter_grid"):
        generated.append(_rel(render_scatter_grid(figure_records, cfg)))
    if _enabled(cfg, "scatter_hist_grid"):
        generated.append(_rel(render_scatter_hist_grid(figure_records, cfg)))
    if _enabled(cfg, "langevin_trace_grid"):
        generated.append(_rel(render_langevin_trace_grid(figure_records, cfg)))
    # KL convergence curve plots — use all seeds for aggregation
    if _enabled(cfg, "kl_iteration_grid") or _enabled(cfg, "kl_time_grid"):
        kl_curve_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.kl_curve_methods],
            targets=[str(t) for t in cfg.selection.kl_curve_targets],
            seeds=_selection_seeds(cfg.selection.seeds),
        )
        if _enabled(cfg, "kl_iteration_grid"):
            generated.append(_rel(render_kl_iteration_grid(kl_curve_records, cfg)))
        if _enabled(cfg, "kl_time_grid"):
            generated.append(_rel(render_kl_time_grid(kl_curve_records, cfg)))
    if _enabled(cfg, "grad_norm_iteration_grid"):
        grad_norm_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.grad_norm_methods],
            targets=[str(t) for t in cfg.selection.grad_norm_targets],
            seeds=_selection_seeds(cfg.selection.seeds),
        )
        generated.append(_rel(render_grad_norm_iteration_grid(grad_norm_records, cfg)))
    if _enabled(cfg, "weight_norm_iteration_grid"):
        weight_norm_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.weight_norm_methods],
            targets=[str(t) for t in cfg.selection.weight_norm_targets],
            seeds=_selection_seeds(cfg.selection.seeds),
        )
        generated.append(_rel(render_weight_norm_iteration_grid(weight_norm_records, cfg)))
    if _enabled(cfg, "m_eps_iteration_grid"):
        m_eps_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.m_eps_methods],
            targets=[str(t) for t in cfg.selection.m_eps_targets],
            seeds=_selection_seeds(cfg.selection.seeds),
        )
        generated.append(_rel(render_m_eps_iteration_grid(m_eps_records, cfg)))
    if _enabled(cfg, "vi_fourth_moment_iteration_grid"):
        vi_4th_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.vi_fourth_moment_methods],
            targets=[str(t) for t in cfg.selection.vi_fourth_moment_targets],
            seeds=_selection_seeds(cfg.selection.seeds),
        )
        generated.append(_rel(render_vi_fourth_moment_iteration_grid(vi_4th_records, cfg)))
    if _enabled(cfg, "score_4th_moment_iteration_grid"):
        score_4th_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.score_4th_moment_methods],
            targets=[str(t) for t in cfg.selection.score_4th_moment_targets],
            seeds=_selection_seeds(cfg.selection.seeds),
        )
        generated.append(_rel(render_score_p_4th_moment_iteration_grid(score_4th_records, cfg)))
        generated.append(_rel(render_score_q_4th_moment_iteration_grid(score_4th_records, cfg)))
    if _enabled(cfg, "score_diff_l2_fourth_iteration_grid"):
        score_diff_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.score_4th_moment_methods],
            targets=[str(t) for t in cfg.selection.score_4th_moment_targets],
            seeds=_selection_seeds(cfg.selection.seeds),
        )
        generated.append(_rel(render_score_diff_l2_fourth_iteration_grid(score_diff_records, cfg)))
    if _enabled(cfg, "score_linearity_grid"):
        score_lin_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.score_linearity_methods],
            targets=[str(t) for t in cfg.selection.score_linearity_targets],
            seeds=[int(s) for s in cfg.selection.score_linearity_seeds],
        )
        generated.append(_rel(render_score_linearity_grid(score_lin_records, cfg)))
    if _enabled(cfg, "score_norm_linearity_grid"):
        score_norm_lin_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.score_norm_linearity_methods],
            targets=[str(t) for t in cfg.selection.score_norm_linearity_targets],
            seeds=[int(s) for s in cfg.selection.score_norm_linearity_seeds],
        )
        generated.append(_rel(render_score_norm_linearity_grid(score_norm_lin_records, cfg)))
    if _enabled(cfg, "score_norm_linearity_uniform_grid"):
        score_norm_lin_uniform_records = select_runs(
            all_records,
            methods=[str(m) for m in cfg.selection.score_norm_linearity_uniform_methods],
            targets=[str(t) for t in cfg.selection.score_norm_linearity_uniform_targets],
            seeds=[int(s) for s in cfg.selection.score_norm_linearity_uniform_seeds],
        )
        generated.append(_rel(render_score_norm_linearity_uniform_grid(score_norm_lin_uniform_records, cfg)))
    if any(
        _enabled(cfg, name)
        for name in ("toy_tables", "toy_method_grid", "langevin_table", "student_edge_table", "bnn_table")
    ):
        table_paths = render_tables(summary_rows, cfg)
        for name, path in table_paths.items():
            if _enabled(cfg, "bnn_table") or name != "bnn":
                generated.append(_rel(path))

    warning_summary = _read_csv(out_dir / "reevaluation_warning_summary.csv")

    report_path = out_dir / "finalization_report.md"
    lines = [
        "# Finalization Report",
        "",
        f"Completed manifest runs discovered: {len(all_records)}",
        f"Evaluation runs selected: {len(eval_records)}",
        f"Per-run rows: {len(run_rows)}",
        f"Aggregate rows: {len(summary_rows)}",
        "",
        "## Generated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated)
    if warning_summary:
        lines.extend(["", "## Warnings", ""])
        total_warnings = sum(int(row.get("count") or 0) for row in warning_summary)
        lines.append(f"Constrained W2 sampling process failed for {total_warnings} metric(s); edge-length fallbacks were used.")
        for row in warning_summary:
            lines.append(
                "- "
                f"{row.get('target', '')}/{row.get('method', '')}/{row.get('metric', '')}: "
                f"{row.get('count', '')}"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {_rel(report_path)}")


if __name__ == "__main__":
    main()
