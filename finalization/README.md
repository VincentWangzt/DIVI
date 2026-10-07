# Finalization Module

Processes experiment results from the `default_config_grid` campaign into
publication-ready figures and LaTeX tables.

## Usage

Run the full pipeline:

```bash
python scripts/run_finalization.py
```

Run selected modules with `--only` (may be repeated):

```bash
python scripts/run_finalization.py --only scatter_grid
python scripts/run_finalization.py --only evaluate --set evaluation.overwrite=true
python scripts/run_finalization.py --only kl_iteration_grid --only kl_time_grid
```

Override configuration values with `--set`:

```bash
python scripts/run_finalization.py --set selection.seeds=[42] --set evaluation.device=cpu
```

The default configuration is at `configs/finalization/default_config_grid.yaml`.
A custom config can be passed with `--config <path>`.

## Score approximation on frozen DIVI checkpoints

```bash
python scripts/run_score_approximation.py --check-only
python scripts/run_score_approximation.py selection.targets=[x_shaped] evaluation.device=cuda
```

This uses `configs/finalization/score_approximation.yaml`. Explicit
`selection.run_dirs=[results/DIVI/x_shaped/<timestamp>]` replaces manifest
selection. Each source run must include `full_config.yaml` and both
`vi_model.pt` and `reverse_model.pt` for every selected checkpoint epoch.

SIVI, UIVI, AISIVI, and DIVI all load the same frozen ConditionalGaussian VI
checkpoint and receive the same `(epsilon, z)` input bank. DIVI loads its
matching score-network checkpoint. AISIVI fits its reverse proposal to samples
from this fixed VI model; the VI parameters never change. Completed proposal
fits are cached and identified by checkpoint, configuration, and code hashes.
The other estimators use their existing target-specific configs and native
auxiliary sample counts. NFVI is a separate variational family and is excluded.

The sole reference is posterior HMC for `q(epsilon | z)`. The default uses
20 chains, 100,000 retained draws in total per input, and adaptation only during
burn-in. Reports retain acceptance, divergence, step-size, epsilon/score R-hat,
and quality warnings. The squared L2 error is measured against the mean of
chain scores; reference internal variability and Monte Carlo error are also
reported. Mean and standard error are aggregated over the actual selected
source seeds.

Reports, proposal/HMC caches, and runner scratch files live under
`results/score_approximation/`; scratch TensorBoard logs live under
`tb_logs/score_approximation/`. Check-only mode lists inputs without fitting.

## Native score-estimator timing

```bash
python scripts/run_score_estimator_timing.py --check-only
python scripts/run_score_estimator_timing.py evaluation.device=cuda
```

The timing config overlays the score-approximation defaults, selecting the final
10,000-iteration checkpoint, batches of 1 and 128, 10 warmup calls, and 100 timed
calls. It uses the same frozen DIVI checkpoint and cached AISIVI proposal fits as
score approximation. Each method receives identical pre-generated input banks.

Timing measures synchronized native inference, including SIVI/AISIVI autograd
and UIVI's native HMC transitions. Loading, proposal fitting, sample-bank
generation, warmup, reference HMC, and diagnostics are outside the timer.
`timing_per_checkpoint.csv` reports call-level mean/SD; `timing_summary.csv`
reports mean/SE across source-seed means. Raw repetitions, checkpoint/proposal
hashes, native auxiliary counts, and hardware details are retained. Reports live
under `results/score_estimator_timing/`; proposal caches remain shared under
`results/score_approximation/cache/`.

Available `--only` modules:

| Module | Description |
|--------|-------------|
| `evaluate` | Re-evaluate checkpoints and write per-run metrics |
| `scatter_grid` | Toy target sample scatter plots |
| `scatter_hist_grid` | Scatter plots with marginal histograms |
| `langevin_trace_grid` | Langevin target trace plots |
| `kl_iteration_grid` | KL divergence vs. iteration curves |
| `kl_time_grid` | KL divergence vs. wall-clock time curves |
| `grad_norm_iteration_grid` | Gradient norm vs. iteration curves |
| `weight_norm_iteration_grid` | Weight norm vs. iteration curves |
| `m_eps_iteration_grid` | Mixing samples (m_eps) vs. iteration curves |
| `score_4th_moment_iteration_grid` | Score fourth moments vs. iteration curves |
| `score_diff_l2_fourth_iteration_grid` | E[\\|\\|score_p - score_q\\|\\|^4] vs. iteration curves |
| `score_linearity_grid` | Score linearity bound scatter: log\\|\\|diff\\|\\| - log(\\|\\|z\\|\\| + 1) vs \\|\\|z\\|\\| |
| `toy_tables` | Summary metrics table for toy targets |
| `toy_method_grid` | Per-method breakdown table for toy targets |
| `langevin_table` | Metrics table for the Langevin target |
| `student_edge_table` | Edge-length W2 table for Student-UC target |
| `bnn_table` | RMSE/NLL table for BNN targets |

## Outputs

All outputs are written to:

```
campaigns/default_config_grid/generated_reports/finalization/
├── figures/          # PNG and PDF figures
├── tables/           # LaTeX .tex table files
├── reevaluation_summary.csv
└── finalization_report.md
```
