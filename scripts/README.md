# Scripts

This directory contains runnable experiment entrypoints for the default-config
grid campaign and baseline generation.

## Quick Start

Run commands from the repository root with the project environment activated.
Invoke Python scripts with `python` and shell wrappers with `bash`, as shown below.

The two shell scripts provide one-command entrypoints for the most common
workflows. See each script's header comments for the full option list.

### Reproduce all baselines

```bash
bash scripts/reproduce_baselines.sh
```

This regenerates:
- Exact toy 2D baselines under `baselines/exact/` (banana, multimodal,
  8_gaussians, x_shaped, student_uc)
- SGLD Langevin_post samples under `baselines/mcmc/` (1K-chain and 100K-chain
  variants)

### Run the full default-config grid (sweep + finalization)

```bash
bash scripts/run_default_config_grid.sh \
  --seeds "42 43 44 45 46" \
  --exclude-methods "rsivi" \
  --finalize-workers 2
```

This runs the two-phase pipeline end-to-end:
1. **Phase 1 (sweep):** schedules all `<method> x <target> x <seed>` jobs across
   available GPUs via `run_default_config_grid_sweep.py`.
2. **Phase 2 (finalization):** runs evaluation, scatter grids, diagnostic plots,
   and summary tables via `run_finalization.py`.

Use `--dry-run` to preview the sweep plan without launching jobs. Use
`--skip-sweep` or `--skip-finalization` to run only one phase.

## Campaign Scripts

| Script | Purpose |
| --- | --- |
| `run_default_config_grid_sweep.py` | Dynamic GPU scheduler for the `<method>_<target>` grid. Handles resume, retry, stale-detection, and per-run async finalization. |
| `run_finalization.py` | Runs final evaluation, figures, and tables for a completed campaign. |
| `run_default_config_grid.sh` | End-to-end wrapper that runs the sweep then finalization in sequence. |

### Sweep examples

Preview the current default grid without launching jobs:

```bash
python scripts/run_default_config_grid_sweep.py --dry-run
```

Run the sweep with five seeds on auto-discovered GPUs:

```bash
python scripts/run_default_config_grid_sweep.py \
  --seeds 42 43 44 45 46 \
  --finalize-mode async \
  --finalize-workers 1
```

Detect stale artifacts after a config change and rerun them:

```bash
python scripts/run_default_config_grid_sweep.py \
  --seeds 42 43 44 45 46 \
  --hash-existing-artifacts

python scripts/run_default_config_grid_sweep.py \
  --seeds 42 43 44 45 46 \
  --rerun-stale
```

### Finalization examples

Run the full finalization pass (evaluate, plot, tabulate):

```bash
python scripts/run_finalization.py
```

Run only evaluation with overwrite:

```bash
python scripts/run_finalization.py \
  --only evaluate \
  --set evaluation.overwrite=true
```

Regenerate specific outputs from existing evaluation data:

```bash
python scripts/run_finalization.py \
  --only scatter_grid \
  --only toy_tables \
  --only bnn_table
```

## Score analysis entrypoints

```bash
python scripts/run_score_approximation.py --check-only
python scripts/run_score_approximation.py evaluation.device=cuda
python scripts/run_score_estimator_timing.py evaluation.device=cuda
```

These entrypoints compare SIVI, UIVI, AISIVI, and DIVI on the same frozen DIVI
variational checkpoints. They accept a custom `--config` and OmegaConf dotlist
overrides. AISIVI's proposal is fitted against the fixed distribution and cached
for reuse by both entrypoints. Accuracy uses posterior HMC only; timing excludes
fitting, reference evaluation, and input generation. See
[`finalization/README.md`](../finalization/README.md) for checkpoint requirements,
configuration defaults, report definitions, and output locations.

## Baseline Scripts

| Script | Purpose |
| --- | --- |
| `reproduce_baselines.sh` | One-command regeneration of all baseline sample files (exact + SGLD). |
| `generate_exact_baselines.py` | Draws 100K exact samples for each toy 2D target with an analytic sampler. |
| `run_sgld_baseline.py` | Generates SGLD samples for a given target. Supports multi-chain runs with configurable step size, burn-in, thinning, and gradient clipping. |
| `grid_finalization.py` | Shared library module providing config-hash computation, event logging, and per-run finalization helpers used by the sweep scheduler. |

### Baseline examples

Generate only exact toy baselines:

```bash
python scripts/generate_exact_baselines.py --seed 42 --num-samples 100000
```

Generate SGLD samples for Langevin_post with 1K chains:

```bash
python scripts/run_sgld_baseline.py \
  --target Langevin_post \
  --num-samples 100000 \
  --burn-in 100000 \
  --step-size 1e-4 \
  --num-chains 1000 \
  --max-grad-norm 1000.0 \
  --seed 42
```

## Output Locations

| Artifact | Path |
| --- | --- |
| Run results | `results/default_config_grid/` |
| TensorBoard logs | `tb_logs/default_config_grid/` |
| Campaign manifest | `campaigns/default_config_grid/` |
| Generated reports | `campaigns/default_config_grid/generated_reports/` |
| Exact baselines | `baselines/exact/` |
| MCMC baselines | `baselines/mcmc/` |
