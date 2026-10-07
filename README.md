# Denoising Implicit Variational Inference (DIVI)

Official codebase for **Denoising Implicit Variational Inference**, accepted at
**NeurIPS 2026**.

## Environment Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
source .venv/bin/activate
```

This creates a `.venv/` with the correct Python version and all dependencies.
Alternatively, prefix commands with `uv run` to skip activation.

For data-dependent targets (BNN, logistic regression), run once before first use:

```bash
python prepare_data.py
```

## Architecture

### Config System

Configs use OmegaConf YAML with CLI dotlist overrides (`key.subkey=value`).

| Layer | Path | Purpose |
|-------|------|---------|
| Experiment | `configs/{runner}_{target}.yaml` | Top-level run config |
| Target | `configs/targets/` | Distribution parameters |
| Reverse model | `configs/reverse_models/` | Reverse conditional architecture |
| VI model | `configs/vi_models/` | Variational family architecture |

### Core Abstractions

| Abstraction | File | Interface |
|-------------|------|-----------|
| Target Distribution | `models/target_models.py` | `logp(X)`, `score(X)` |
| VI Model | `models/vi_model.py` | `sample_epsilon()`, `forward(eps)` → z |
| Reverse Model | `models/reverse_model.py` | `log_prob(eps, z)`, `sample(z)` |

Data-dependent targets are wrapped by `models/data_bound_target.py` to expose the same `logp`/`score` interface with a bound dataset.

### Runners

All runners implement `runner.learn()`, called by `src.py`.

| Runner | File | Reverse Model | Key Idea |
|--------|------|---------------|----------|
| SIVI | `runner/sivi.py` | None (prior) | Standard semi-implicit VI |
| UIVI | `runner/uivi.py` | HMC | Unbiased importance weighting |
| AISIVI | `runner/aisivi.py` | Learned + annealed | Annealed importance sampling |
| DIVI | `runner/divi.py` | Denoising | Denoising score estimation |
| KSIVI | `runner/ksivi.py` | None | Kernel Stein discrepancy |
| NFVI | `runner/nfvi.py` | None | Exact-density RealNVP variational inference |

Runner types are registered in `runner/runners.py` and selected via `runner_type` in config.

## Targets

**Toy 2D** (exact baselines in `baselines/exact/`):

| Target | Dim | Config |
|--------|-----|--------|
| `banana` | 2 | `configs/targets/banana.yaml` |
| `multimodal` | 2 | `configs/targets/multimodal.yaml` |
| `x_shaped` | 2 | `configs/targets/x_shaped.yaml` |
| `student_uc` | 2 | `configs/targets/student_uc.yaml` |
| `8_gaussians` | 2 | `configs/targets/8_gaussians.yaml` |

**High-dimensional** (MCMC baseline in `baselines/mcmc/`):

| Target | Dim | Config |
|--------|-----|--------|
| `Langevin_post` | 100 | `configs/targets/Langevin_post.yaml` |

**Data-dependent** (ELBO evaluation only; require `prepare_data.py`):

| Target | Dim | Config |
|--------|-----|--------|
| `LRwaveform` | 22 | `configs/targets/LRwaveform.yaml` |
| `Bnn_boston` | 751 | `configs/targets/Bnn_boston.yaml` |
| `Bnn_concrete` | 501 | `configs/targets/Bnn_concrete.yaml` |
| `Bnn_power` | 301 | `configs/targets/Bnn_power.yaml` |
| `Bnn_protein` | 551 | `configs/targets/Bnn_protein.yaml` |
| `Bnn_winered` | 651 | `configs/targets/Bnn_winered.yaml` |
| `Bnn_yacht` | 401 | `configs/targets/Bnn_yacht.yaml` |

## Running Experiments

### Single Experiment

```bash
python src.py --config configs/divi_x_shaped.yaml
python src.py --config configs/sivi_banana.yaml
python src.py --config configs/sivi_banana.yaml train.epochs=20000 train.vi.lr=0.001
```

The RealNVP baseline runs through the same entrypoint:

```bash
python src.py --config configs/nfvi_8_gaussians.yaml
```

`configs/vi_models/RealNVP.yaml` defaults to four coupling layers. NFVI optimizes
the exact flow density and does not use a reverse conditional or score proxy.
It is available for individual runs without changing the standard config grid.

Monitor with:

```bash
tensorboard --logdir tb_logs/
```

### Reproducing Results

For full campaign sweeps, baseline generation, and finalization (evaluation + report generation), see [`scripts/README.md`](scripts/README.md).

For score accuracy and native estimator timing on shared frozen DIVI
checkpoints, see [`finalization/README.md`](finalization/README.md). These
entrypoints load given checkpoints and use posterior HMC as the reference.

## Output Layout

```
results/{runner_type}/{target_type}/{timestamp}/   # checkpoints, samples, metrics
tb_logs/{runner_type}/{target_type}/{timestamp}/   # TensorBoard event files
campaigns/default_config_grid/
  ├── manifest.csv                                 # job index
  └── generated_reports/                           # figures, tables
baselines/exact/                                   # toy 2D exact samples (.pt)
baselines/mcmc/                                    # SGLD samples (.pt)
```

## Scripts & Tools

**Scripts** (`scripts/`): Grid orchestration, baseline generation, and finalization.

Run Python entrypoints with `python` and shell wrappers with `bash` from the
repository root.

- Key scripts: `reproduce_baselines.sh`, `run_default_config_grid.sh`, `run_finalization.py`
- Full reference: [`scripts/README.md`](scripts/README.md)

**Config Reviewer** (`tools/config_reviewer/`): Local web UI for comparing YAML configs across methods and targets.

```bash
python tools/config_reviewer/server.py --port 8765
```

Details: [`tools/config_reviewer/README.md`](tools/config_reviewer/README.md)

## Citation

The official BibTeX entry will be added after the preprint is uploaded to arXiv.
