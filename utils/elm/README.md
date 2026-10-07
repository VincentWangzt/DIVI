# Expected Log Marginal (ELM)

In this codebase, **Expected Log Marginal (ELM)** refers to the adopted
paper-style coordinate-wise KDE evaluation metric:

```text
ELM = E_{z ~ r}[sum_j log q_hat_phi,j(z_j)].
```

Here `r` is the reference distribution represented by baseline samples, and
`q_hat_phi,j` is a one-dimensional Gaussian KDE fit to generated VI samples in
coordinate `j`. This is the metric used by the training-time
`metric.expected_log_marginal` hook and by final evaluation code.

This is intentionally not the full semi-implicit marginal density
`E_{z ~ r}[log q_phi(z)]`. Older reverse-IS helpers for that quantity still
exist in this package for historical analysis, but they are not the adopted ELM
metric.

## KDE-ELM Workflow

1. Draw or load reference samples from the target/baseline sample store.
2. Generate samples from the trained VI model.
3. For each coordinate, fit a Gaussian KDE to the generated VI samples.
4. Evaluate each reference point under the coordinate-wise KDE marginals.
5. Average the summed coordinate log densities over reference points.

The estimator is coordinate-marginal by design:

```text
ELM ~= (1 / N_ref) sum_i sum_j log q_hat_phi,j(z_ij).
```

## Code Map

- `estimators.py`
  - `load_baseline_sample_store`: loads saved baseline samples from either a raw
    tensor or a dict containing a `samples` tensor.
  - `sample_reference_samples`: selects fixed reference samples for the outer
    expectation over `z`.
  - `kde_expected_log_marginal`: computes the adopted KDE-ELM metric from
    reference samples and generated VI samples.
  - `estimate_log_q_prior`, `estimate_log_q_reverse_is`, `summarize_elm`:
    legacy full-marginal helpers retained for historical analysis.
- `types.py`
  - `KDEELMEstimate`: return type for the adopted KDE-ELM metric.
  - `ELMEstimate`, `LogQEstimate`, `ReverseProposalFit`: legacy/full-marginal
    helper result types.
- `proposal.py`
  - Reverse proposal utilities used only by the legacy reverse-IS helpers.

## Reading The Metric

Higher ELM is better: the generated VI coordinate marginals assign higher
expected log density to reference target samples. The estimate depends on the
reference sample set, generated VI sample budget, KDE bandwidth rule, and chunk
sizes used for memory control.

Chunk sizes should not change the statistical estimate; they only trade memory
for runtime. The generated sample budget does change the KDE fit and therefore
the reported metric.
