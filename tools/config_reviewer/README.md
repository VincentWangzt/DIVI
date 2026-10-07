# Config Reviewer

Read-only local UI for comparing base YAML configs across methods and targets.

## Usage

Run from the repository root:

```bash
python tools/config_reviewer/server.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

## Comparison Modes

**Methods on target** - Compare all selected methods for a single target.
Shows how configs differ across SIVI, UIVI, AISIVI, DIVI, and KSIVI for the
same target distribution.

**Targets for method** - Compare all selected targets for a single method.
Shows how configs differ across target distributions for the same method.
Targets can be filtered by group (toy, LRwaveform, BNN).

## Views

- **Key table** - Flat dot-notation comparison matrix highlighting rows where
  values differ across configs.
- **Resolved YAML** - Side-by-side panels showing the fully resolved effective
  config (with nested target/vi_model/reverse_model configs expanded).
- **Summary matrix** - High-level statistics (config count, changed rows,
  missing keys, etc.).

## Config Resolution

The reviewer expands the same nested config layers the runners load at runtime:

- `target_config_path` into `target`
- `vi_model_config_path` into `vi_model`
- `reverse_model_config_path` into `reverse_model` (AISIVI/DIVI)
- `reverse_model_config_path` into `hmc` (UIVI)

If a nested path is omitted, the reviewer uses the runner-style default path
derived from `target_type`, `vi_model_type`, or `reverse_model_type`.

## Live Reload

The server reloads YAML data on every request and the browser polls for file
changes every 3 seconds, so edits made in an editor appear without restarting
the server.

## Methods

- `sivi` (SIVI)
- `uivi` (UIVI)
- `aisivi` (AISIVI)
- `divi` (DIVI)
- `ksivi` (KSIVI)
