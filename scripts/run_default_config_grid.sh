#
# End-to-end script for the default_config_grid campaign.
#
# Phase 1: Run the grid sweep (training + per-run finalization)
# Phase 2: Run the finalization pipeline (evaluation, figures, tables)
#
# Usage:
#   bash scripts/run_default_config_grid.sh [OPTIONS]
#
# Options (all optional):
#   --seeds "42 43 44 45 46"   Seeds for the sweep (default: "42")
#   --methods "sivi uivi ..."  Methods to run (default: all 6)
#   --exclude-methods "rsivi"  Methods to exclude
#   --targets "x_shaped ..."   Restrict to specific targets
#   --gpus "0 1 2 3"           GPU indices (default: auto-discover)
#   --finalize-config PATH     Config for run_finalization.py (default: configs/finalization/default_config_grid.yaml)
#   --finalize-only MODULES    Comma-separated modules to run in finalization (default: all)
#   --finalize-set OVERRIDE    OmegaConf override for finalization (repeatable)
#   --skip-sweep               Skip Phase 1, run only finalization
#   --skip-finalization        Skip Phase 2, run only the sweep
#   --dry-run                  Preview the sweep plan without running
#   --retry-failed             Retry previously failed runs
#   --rerun-stale              Rerun runs whose config hash changed
#   --force-rerun              Treat all matching completed runs as stale
#   --finalize-workers N       Async finalization workers (default: 1)
#   --help                     Print this help and exit
#
# Example:
#   bash scripts/run_default_config_grid.sh \
#     --seeds "42 43 44 45 46" \
#     --exclude-methods "rsivi" \
#     --finalize-workers 2
#
set -euo pipefail

# ─── Resolve repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── Defaults ─────────────────────────────────────────────────────────────────
SEEDS="42"
METHODS=""
EXCLUDE_METHODS=""
TARGETS=""
GPUS=""
FINALIZE_CONFIG="configs/finalization/default_config_grid.yaml"
FINALIZE_ONLY=""
FINALIZE_SET=()
SKIP_SWEEP=false
SKIP_FINALIZATION=false
DRY_RUN=false
RETRY_FAILED=false
RERUN_STALE=false
FORCE_RERUN=false
FINALIZE_WORKERS=1
EXTRA_OVERRIDES=()

# ─── Parse arguments ──────────────────────────────────────────────────────────
print_help() {
    sed -n '/^# Usage:/,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"; shift 2 ;;
        --methods)
            METHODS="$2"; shift 2 ;;
        --exclude-methods)
            EXCLUDE_METHODS="$2"; shift 2 ;;
        --targets)
            TARGETS="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --finalize-config)
            FINALIZE_CONFIG="$2"; shift 2 ;;
        --finalize-only)
            FINALIZE_ONLY="$2"; shift 2 ;;
        --finalize-set)
            FINALIZE_SET+=("$2"); shift 2 ;;
        --skip-sweep)
            SKIP_SWEEP=true; shift ;;
        --skip-finalization)
            SKIP_FINALIZATION=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --retry-failed)
            RETRY_FAILED=true; shift ;;
        --rerun-stale)
            RERUN_STALE=true; shift ;;
        --force-rerun)
            FORCE_RERUN=true; shift ;;
        --finalize-workers)
            FINALIZE_WORKERS="$2"; shift 2 ;;
        --extra-override)
            EXTRA_OVERRIDES+=("$2"); shift 2 ;;
        --help|-h)
            print_help ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ─── Detect Python ────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python not found. Set PYTHON env var or activate your venv." >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo " default_config_grid — End-to-End Pipeline"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Python:     $PYTHON"
echo "  Repo root:  $REPO_ROOT"
echo "  Seeds:      $SEEDS"
echo "  Dry run:    $DRY_RUN"
echo ""

# ─── Phase 1: Grid Sweep ──────────────────────────────────────────────────────
if [[ "$SKIP_SWEEP" == false ]]; then
    echo "───────────────────────────────────────────────────────────────────────"
    echo " Phase 1: Grid Sweep (run_default_config_grid_sweep.py)"
    echo "───────────────────────────────────────────────────────────────────────"
    echo ""

    SWEEP_ARGS=()
    # shellcheck disable=SC2086
    SWEEP_ARGS+=(--seeds $SEEDS)

    if [[ -n "$METHODS" ]]; then
        # shellcheck disable=SC2086
        SWEEP_ARGS+=(--methods $METHODS)
    fi
    if [[ -n "$EXCLUDE_METHODS" ]]; then
        # shellcheck disable=SC2086
        SWEEP_ARGS+=(--exclude-methods $EXCLUDE_METHODS)
    fi
    if [[ -n "$TARGETS" ]]; then
        # shellcheck disable=SC2086
        SWEEP_ARGS+=(--targets $TARGETS)
    fi
    if [[ -n "$GPUS" ]]; then
        # shellcheck disable=SC2086
        SWEEP_ARGS+=(--gpus $GPUS)
    fi
    if [[ "$DRY_RUN" == true ]]; then
        SWEEP_ARGS+=(--dry-run)
    fi
    if [[ "$RETRY_FAILED" == true ]]; then
        SWEEP_ARGS+=(--retry-failed)
    fi
    if [[ "$RERUN_STALE" == true ]]; then
        SWEEP_ARGS+=(--rerun-stale)
    fi
    if [[ "$FORCE_RERUN" == true ]]; then
        SWEEP_ARGS+=(--force-rerun)
    fi

    SWEEP_ARGS+=(--finalize-mode async)
    SWEEP_ARGS+=(--finalize-workers "$FINALIZE_WORKERS")

    for override in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
        SWEEP_ARGS+=(--extra-override "$override")
    done

    echo "  Command: $PYTHON scripts/run_default_config_grid_sweep.py ${SWEEP_ARGS[*]}"
    echo ""

    "$PYTHON" scripts/run_default_config_grid_sweep.py "${SWEEP_ARGS[@]}"
    SWEEP_EXIT=$?

    if [[ $SWEEP_EXIT -ne 0 ]]; then
        echo ""
        echo "ERROR: Grid sweep exited with code $SWEEP_EXIT" >&2
        exit $SWEEP_EXIT
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        echo "Dry run complete. Exiting."
        exit 0
    fi

    echo ""
    echo "✓ Phase 1 complete."
    echo ""
else
    echo "  [skipping Phase 1: --skip-sweep]"
    echo ""
fi

# ─── Phase 2: Finalization ────────────────────────────────────────────────────
if [[ "$SKIP_FINALIZATION" == false ]]; then
    echo "───────────────────────────────────────────────────────────────────────"
    echo " Phase 2: Finalization (run_finalization.py)"
    echo "───────────────────────────────────────────────────────────────────────"
    echo ""

    FINAL_ARGS=()
    FINAL_ARGS+=(--config "$FINALIZE_CONFIG")

    if [[ -n "$FINALIZE_ONLY" ]]; then
        IFS=',' read -ra ONLY_MODULES <<< "$FINALIZE_ONLY"
        for module in "${ONLY_MODULES[@]}"; do
            FINAL_ARGS+=(--only "$module")
        done
    fi

    for override in "${FINALIZE_SET[@]+"${FINALIZE_SET[@]}"}"; do
        FINAL_ARGS+=(--set "$override")
    done

    echo "  Command: $PYTHON scripts/run_finalization.py ${FINAL_ARGS[*]}"
    echo ""

    "$PYTHON" scripts/run_finalization.py "${FINAL_ARGS[@]}"
    FINAL_EXIT=$?

    if [[ $FINAL_EXIT -ne 0 ]]; then
        echo ""
        echo "ERROR: Finalization exited with code $FINAL_EXIT" >&2
        exit $FINAL_EXIT
    fi

    echo ""
    echo "✓ Phase 2 complete."
    echo ""
else
    echo "  [skipping Phase 2: --skip-finalization]"
    echo ""
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo " All phases complete."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Campaign dir:   campaigns/default_config_grid/"
echo "  Results:        results/default_config_grid/"
echo "  TB logs:        tb_logs/default_config_grid/"
echo "  Reports:        campaigns/default_config_grid/generated_reports/"
echo "  Finalization:   campaigns/default_config_grid/generated_reports/finalization/"
echo ""
