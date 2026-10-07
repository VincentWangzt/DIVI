# Reproduce all baseline sample files under baselines/.
#
# 1) Exact toy 2D baselines  -> baselines/exact/<target>_exact_100k.pt
# 2) SGLD Langevin_post      -> baselines/mcmc/Langevin_post.pt        (1K chains)
#                             -> baselines/mcmc/Langevin_post_sgld_100k.pt (100K chains)
#
# Usage:
#   bash scripts/reproduce_baselines.sh

set -euo pipefail

SEED=42

# ─── 1. Exact toy baselines ─────────────────────────────────────────────────

echo "=== Generating exact toy baselines ==="
python scripts/generate_exact_baselines.py --seed "$SEED"

# ─── 2. SGLD Langevin_post (1K chains, 100K steps) ──────────────────────────

echo ""
echo "=== SGLD Langevin_post: 1K chains, 100K steps ==="

SGLD_1K_DIR=$(mktemp -d)
python scripts/run_sgld_baseline.py \
    --target Langevin_post \
    --num-samples 1000 \
    --burn-in 100000 \
    --step-size 1e-4 \
    --thinning 1 \
    --num-chains 1000 \
    --max-grad-norm 1000.0 \
    --seed "$SEED" \
    --output-dir "$SGLD_1K_DIR" \
    --overwrite

cp "$SGLD_1K_DIR/samples.pt" baselines/mcmc/Langevin_post.pt
rm -rf "$SGLD_1K_DIR"
echo "  -> baselines/mcmc/Langevin_post.pt"

# ─── 3. SGLD Langevin_post (100K chains, 100K steps) ────────────────────────

echo ""
echo "=== SGLD Langevin_post: 100K chains, 100K steps ==="

SGLD_100K_DIR=$(mktemp -d)
python scripts/run_sgld_baseline.py \
    --target Langevin_post \
    --num-samples 100000 \
    --burn-in 100000 \
    --step-size 1e-4 \
    --thinning 1 \
    --num-chains 100000 \
    --max-grad-norm 1000.0 \
    --seed "$SEED" \
    --output-dir "$SGLD_100K_DIR" \
    --overwrite

cp "$SGLD_100K_DIR/samples.pt" baselines/mcmc/Langevin_post_sgld_100k.pt
rm -rf "$SGLD_100K_DIR"
echo "  -> baselines/mcmc/Langevin_post_sgld_100k.pt"

echo ""
echo "=== Done. All baselines regenerated. ==="
