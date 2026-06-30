#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

# ── Find the latest checkpoint for a given PDE type ──────────────────────────
find_latest_ckpt() {
    local pde_type="$1"
    local ckpt
    # Search flat pretrained files and nested trained checkpoints; pick most recently modified
    ckpt=$(
        { find "$ROOT/checkpoints" -maxdepth 1 -name "*${pde_type}*.ckpt" 2>/dev/null
          find "$ROOT/checkpoints/genPANIS_${pde_type}_"* -name "*.ckpt" 2>/dev/null; } \
        | xargs ls -t 2>/dev/null | head -1
    )
    if [[ -z "$ckpt" ]]; then
        echo "ERROR: No checkpoint found for '${pde_type}' under $ROOT/checkpoints/" >&2
        echo "       Expected a flat file like $ROOT/checkpoints/${pde_type}*_pretrained.ckpt" >&2
        echo "       or a trained checkpoint under $ROOT/checkpoints/genPANIS_${pde_type}_*" >&2
        echo "       Train a model first: python experiments/demo/train_${pde_type}.py" >&2
        exit 1
    fi
    echo "$ckpt"
}

usage() {
    echo "Usage: bash test_run.sh [--num_samples N] [--burn N] [--figs_dir PATH]"
    echo ""
    echo "  --num_samples  number of HMC samples (default: 200)"
    echo "  --burn         number of HMC burn-in steps (default: 100)"
    echo "  --figs_dir     directory for output figures (default: session_dir/figs)"
    exit 0
}

NUM_SAMPLES=200
BURN=100
FIGS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --burn)        BURN="$2";        shift 2 ;;
        --figs_dir)    FIGS_DIR="$2";    shift 2 ;;
        --help|-h)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

DARCY_CKPT=$(find_latest_ckpt darcy)
HELM_CKPT=$(find_latest_ckpt helmholz)
[[ -z "$FIGS_DIR" ]] && FIGS_DIR="$DIR/figs"

mkdir -p "$FIGS_DIR"

echo "=== test_run.sh ==="
echo "  num_samples : $NUM_SAMPLES"
echo "  burn        : $BURN"
echo "  figs_dir    : $FIGS_DIR"
echo "  darcy ckpt  : $DARCY_CKPT"
echo "  helm  ckpt  : $HELM_CKPT"
echo ""

echo "=== [1/2] Darcy ==="
bash "$DIR/run_darcy.sh" --checkpoint "$DARCY_CKPT" --num_samples "$NUM_SAMPLES" --burn "$BURN" --figs_dir "$FIGS_DIR"

echo "=== [2/2] Helmholtz ==="
bash "$DIR/run_helmholz.sh" --checkpoint "$HELM_CKPT" --num_samples "$NUM_SAMPLES" --burn "$BURN" --figs_dir "$FIGS_DIR"

echo "=== test_run.sh complete — figures saved to: $FIGS_DIR ==="
