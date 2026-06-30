#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Helmholtz equation experiments ==="

echo "[1/5] fullObservationsHelmholz"
python3 "$DIR/fullObservationsHelmholz.py" "$@"

echo "[2/5] partialObs_11x11_helmholz"
python3 "$DIR/partialObs_11x11_helmholz.py" "$@"

echo "[3/5] randomObs_40_helmholz"
python3 "$DIR/randomObs_40_helmholz.py" "$@"

echo "[4/5] outVf_helmholz"
python3 "$DIR/outVf_helmholz.py" "$@"

echo "[5/5] outBc_helmholz"
python3 "$DIR/outBc_helmholz.py" "$@"

echo "=== All Helmholtz experiments done. Plots in $DIR/figs/ ==="
