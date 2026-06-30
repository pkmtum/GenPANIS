#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Darcy flow experiments ==="

echo "[1/5] fullObservations"
python3 "$DIR/fullObservations.py" "$@"

echo "[2/5] partialObs_11x11"
python3 "$DIR/partialObs_11x11.py" "$@"

echo "[3/5] randomObs_40"
python3 "$DIR/randomObs_40.py" "$@"

echo "[4/5] outVf"
python3 "$DIR/outVf.py" "$@"

echo "[5/5] outBc"
python3 "$DIR/outBc.py" "$@"

echo "=== All Darcy experiments done. Plots in $DIR/figs/ ==="
