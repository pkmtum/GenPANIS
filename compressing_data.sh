#!/bin/bash
# Compresses the 5 large files (>100 MB) into a single genpanisData.tar.
# Run from the repository root: bash compressing_data.sh
#
# Steps:
#   1. gzip each large file in place (removes original, creates .gz beside it)
#   2. collect all .gz files into genpanisData.tar (preserving directory structure)
#   3. remove intermediate .gz files
#
# Upload genpanisData.tar to Zenodo. After cloning, run decompress_data.sh.

set -e

TAR="genpanisData.tar"

# ── Step 1: gzip each large file ──────────────────────────────────────────────

echo ""
echo "=== Step 1: gzipping large files ==="
echo ""

gzip_file() {
    local src="$1"
    if [ ! -f "$src" ]; then
        echo "ERROR: file not found: $src"
        exit 1
    fi
    local size
    size=$(du -sh "$src" | cut -f1)
    echo "  →  $src  ($size)"
    gzip "$src"
    echo "     done  →  ${src}.gz"
}

gzip_file "data/labeledData20000_129.pth"
gzip_file "data/labeledData20000Helm_129.pth"
gzip_file "model/pde/RefSolutions/condFields/gpCov.dat"
gzip_file "model/pde/RefSolutions/condFields/gpCovDetailed.dat"
gzip_file "experiments/truePosterior/darcy/mcmcSamples_iter1000000_1chain64x64.pt"

# ── Step 2: tar all .gz files ─────────────────────────────────────────────────

echo ""
echo "=== Step 2: creating $TAR ==="
echo ""

tar -cf "$TAR" \
    "data/labeledData20000_129.pth.gz" \
    "data/labeledData20000Helm_129.pth.gz" \
    "model/pde/RefSolutions/condFields/gpCov.dat.gz" \
    "model/pde/RefSolutions/condFields/gpCovDetailed.dat.gz" \
    "experiments/truePosterior/darcy/mcmcSamples_iter1000000_1chain64x64.pt.gz"

size=$(du -sh "$TAR" | cut -f1)
echo "  created $TAR  ($size)"

# ── Step 3: remove intermediate .gz files ─────────────────────────────────────

echo ""
echo "=== Step 3: removing intermediate .gz files ==="
echo ""

rm "data/labeledData20000_129.pth.gz"
rm "data/labeledData20000Helm_129.pth.gz"
rm "model/pde/RefSolutions/condFields/gpCov.dat.gz"
rm "model/pde/RefSolutions/condFields/gpCovDetailed.dat.gz"
rm "experiments/truePosterior/darcy/mcmcSamples_iter1000000_1chain64x64.pt.gz"
echo "  done"

echo ""
echo "=== Done. Upload $TAR to Zenodo. ==="
