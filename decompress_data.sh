#!/bin/bash
# Decompresses genpanisData.tar and restores all large files to their original locations.
# Run from the repository root after cloning: bash decompress_data.sh
#
# Before running, download genpanisData.tar from Zenodo and place it in the repo root.
#
# Steps:
#   1. extract genpanisData.tar (.gz files land in their correct subdirectories)
#   2. gunzip each .gz file in place (removes .gz, creates the original file)
#   3. remove genpanisData.tar

set -e

TAR="genpanisData.tar"

# ── Step 1: extract the tar ───────────────────────────────────────────────────

echo ""
echo "=== Step 1: extracting $TAR ==="
echo ""

if [ ! -f "$TAR" ]; then
    echo "ERROR: $TAR not found in the current directory."
    echo "Download it from https://zenodo.org/records/21068584 and place it here before running this script."
    exit 1
fi

tar -xf "$TAR"
echo "  extracted:"
echo "    data/labeledData20000_129.pth.gz"
echo "    data/labeledData20000Helm_129.pth.gz"
echo "    model/pde/RefSolutions/condFields/gpCov.dat.gz"
echo "    model/pde/RefSolutions/condFields/gpCovDetailed.dat.gz"
echo "    experiments/truePosterior/darcy/mcmcSamples_iter1000000_1chain64x64.pt.gz"

# ── Step 2: gunzip each file in place ─────────────────────────────────────────

echo ""
echo "=== Step 2: decompressing .gz files ==="
echo ""

gunzip_file() {
    local gz="$1"
    if [ ! -f "$gz" ]; then
        echo "ERROR: file not found after tar extraction: $gz"
        exit 1
    fi
    echo "  →  $gz"
    gunzip "$gz"
    local dest="${gz%.gz}"
    local size
    size=$(du -sh "$dest" | cut -f1)
    echo "     done  →  $dest  ($size)"
}

gunzip_file "data/labeledData20000_129.pth.gz"
gunzip_file "data/labeledData20000Helm_129.pth.gz"
gunzip_file "model/pde/RefSolutions/condFields/gpCov.dat.gz"
gunzip_file "model/pde/RefSolutions/condFields/gpCovDetailed.dat.gz"
gunzip_file "experiments/truePosterior/darcy/mcmcSamples_iter1000000_1chain64x64.pt.gz"

# ── Step 3: remove the tar ────────────────────────────────────────────────────

echo ""
echo "=== Step 3: cleanup ==="
echo ""

rm "$TAR"
echo "  removed $TAR"

echo ""
echo "=== Done. All files are in their original locations: ==="
echo "    data/labeledData20000_129.pth"
echo "    data/labeledData20000Helm_129.pth"
echo "    model/pde/RefSolutions/condFields/gpCov.dat"
echo "    model/pde/RefSolutions/condFields/gpCovDetailed.dat"
echo "    experiments/truePosterior/darcy/mcmcSamples_iter1000000_1chain64x64.pt"
