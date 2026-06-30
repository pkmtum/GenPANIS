# GenPANIS

**GenPANIS: A latent-variable generative framework for forward and inverse PDE problems in multiphase media**

> M. Chatzopoulos and P.-S. Koutsourelakis, *GenPANIS: A latent-variable generative framework for forward and inverse PDE problems in multiphase media*, **Journal of Computational Physics** 564 (2026) 115140.
> [https://doi.org/10.1016/j.jcp.2026.115140](https://doi.org/10.1016/j.jcp.2026.115140)

---

## Description

GenPANIS is a unified generative framework that jointly models microstructures and PDE solutions in multiphase media. By learning a joint distribution over a latent microstructure embedding and a PDE solution field, the model enables **bidirectional inference in a single architecture**:

- **Forward problem**: given a microstructure, predict the PDE solution distribution.
- **Inverse problem**: given (partial/sparse) PDE observations, infer the microstructure posterior.

Key design choices:
- **Physics-aware decoder** — a differentiable coarse-grained finite-element solver enforces the governing PDE at training time, so no labeled solution data are required for training.
- **Normalizing flow prior** (RealNVP) — exact log-likelihood on the latent space enables gradient-based posterior inference via Hamiltonian Monte Carlo (HMC).
- **Continuous latent embedding** — discrete binary microstructures are embedded in a smooth latent space, preserving their exact geometry while allowing gradient flow.

Demonstrated on 2-D **Darcy flow** and **Helmholtz equation** in two-phase media. Compared against PINO and FunDPS:

| Model | Parameters | Training time |
|-------|-----------|---------------|
| **GenPANIS** | **2.9 M** | **~0.6 h** |
| PINO | 13.1 M | ~38 h |
| FunDPS | 183.3 M | ~45 h |

Representative results:
- Pixel Accuracy (PA) ≥ 0.96 on within-distribution inverse problems (Darcy, full observations)
- PA ≥ 0.92 on Helmholtz inverse problems
- Robust to out-of-distribution boundary conditions and volume fractions
- Works with sparse and randomly placed observations

## Badges

[![Journal](https://img.shields.io/badge/JCP-115140-blue)](https://doi.org/10.1016/j.jcp.2026.115140)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Visuals

![Inverse problem: partial observations — PINO vs FunDPS vs GenPANIS on fixed grids (11×11, 7×7, 5×5)](assets/cover.png)

*Microstructure reconstruction from partial pressure observations. Columns: ground truth microstructure, PINO point estimate, FunDPS posterior mean and std, GenPANIS posterior mean and std. GenPANIS achieves PA ≥ 0.93 even at the coarsest 5×5 observation grid, while preserving sharp phase boundaries and providing well-calibrated uncertainty estimates.*

Paper figures (available at [https://doi.org/10.1016/j.jcp.2026.115140](https://doi.org/10.1016/j.jcp.2026.115140)):

| Figure | Content |
|--------|---------|
| Fig. 2 | Inference time comparison across methods |
| Figs. 3, 5, 6 | Forward and inverse results, full observations (Darcy and Helmholtz) |
| Figs. 9, 10 | Partial observations (11×11 grid) |
| Figs. 11, 12 | Random sparse observations (40 points) |
| Fig. 13 | Data efficiency: labeled samples vs. accuracy |
| Figs. 15–18 | Out-of-distribution boundary conditions and volume fractions |

## Installation

**The only requirement is [Docker](https://docs.docker.com/get-docker/).** Pull the pre-built image:

```bash
docker pull mattchatz/fenics
```

Then clone this repository and launch a container with the repo mounted:

```bash
git clone https://github.com/pkmtum/GenPANIS.git
cd GenPANIS
docker run --rm -it --gpus all -v "$(pwd)":/workspace mattchatz/fenics
```

All dependencies (FEniCS, PyTorch, and everything else required) are already included in the image. No conda setup or manual package installation is needed.

## Data

The large data files (training datasets, GP covariance matrices, reference MCMC samples) are hosted on Zenodo:

> **[https://zenodo.org/records/21068584](https://zenodo.org/records/21068584)**

Download `genpanisData.tar` from that page and place it in the repository root. Then run:

```bash
bash decompress_data.sh
```

This extracts and decompresses all files to their correct locations. The repository is then in a **ready-to-run** state — inference, training, and the Gibbs sampler will all work out of the box.

## Usage

### Pretrained checkpoints

Pretrained checkpoints are included in the repository and are ready to use out of the box:

- `checkpoints/darcy10k_pretrained.ckpt` — Darcy flow model
- `checkpoints/helmholz10k_pretrained.ckpt` — Helmholtz equation model

You can go straight to inference without training anything.

### 1. Run inference

All inference scripts **automatically discover the most recently trained checkpoint** for their PDE type — no path configuration needed.

Two bash scripts cover the full experiment suite:

```bash
bash experiments/demo/run_darcy.sh    --num_samples <N> --burn <B>
bash experiments/demo/run_helmholz.sh --num_samples <N> --burn <B>
```

Or run the full suite (Darcy + Helmholtz) in one go with `test_run.sh`:

```bash
bash experiments/demo/test_run.sh --num_samples <N> --burn <B>
```

**Quick sanity check** — fast run to verify the pipeline end-to-end:

```bash
bash experiments/demo/test_run.sh --num_samples 200 --burn 100
```

**Full run** — use enough samples for HMC convergence:

```bash
bash experiments/demo/test_run.sh --num_samples 15000 --burn 10000
```

Optionally save figures to a custom directory (avoids overwriting previous results):

```bash
bash experiments/demo/test_run.sh \
    --num_samples 15000 --burn 10000 --figs_dir experiments/demo/figs_v2
```

All output plots are saved to `experiments/demo/figs/` by default, with unique filenames per experiment (e.g. `resInverse_partialObs_11x11.png`).

Individual scripts can also be run directly. They auto-discover the checkpoint, but you can override it:

```bash
# Auto-discover latest darcy checkpoint
python3 experiments/demo/outVf.py --num_samples 15000 --burn 10000

# Or point to a specific checkpoint
python3 experiments/demo/outVf.py \
    --checkpoint checkpoints/darcy10k_pretrained.ckpt \
    --num_samples 15000 --burn 10000
```

### 2. Retrain (optional)

If you want to train your own models from scratch, two training scripts are provided:

```bash
# Darcy flow
python experiments/demo/train_darcy.py

# Helmholtz equation
python experiments/demo/train_helmholz.py
```

New checkpoints are saved under `checkpoints/genPANIS_darcy_10000/` and `checkpoints/genPANIS_helmholz_10000/` respectively. The inference scripts automatically pick up the most recently saved checkpoint, so re-running inference after training will use the new weights.

#### Mixed-data training (labeled + unlabeled + virtual observables)

[`experiments/demo/train_darcy_mixed.py`](experiments/demo/train_darcy_mixed.py) demonstrates training on a mixture of three data regimes simultaneously:

| Split | Samples | What the model sees |
|-------|---------|---------------------|
| Labeled | 3 000 | Microstructure + PDE solution — full supervision |
| Unlabeled | 3 000 | Microstructure only — VAE prior + reconstruction, no physics |
| Virtual observables (VO) | 3 000 | Microstructure only — physics residual enforced via the differentiable FE solver, no solution labels |

```bash
python experiments/demo/train_darcy_mixed.py --epochs 500
```

Checkpoints are saved under `checkpoints/genPANIS_darcy_3k_mixed/`.

> **Note on training speed.** Expect the mixed run to be noticeably slower per epoch than the labeled-only baseline. The bottleneck is the VO term: for every VO batch the model must evaluate the PDE residual by numerically integrating the terms of the weighted residuals.

### 3. True posterior — Gibbs sampler

[`experiments/truePosterior/gibbsSampler.py`](experiments/truePosterior/gibbsSampler.py) runs a pixel-wise Gibbs sampler on the exact posterior over the Darcy microstructure, using the last sample of the dataset as the test case. Unlike GenPANIS inference (HMC in latent space), this operates directly on the binary pixel field and calls the FEniCS solver at every step — providing a reference true posterior for comparison.

```bash
python experiments/truePosterior/gibbsSampler.py \
    --n_iter 200 --burn_in 50 --grid_size 32 --snr_db 20
```

Outputs are saved to `results/figs/gibbs_posterior_stats.png` and `gibbs_ergodic_mean.png`. To reload previously saved samples without re-running the sampler, add `--load`.

> **Note on speed.** Each Gibbs sweep visits every pixel and calls the FE solver twice per pixel (once for value 0, once for 1). At the default `--grid_size 32` this is ~2 000 solver calls per sweep. Use `--grid_size 8` or `--grid_size 16` for quick tests.

## Authors and acknowledgment

- Matthaios Chatzopoulos (TU Munich)
- Phaedon-Stelios Koutsourelakis (TU Munich)

If you use this code, please cite:

```bibtex
@article{chatzopoulos2026genpannis,
  title   = {{GenPANIS}: A latent-variable generative framework for forward and inverse {PDE} problems in multiphase media},
  author  = {Chatzopoulos, Matthaios and Koutsourelakis, Phaedon-Stelios},
  journal = {Journal of Computational Physics},
  volume  = {564},
  pages   = {115140},
  year    = {2026},
  doi     = {10.1016/j.jcp.2026.115140}
}
```

## License

MIT
