import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from input import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=20000)
parser.add_argument('--burn',        type=int, default=10000)
parser.add_argument('--checkpoint',  type=str, default=None,
                    help='Path to checkpoint (default: auto-discover latest helmholz checkpoint)')
parser.add_argument('--figs_dir',    type=str, default=None,
                    help='Directory to save output figures (default: session_dir/figs)')
args = parser.parse_args()

# Override to Helmholtz before constructing pde
options['boundaryCondition'] = 1.
options['pde_type'] = 'helmholz'

from utils.variousFunctions import setupDevice, find_latest_checkpoint
from model.pde.pdeForm2D import pdeForm
from model.ProbModels.cgCnnMvn import probabModel
from utils.PostProcessing import postProcessing
from model.pde.pdeTrueSolFenics import solve_pde_Helmholz

# ── Device ────────────────────────────────────────────────────────────────────
device = setupDevice(cudaIndex, device, dataType)

# ── Session paths ─────────────────────────────────────────────────────────────
SESSION_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR    = args.figs_dir if args.figs_dir else os.path.join(SESSION_DIR, 'figs')

os.makedirs(FIGS_DIR, exist_ok=True)
post = postProcessing(path='./results/data/', fpath=FIGS_DIR + '/', displayPlots=False, cleanFigs=False)

# ── PDE (Helmholtz) ───────────────────────────────────────────────────────────
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp,
              createNewCondField, device, post,
              rhs=rhs, reducedDim=reducedDim, options=options)

# ── Model ─────────────────────────────────────────────────────────────────────
samples = probabModel(pde, poly_pow=poly_pow, stdInit=stdInit, gradLr=gradLr,
                      lr=lr, sigma_r=sigma_r, sigma_w=sigma_w, yFMode=yFMode,
                      randResBatchSize=randResBatchSize, reducedDim=reducedDim,
                      dimz=60)

CHECKPOINT = args.checkpoint or find_latest_checkpoint('helmholz')
ckpt = torch.load(CHECKPOINT, map_location=device)
samples.load_state_dict(ckpt['state_dict'])
samples.neuralNet.eval()
samples.flow.eval()
samples.xDecoder.eval()
samples.ztoXDecoder.eval()
samples.to(device)

# ── Dataset statistics (Helmholtz training distribution) ──────────────────────
dxy = (torch.load('./data/labeledData20000Helm_129.pth', map_location=device)
       .unsqueeze(-3).flatten(-2)[:, :200, :, :].to(dtype=torch.float32))
samples.ymin  = dxy[1].min()
samples.ymax  = dxy[1].max()
samples.ystd  = dxy[1].std()
samples.ymean = dxy[1].mean()

# ── Out-of-distribution test pair (VF10) ──────────────────────────────────────
data = torch.load(os.path.join(SESSION_DIR, 'vf10_helmholz.pt'),
                  map_location=device).clone().detach()

# Interpolate x to the FEniCS mesh resolution (128×128), solve, then back to integration grid
x = torch.nn.functional.interpolate(
    data[0].reshape(1, 1, data[0].size(0), data[0].size(1)),
    size=(128, 128), mode='bilinear', align_corners=True
).squeeze(0).squeeze(0)

y = solve_pde_Helmholz(
    x.cpu().numpy(), pde.rhs,
    k2=(1 / x ** 1.30103).cpu(),
    uBc=pde.uBc, options=pde.optionsMod, idx=pde.idx
).to(device)

data[1] = torch.nn.functional.interpolate(
    y.reshape(1, 1, y.size(0), y.size(1)),
    size=(pde.sgrid.size(-1), pde.sgrid.size(-1)),
    mode='bilinear', align_corners=True
).squeeze(0).squeeze(0)

torch.save(data.clone().detach(), os.path.join(SESSION_DIR, 'outVf_helmholz_pair.pt'))

# ── Inverse problem HMC (Helmholtz, out-of-distribution VF) ──────────────────
samples.InverseProblemHMC(
    y_obs=data[1].unsqueeze(0).unsqueeze(0), x_true=data[0],
    num_samples=args.num_samples, burn=args.burn, SNR_in_dB=20, L=3, step_size=0.001,
    pde='helmholz',
    path=os.path.join(SESSION_DIR, 'outVf_helmholz.pt'),
)

# ── Plot ──────────────────────────────────────────────────────────────────────
post.plotInversePaper(
    data_path=os.path.join(SESSION_DIR, 'outVf_helmholz.pt'), name='outVf_helmholz'
)

print(f"Plots saved to: {FIGS_DIR}")
