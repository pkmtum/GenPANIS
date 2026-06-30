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

# ── Out-of-distribution test pair (linear boundary condition) ─────────────────
data = torch.load(os.path.join(SESSION_DIR, 'invPairH.pt'),
                  map_location=device).clone().detach()

# Recompute the PDE solution under the linear (out-of-distribution) BC
yFenics = solve_pde_Helmholz(
    data[0], rhs=1.,
    k2=(1 / data[0] ** 1.30103).to(dtype=torch.float64).clone().detach().cpu().t(),
    mode='Linear', options=options, plotSol=False
).t()
data[1] = yFenics.to(device)

# Build the linear BC tensor and inject it into the neural network
f   = 5.
z   = f * pde.shapeFunc.sgridRx + f * pde.shapeFunc.sgridRy
uBc = z.flatten()[pde.shapeFunc.bcNodesOrder]
samples.neuralNet.uBc = uBc

torch.save(data.clone().detach(), os.path.join(SESSION_DIR, 'outBcPair_helmholz.pt'))

# CG reference solution under the new BC
ucg = pde.shapeFunc.solveCGPDE_helmholz(c_x=data[0], f=100., uBc=uBc)

# ── Inverse problem HMC (Helmholtz, out-of-distribution BC) ──────────────────
samples.InverseProblemHMC(
    y_obs=data[1].unsqueeze(0).unsqueeze(0), x_true=data[0],
    num_samples=args.num_samples, burn=args.burn, SNR_in_dB=20, L=3, step_size=0.001,
    pde='helmholz',
    path=os.path.join(SESSION_DIR, 'outBc_helmholz.pt'),
)

# ── Plot ──────────────────────────────────────────────────────────────────────
post.plotInversePaper(
    data_path=os.path.join(SESSION_DIR, 'outBc_helmholz.pt'), name='outBc_helmholz'
)

print(f"Plots saved to: {FIGS_DIR}")
