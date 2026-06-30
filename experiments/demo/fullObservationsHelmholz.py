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

# ── Test pair ─────────────────────────────────────────────────────────────────
data = torch.load(os.path.join(SESSION_DIR, 'invPairH.pt'),
                  map_location=device)

# ── Forward problem HMC (Helmholtz, SNR = 40 dB) ─────────────────────────────
samples.ForwardProblemHMC(
    y_true=data[1].unsqueeze(0).unsqueeze(0), x_obs=data[0],
    num_samples=args.num_samples, burn=args.burn, L=3, step_size=0.001, SNR_in_dB=40,
    pde='helmholz',
    path=os.path.join(SESSION_DIR, 'f_helmholz.pt'),
)

# ── Inverse problem HMC (Helmholtz, SNR = 10 dB) ─────────────────────────────
samples.InverseProblemHMC(
    y_obs=data[1].unsqueeze(0).unsqueeze(0), x_true=data[0],
    num_samples=args.num_samples, burn=args.burn, SNR_in_dB=20, L=3, step_size=0.001,
    pde='helmholz',
    path=os.path.join(SESSION_DIR, 'Mi_helmholz.pt'),
)

# ── Plots ─────────────────────────────────────────────────────────────────────
post.plotForwardPaper(
    data_path=os.path.join(SESSION_DIR, 'f_helmholz.pt'), name='fullObsHelmholz'
)
post.plotInversePaper(
    data_path=os.path.join(SESSION_DIR, 'Mi_helmholz.pt'), name='fullObsHelmholz'
)

print(f"Plots saved to: {FIGS_DIR}")
