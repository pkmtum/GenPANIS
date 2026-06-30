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
parser.add_argument('--num_samples', type=int, default=10000)
parser.add_argument('--burn',        type=int, default=3000)
parser.add_argument('--checkpoint',  type=str, default=None,
                    help='Path to checkpoint (default: auto-discover latest darcy checkpoint)')
parser.add_argument('--figs_dir',    type=str, default=None,
                    help='Directory to save output figures (default: session_dir/figs)')
args = parser.parse_args()
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

# ── PDE ───────────────────────────────────────────────────────────────────────
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp,
              createNewCondField, device, post,
              rhs=rhs, reducedDim=reducedDim, options=options)

# ── Model ─────────────────────────────────────────────────────────────────────
samples = probabModel(pde, poly_pow=poly_pow, stdInit=stdInit, gradLr=gradLr,
                      lr=lr, sigma_r=sigma_r, sigma_w=sigma_w, yFMode=yFMode,
                      randResBatchSize=randResBatchSize, reducedDim=reducedDim,
                      dimz=60)

CHECKPOINT = args.checkpoint or find_latest_checkpoint('darcy')
ckpt = torch.load(CHECKPOINT, map_location=device)
samples.load_state_dict(ckpt['state_dict'])
samples.neuralNet.eval()
samples.flow.eval()
samples.xDecoder.eval()
samples.ztoXDecoder.eval()
samples.to(device)

# ── Dataset statistics required by the model ──────────────────────────────────
dxy = (torch.load('./data/labeledData20000_129.pth', map_location=device)
       .unsqueeze(-3).flatten(-2)[:, :200, :, :].to(dtype=torch.float32))
samples.ymin  = dxy[1].min()
samples.ymax  = dxy[1].max()
samples.ystd  = dxy[1].std()
samples.ymean = dxy[1].mean()

# ── Test pair ─────────────────────────────────────────────────────────────────
data = torch.load(os.path.join(SESSION_DIR, 'testPair.pt'),
                  map_location=device)

# ── Observation grid (11×11, stride=12) ───────────────────────────────────────
def make_observationGrid(stride, data):
    H, W = data[1].size(-1), data[1].size(-1)
    start = int(torch.floor(torch.tensor(5 / 2)))
    obs_rows = torch.arange(start, H, stride)
    obs_cols = torch.arange(start, W, stride)
    yy, xx = torch.meshgrid(obs_rows, obs_cols, indexing='ij')
    flat_indices = yy * W + xx
    mask = torch.zeros(H * W)
    mask[flat_indices.flatten()] = 1.
    return flat_indices.flatten(), mask.reshape(H, W)

observationGrid = make_observationGrid(stride=12, data=data)[0]

# ── Inverse problem HMC (partial observations) ────────────────────────────────
samples.InverseProblemHMC(
    y_obs=data[1].unsqueeze(0).unsqueeze(0), x_true=data[0],
    num_samples=args.num_samples, burn=args.burn, SNR_in_dB=20, L=3, step_size=0.001,
    observationGrid=observationGrid,
    path=os.path.join(SESSION_DIR, 'partialObs_11x11.pt'),
)

# ── Plot ──────────────────────────────────────────────────────────────────────
post.plotInversePaper(
    data_path=os.path.join(SESSION_DIR, 'partialObs_11x11.pt'), name='partialObs_11x11'
)

print(f"Plots saved to: {FIGS_DIR}")
