import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from input import *
from utils.variousFunctions import setupDevice
from model.pde.pdeForm2D import pdeForm
from model.ProbModels.cgCnnMvn import probabModel
from utils.PostProcessing import postProcessing

parser = argparse.ArgumentParser(description='Train genPANIS on Darcy flow (10 000 labeled samples)')
parser.add_argument('--epochs', type=int, default=500,
                    help='Total number of training epochs (default: 500)')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to a .ckpt file to resume training from')
parser.add_argument('--cuda_index', type=int, default=0,
                    help='CUDA device index to train on (default: 0)')
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
cudaIndex = args.cuda_index
device = setupDevice(cudaIndex, device, dataType)

# ── Post-processing helper (needed by pdeForm) ────────────────────────────────
post = postProcessing(path='./results/data/', displayPlots=False)

# ── PDE (Darcy: BC = 0., pde_type already set in input.py) ───────────────────
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp,
              createNewCondField, device, post,
              rhs=rhs, reducedDim=reducedDim, options=options)

# ── Probabilistic model ───────────────────────────────────────────────────────
samples = probabModel(pde, poly_pow=poly_pow, stdInit=stdInit, gradLr=gradLr,
                      lr=lr, sigma_r=sigma_r, sigma_w=sigma_w, yFMode=yFMode,
                      randResBatchSize=randResBatchSize, reducedDim=reducedDim,
                      dimz=60)

# ── Labeled dataset: all 20 000 samples; 50 % split → 10 000 for training ────
dxy = (torch.load('./data/labeledData20000_129.pth', map_location=device)
       .unsqueeze(-3).flatten(-2)[:, :20000, :, :].to(dtype=torch.float32))

samples.ymin  = dxy[1].min()
samples.ymax  = dxy[1].max()
samples.ystd  = dxy[1].std()
samples.ymean = dxy[1].mean()

# Augment x with volume-fraction logit feature (matches main.py)
u = torch.sum(torch.where(dxy[0] > 0.9, 1., 0.).flatten(-2) / dxy[0].flatten(-2).size(-1), dim=-1)
u = -torch.log(1. / u - 1.)
dxyu = torch.cat((dxy[0], u.reshape(-1, 1, 1)), dim=-1).to(dtype=torch.float32)

# ── Virtual-observables dataset: all 20 000 samples ──────────────────────────
dxyvo = (torch.load('./data/labeledData20000_129.pth', map_location=device)
         .unsqueeze(-3).flatten(-2)[:, :20000, :, :].to(dtype=torch.float32))
u_vo = torch.sum(torch.where(dxyvo[0] > 0.9, 1., 0.).flatten(-2) / dxyvo[0].flatten(-2).size(-1), dim=-1)
u_vo = -torch.log(1. / u_vo - 1.)
dxyu_vo = torch.cat((dxyvo[0], u_vo.reshape(-1, 1, 1)), dim=-1).to(dtype=torch.float32)

# ── Dataset splits ────────────────────────────────────────────────────────────
torch.cuda.manual_seed_all(42)
generator = torch.Generator(device=device)

dataset    = torch.utils.data.TensorDataset(dxyu, dxy[1])
train_size = int(0.5 * len(dataset))               # 10 000
val_size   = int(0.3 * len(dataset))               #  6 000
test_size  = len(dataset) - train_size - val_size  #  4 000
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size], generator=generator)

vodataset    = torch.utils.data.TensorDataset(dxyu_vo, torch.zeros(dxyvo.size(1)))
votrain_size = int(0.5 * len(vodataset))
voval_size   = int(0.3 * len(vodataset))
votest_size  = len(vodataset) - votrain_size - voval_size
votrain_set, _, _ = torch.utils.data.random_split(
    vodataset, [votrain_size, voval_size, votest_size], generator=generator)

# ── DataLoaders ───────────────────────────────────────────────────────────────
batchsize    = 250
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batchsize, shuffle=True, drop_last=False, generator=generator)
vo_loader    = torch.utils.data.DataLoader(
    votrain_set, batch_size=batchsize // 5, shuffle=True, generator=generator)

samples.vo_loader = vo_loader
samples.vo_iter   = iter(vo_loader)

# ── Trainer ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints"
model_name      = "genPANIS_darcy_10000"

trainer = pl.Trainer(
    default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=args.epochs,
    callbacks=[
        ModelCheckpoint(every_n_epochs=args.epochs - 1),
        LearningRateMonitor("epoch"),
    ],
)
trainer.logger._log_graph = True
trainer.logger._default_hp_metric = None

if args.resume:
    print(f"Resuming Darcy training from: {args.resume}")
    trainer.fit(samples, train_loader, ckpt_path=args.resume)
else:
    print("Starting Darcy training from scratch...")
    trainer.fit(samples, train_loader)

print("Darcy training finished.")
