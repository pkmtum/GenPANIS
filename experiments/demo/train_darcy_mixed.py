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

parser = argparse.ArgumentParser(
    description='Train genPANIS on Darcy flow with 3000 labeled + 3000 unlabeled + 3000 VO samples')
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

# ── PDE (Darcy) ───────────────────────────────────────────────────────────────
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp,
              createNewCondField, device, post,
              rhs=rhs, reducedDim=reducedDim, options=options)

# ── Probabilistic model ───────────────────────────────────────────────────────
samples = probabModel(pde, poly_pow=poly_pow, stdInit=stdInit, gradLr=gradLr,
                      lr=lr, sigma_r=sigma_r, sigma_w=sigma_w, yFMode=yFMode,
                      randResBatchSize=randResBatchSize, reducedDim=reducedDim,
                      dimz=60, useL=True, useVO=True, useUnlabeled=True)

# ── Load raw data: shape [2, 20000, 1, 65*65] ────────────────────────────────
# [0] = microstructures, [1] = PDE solutions
raw = (torch.load('./data/labeledData20000_129.pth', map_location=device)
       .unsqueeze(-3).flatten(-2).to(dtype=torch.float32))

# ── Three non-overlapping 3000-sample chunks ──────────────────────────────────
raw_l  = raw[:, :6000]        # labeled:    indices 0–2999
raw_un = raw[:, 6000:12000]    # unlabeled:  indices 3000–5999
raw_vo = raw[:, 12000:18000]    # VO:         indices 6000–8999

# ── Labeled statistics ────────────────────────────────────────────────────────
samples.ymin  = raw_l[1].min()
samples.ymax  = raw_l[1].max()
samples.ystd  = raw_l[1].std()
samples.ymean = raw_l[1].mean()

# ── Helper: augment microstructure with volume-fraction logit feature ─────────
def add_vf_feature(x):
    # x: [N, 1, 65*65]
    u = torch.sum(torch.where(x > 0.9, 1., 0.).flatten(-2) / x.flatten(-2).size(-1), dim=-1)
    u = -torch.log(1. / u - 1.)
    return torch.cat((x, u.reshape(-1, 1, 1)), dim=-1)  # [N, 1, 65*65+1]

# ── Labeled: microstructure + VF feature, paired with PDE solution ────────────
dxyu_l = add_vf_feature(raw_l[0])   # [3000, 1, 4226]
dy_l   = raw_l[1]                    # [3000, 1, 4225]

# ── Unlabeled: raw microstructures only, no VF feature, no PDE solution ───────
# training_step reads this as x_unlabeled.reshape(-1, 1, grid, grid) directly
dx_un = raw_un[0]                    # [3000, 1, 4225]

# ── VO: microstructure + VF feature, no PDE solution (physics residual only) ──
# training_step strips the last element via x_vo[..., :-1] before reshaping
dxyu_vo = add_vf_feature(raw_vo[0]) # [3000, 1, 4226]

# ── Dataset splits ────────────────────────────────────────────────────────────
torch.cuda.manual_seed_all(42)
generator = torch.Generator(device=device)

labeled_dataset = torch.utils.data.TensorDataset(dxyu_l, dy_l)
train_size = int(0.5 * len(labeled_dataset))               # 1500
val_size   = int(0.3 * len(labeled_dataset))               #  900
test_size  = len(labeled_dataset) - train_size - val_size  #  600
train_set, val_set, test_set = torch.utils.data.random_split(
    labeled_dataset, [train_size, val_size, test_size], generator=generator)

unlabeled_dataset = torch.utils.data.TensorDataset(dx_un, torch.zeros(dx_un.size(0)))
vodataset         = torch.utils.data.TensorDataset(dxyu_vo, torch.zeros(dxyu_vo.size(0)))

# ── DataLoaders ───────────────────────────────────────────────────────────────
batchsize = 50

train_loader     = torch.utils.data.DataLoader(
    train_set, batch_size=batchsize, shuffle=True, drop_last=False, generator=generator)
unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_dataset, batch_size=batchsize, shuffle=True, generator=generator)
vo_loader        = torch.utils.data.DataLoader(
    vodataset, batch_size=batchsize, shuffle=True, generator=generator)

samples.unlabeled_loader = unlabeled_loader
samples.unlabeled_iter   = iter(unlabeled_loader)
samples.vo_loader        = vo_loader
samples.vo_iter          = iter(vo_loader)

# ── Trainer ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints"
model_name      = "genPANIS_darcy_3k_mixed"

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
    print(f"Resuming Darcy (mixed) training from: {args.resume}")
    trainer.fit(samples, train_loader, ckpt_path=args.resume)
else:
    print("Starting Darcy (mixed) training from scratch (3000 labeled + 3000 unlabeled + 3000 VO)...")
    trainer.fit(samples, train_loader)

print("Darcy (mixed) training finished.")
