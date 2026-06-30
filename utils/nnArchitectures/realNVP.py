from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision
import torch.utils.data as data
import torch.optim as optim
import torch
import torch.nn as nn
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.distributions.multivariate_normal import MultivariateNormal


class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size):
        """
        mask: Tensor of shape [D]
        hidden_size: int, number of hidden units in the s and t networks
        """
        super().__init__()
        self.dim = len(mask)  # D
        self.mask = nn.Parameter(mask, requires_grad=False)  # [D]

        self.s_func = nn.Sequential(
            nn.Linear(self.dim, hidden_size),  # [B, D] -> [B, H]
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),  # [B, H] -> [B, H]
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.dim)  # [B, H] -> [B, D]
        )

        self.scale = nn.Parameter(torch.ones(self.dim))  # [D]

        self.t_func = nn.Sequential(
            nn.Linear(self.dim, hidden_size),  # [B, D] -> [B, H]
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),  # [B, H] -> [B, H]
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.dim)  # [B, H] -> [B, D]
        )

    def forward(self, x):
        """
        x: [B, D]
        Returns:
            y: [B, D]
            log_det_jac: [B]
        """
        x_mask = x * self.mask  # [B, D]
        s = torch.tanh(self.s_func(x_mask)) * self.scale # [B, D]
        t = self.t_func(x_mask)  # [B, D]

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)  # [B, D]
        log_det_jac = ((1 - self.mask) * s).sum(-1)  # [B]
        return y, log_det_jac

    def inverse(self, y):
        """
        y: [B, D]
        Returns:
            x: [B, D]
            inv_log_det_jac: [B]
        """
        y_mask = y * self.mask  # [B, D]
        s = torch.tanh(self.s_func(y_mask)) * self.scale
        t = self.t_func(y_mask)  # [B, D]

        x = y_mask + (1 - self.mask) * (y - t) * torch.exp(-s)  # [B, D]
        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)  # [B]
        return x, inv_log_det_jac



class RealNVP(pl.LightningModule):
    def __init__(self, masks, hidden_size, lr=1e-3, use_flow=True):
        """
        masks: list of binary masks, each of shape [D]
        hidden_size: int, hidden size in each RealNVPNode
        lr: float, learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.dim = len(masks[0])  # D
        self.hidden_size = hidden_size
        self.lr = lr
        self.use_flow=use_flow

        #self.polyCoeff = nn.Parameter(torch.randn(2))

        self.masks = nn.ParameterList([
            nn.Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False) for mask in masks
        ])  # List of [D]

        self.layers = nn.ModuleList([
            RealNVPNode(mask, hidden_size) for mask in self.masks
        ])  # List of RealNVPNode

        self.distribution = MultivariateNormal(torch.zeros(self.dim).to('cuda:0'), torch.eye(self.dim).to('cuda:0'))  # base distribution

    def log_probability(self, x):
        """
        x: [B, D]
        Returns:
            log_prob: [B]
        """
        #x = x[0]
        if self.use_flow:
            log_prob = torch.zeros(x.shape[0], device=x.device)  # [B]
            for layer in reversed(self.layers):
                x, inv_log_det_jac = layer.inverse(x)  # x: [B, D], inv_log_det_jac: [B]
                log_prob += inv_log_det_jac  # [B]
            log_prob += self.distribution.log_prob(x)  # [B]
        else:
            log_prob = self.distribution.log_prob(x)
        return log_prob  # [B]

    def rsample(self, num_samples):
        """
        num_samples: int
        Returns:
            x: [num_samples, D]
            log_prob: [num_samples]
        """
        x = self.distribution.sample((num_samples,)).to(self.device)  # [N, D]
        log_prob = self.distribution.log_prob(x)  # [N]
        if self.use_flow:
            for layer in self.layers:
                x, log_det_jac = layer.forward(x)  # x: [N, D], log_det_jac: [N]
                log_prob += log_det_jac
        return x, log_prob  # [N, D], [N]

    def sample_each_step(self, num_samples):
        """
        Returns intermediate samples after each RealNVPNode
        num_samples: int
        Returns:
            samples: list of numpy arrays, each of shape [num_samples, D]
        """
        samples = []
        x = self.distribution.sample((num_samples,)).to(self.device)  # [N, D]
        samples.append(x.detach().cpu().numpy())  # initial sample

        for layer in self.layers:
            x, _ = layer.forward(x)  # x: [N, D]
            samples.append(x.detach().cpu().numpy())  # after each layer

        return samples  # list of [N, D]

    def training_step(self, batch, batch_idx):
        """
        batch: [B, D]
        """
        x = batch
        log_prob = self.log_probability(x)  # [B]
        loss = -log_prob.mean()  # scalar
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        batch: [B, D]
        """
        x = batch
        log_prob = self.log_probability(x)  # [B]
        val_loss = -log_prob.mean()  # scalar
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def normalize(self, X, mn, mx):
        X = (X - mn)/(mx - mn)
        #uq = torch.round(u*255)
        return X

    def unormalize(self, u, mn, mx):
        return u*(mx-mn) + mn
