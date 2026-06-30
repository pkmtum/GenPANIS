import numpy as np
import torch
from utils.nnArchitectures.nnCoarseGrained import nnPANIS, nnmPANIS, Autoencoder, FullyConnectedNN, nnPANISSigma
from utils.nnArchitectures.XtoxCNN import xDecoder
from utils.nnArchitectures.realNVP import RealNVP
from utils.nnArchitectures.ztoXDecoder import ztoXDecoder
from model.pde.numIntegrators.trapezoid import trapzInt2DParallel
import pytorch_lightning as pl
import hamiltorch
from utils.variousFunctions import calcUncertaintyMetric


### Training by using Fu loss only

class probabModel(pl.LightningModule):
    def __init__(self, pde, poly_pow=None, stdInit=2, display_plots=False, gradLr=0., lr=0.001, sigma_r=0, sigma_w=0, yFMode=True, randResBatchSize=None, reducedDim=None, dataset=None, dimz=100, useL=True, useVO=False, useUnlabeled=False):  # It was stdInit=8
        super().__init__()
        self.pde = pde
        self.gradLr=gradLr
        self.sigma_r = sigma_r
        self.sigma_w = sigma_w
        self.yFMode = yFMode
        self.nele = pde.nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.Nx_samp_phi = self.Nx_samp
        self.x = torch.zeros(self.Nx_samp_phi, 1)
        self.y = torch.zeros(self.Nx_samp_phi, self.nele)
        self.Nx_counter = 0
        self.poly_pow = poly_pow
        self.guideExecCounter = 0
        self.modelExecCounter = 0
        self.dataDriven = True
        self.dataDrivenFlag = False

        self.lowUnif = -1.
        self.highUnif = 1.
        self.readData = 0
        self.xPoolSize = 1
        if self.readData == 1:
            print("Pool of input data x was read from file.")
        else:
            print("Pool of input data x generated for this run.")
            self.data_x = torch.normal(self.mean_px, self.sigma_px, size=(self.xPoolSize,))
            self.data_x = 4 * self.sigma_px * torch.rand(size=(self.xPoolSize,)) - 2 * self.sigma_px
            print(self.data_x)
            print(torch.exp(-self.data_x))
            self.data_x = torch.linspace(-1., 1., self.xPoolSize)


        self.constant_phi_max = torch.ones((pde.NofShFuncs, pde.NofShFuncs))
        self.phi_max = torch.rand((pde.NofShFuncs, 1), requires_grad=True) * 0.01 + torch.ones(pde.NofShFuncs, 1)
        self.phi_max = self.phi_max / torch.linalg.norm(self.phi_max)

        phi_max_leaf = self.phi_max.clone().detach().requires_grad_(True)
        self.phi_max = phi_max_leaf
        self.phiBase = torch.reshape(self.phi_max, [1, -1])

        self.phi_max_history = np.zeros((pde.NofShFuncs, 1))
        self.sigma_history = np.zeros((pde.NofShFuncs, 1))
        self.temp_res = []
        self.full_temp_res = []
        self.model_time = 0
        self.guide_time = 0
        self.sample_time = 0
        self.stdInit = stdInit
        self.randResBatchSize = randResBatchSize
        self.compToKeepInPCA = 4
        self.residualCorrector = torch.sqrt(torch.tensor(self.Nx_samp))
        self.validationIndex = 0
        self.reducedDim = reducedDim
        if self.pde.pde_type == 'helmholz':
            self.Xmax = torch.tensor(0.)
            self.Xmin = torch.tensor(-2.31)
        else:
            self.Xmax = torch.tensor(0.)
            self.Xmin = torch.tensor(-3.)
        self.zmax = torch.tensor(1.)
        self.zmin = torch.tensor(0.)
        self.dimz = dimz
        self.useL = useL
        self.useVO = useVO
        self.useUnlabeled = useUnlabeled

        self.I = torch.ones(self.pde.NofShFuncs)*(stdInit)
        self.I.requires_grad_(True)
        self.vDim = 10
        if yFMode:
            self.V = torch.ones((self.reducedDim**2, self.vDim))*(-3) +torch.randn(self.reducedDim**2, self.vDim)/10
        else:
            self.V = torch.ones(((self.reducedDim-2)**2, self.vDim))*(-3) +torch.randn((self.reducedDim-2)**2, self.vDim)/10
        self.V.requires_grad_(True)

        self.Xfull = torch.randn(10, 1, self.reducedDim**2)

        self.globalSigma = torch.tensor([float(stdInit)])
        self.globalSigma.requires_grad_(True)
        self.sigmaEncoder = torch.tensor(-2.)
        self.sigmaEncoder.requires_grad_(True)
        self.diagSigma = torch.ones(self.pde.sgrid.size(-1)**2)*(stdInit) + torch.randn(self.pde.sgrid.size(-1)**2)/10
        self.diagSigma.requires_grad_(True)
        if yFMode:
            self.NTraining = 99
            self.yF = (torch.ones(self.pde.shapeFunc.aInvBCO.size(-1)) + 0.1 * torch.rand(self.pde.shapeFunc.aInvBCO.size(-1))).repeat(self.NTraining+1, 1).requires_grad_(True)
            self.yWhole = torch.ones(2, 1024) + 0.1 * torch.randn(2, 1024)
            self.yWhole = self.yWhole.requires_grad_(True)


        self.defineSubmodels()

        self.automatic_optimization = False

        numOfPars = self.neuralNet.count_parameters()
        print("Number of NN Parameters Used (DeepONet): ", numOfPars)

    def defineSubmodels(self):
            self.neuralNet = nnPANIS(reducedDim=self.reducedDim, cgCnn=None, xtoXCnn=None, pde=self.pde, extraParams=[self.V, self.globalSigma, self.yFMode, self.Xfull, self.diagSigma, self.sigmaEncoder], dimz=self.dimz)
            #self.neuralNet.load_state_dict(torch.load('./utils/trainedNNs/trainedCGdim1024CR10FR50D0_PANIS.pth'))
            self.xDecoder = xDecoder(reducedDim=self.reducedDim, cgCnn=None, xtoXCnn=None, pde=self.pde, extraParams=[self.V, self.globalSigma, self.yFMode, self.Xfull, self.diagSigma, self.sigmaEncoder], dimz=self.dimz)
            self.ztoXDecoder = ztoXDecoder(dimz=self.dimz)
            num_layers = 12  # or 6, 12, etc.
            masks = []

            base_masks = [
                torch.tensor([i % 2 for i in range(self.dimz)]),
                torch.tensor([(i + 1) % 2 for i in range(self.dimz)])
            ]

            for i in range(num_layers):
                masks.append(base_masks[i % 2])
            self.flow = RealNVP(masks, hidden_size=128, use_flow=True)
            return

    def configure_optimizers(self):
        self.allparams = list(self.flow.parameters()) + list(self.xDecoder.parameters()) + list(self.ztoXDecoder.parameters()) \
        + [p for name, p in self.neuralNet.named_parameters() if name != ['nothing']]
        optimizer = torch.optim.Adam([
                    {'params': self.allparams}
                ], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.99)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        x = batch[0][..., :-1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
        u = batch[0][..., -1]
        y = batch[1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
        gksig = self.neuralNet(x, give_X=True, KLE=False,  mn=self.zmin, mx=self.zmax)[0]
        gksig = gksig.unsqueeze(1)
        z = gksig
        z = self.flow.normalize(gksig.squeeze(1).squeeze(1), mn=self.zmin, mx=self.zmax)
        prior = - torch.sum(self.flow.log_probability(z))
        X = self.ztoXDecoder.forward(gksig.squeeze(1).squeeze(1), mn=self.Xmin, mx=self.Xmax)
        h = self.neuralNet.Xtoy(X, rbfCoeff=False)
        likelihood_y = - self.loglikelihood_y(y, h)
        likelihood = - self.xDecoder._get_likelihood(x, gksig)

        loss = (prior + likelihood + likelihood_y)
        self.log('val_bpd', loss)


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x = batch[0][..., :-1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
        u = batch[0][..., -1]
        y = batch[1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))

        useVO = self.useVO
        useUnlabeled = self.useUnlabeled
        useL = self.useL

        loss = 1.
        beta = 1.0

        if useL:
            gksig = self.neuralNet(x, give_X=True, KLE=False,  mn=self.zmin, mx=self.zmax)[0]
            gksig = gksig.unsqueeze(1)
            z = gksig
            z = self.flow.normalize(gksig.squeeze(1).squeeze(1), mn=self.zmin, mx=self.zmax)
            prior = - torch.sum(self.flow.log_probability(z))
            X = self.ztoXDecoder.forward(gksig.squeeze(1).squeeze(1)[..., :60], mn=self.Xmin, mx=self.Xmax)
            h = self.neuralNet.Xtoy(X)
            likelihood_y = - self.loglikelihood_y(y, h)
            likelihood = - self.xDecoder._get_likelihood(x, gksig)
            ### probabilistic encoder ###
            entropy = (z.size(-1)/2*(torch.log(2*torch.pi*torch.pow(10, self.neuralNet.sigmaEncoder))+1.))*z.size(0)
            ### probabilistic encoder ###
            loss = loss + (likelihood + beta*(prior - entropy) + likelihood_y)

        ### Utilization of unlabeled data ###

        if useUnlabeled:
            try:
                x_unlabeled = next(self.unlabeled_iter)[0]
                xu = x_unlabeled.reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
            except StopIteration:
                self.unlabeled_iter = iter(self.unlabeled_loader)
                x_unlabeled = next(self.unlabeled_iter)[0]
                xu = x_unlabeled.reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
            gksig_un = self.neuralNet(xu, give_X=True, KLE=False,  mn=self.zmin, mx=self.zmax)[0]
            gksig_un = gksig_un.unsqueeze(1)
            zu = self.flow.normalize(gksig_un.squeeze(1).squeeze(1), mn=self.zmin, mx=self.zmax)
            prior_xu = - torch.sum(self.flow.log_probability(zu))
            likelihood_xu = - self.xDecoder._get_likelihood(xu, gksig_un)
            ### probabilistic encoder ###
            entropyu = (zu.size(-1)/2*(torch.log(2*torch.pi*torch.pow(10, self.neuralNet.sigmaEncoder))+1.))*zu.size(0)
            ### probabilistic encoder ###
            loss =  loss  + (likelihood_xu+prior_xu) - entropyu ## Labeled + Unlabeled Data
        ### Utilization of unlabeled data ###

        ### Utilization of virtual observables ###

        if useVO:
            try:
                x_vo = next(self.vo_iter)[0]
                xvo = x_vo[..., :-1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
            except StopIteration:
                self.vo_iter = iter(self.vo_loader)
                x_vo = next(self.vo_iter)[0]
                xvo = x_vo[..., :-1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
            gksig_vo = self.neuralNet(xvo, give_X=True, KLE=False,  mn=self.zmin, mx=self.zmax)[0]
            gksig_vo = gksig_vo.unsqueeze(1)
            zvo = self.flow.normalize(gksig_vo.squeeze(1).squeeze(1), mn=self.zmin, mx=self.zmax)
            prior_xvo = - torch.sum(self.flow.log_probability(zvo))
            likelihood_xvo = - self.xDecoder._get_likelihood(xvo, gksig_vo)
            Xvo = self.ztoXDecoder.forward(gksig_vo.squeeze(1).squeeze(1), mn=self.Xmin, mx=self.Xmax)
            hvo = self.neuralNet.Xtoy(Xvo, rbfCoeff=True)
            likelihood_r = - self.loglikelihood_r(xvo, hvo)
            ### probabilistic encoder ###
            entropyvo = (zvo.size(-1)/2*(torch.log(2*torch.pi*torch.pow(10, self.neuralNet.sigmaEncoder))+1.))*zvo.size(0)
            ### probabilistic encoder ###
            loss = loss + (prior_xvo+likelihood_xvo+likelihood_r) - entropyvo
        ### Utilization of virtual observables ###

        if useL:
            with torch.no_grad():
            ### Rate-x ###
                rate = (- entropy + prior)/x.size(0)
            ### Rate-x###
            ### Distortion-x ###
                distortion = likelihood/x.size(0)
            ### Distrotion-x ###
            ### Distortion-x ###
                distortiony = likelihood_y/y.size(0)
            ### Distrotion-x ###

        self.manual_backward(loss)
        opt.step()
        sch = self.lr_schedulers()
        sch.step()
        self.log('train_bpd', loss)
        if useL:
            self.log('rate', rate)
            self.log('distortion-x', distortion)
            self.log('distortion-y', distortiony)
            self.log('sigmaEncoder', torch.sqrt(torch.pow(10, self.neuralNet.sigmaEncoder)).item())
        self.log('diagSigmaMean', torch.sqrt(torch.pow(10, self.neuralNet.diagSigma)).mean())

    @torch.no_grad()
    def sample(self, num_samples, z=None):
        """
        Sample a batch of images from the flow.
        """
        if z is None:
            z = self.flow.rsample(num_samples=num_samples)[0].unsqueeze(1)
            z = self.flow.unormalize(z, mn=self.zmin, mx=self.zmax)
        else:
            z = z.unsqueeze(1)
        x = self.xDecoder(z)
        x = torch.bernoulli(x)
        x[x == 0] = 0.1
        x = x.reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
        X = self.ztoXDecoder(z[..., :60], mn=self.Xmin, mx=self.Xmax)
        X = X.reshape(-1, 1, 17, 17)
        y = self.gaussiany(X)
        return z, x, X, y

    @torch.no_grad()
    def sample_zOnly(self, num_samples, z=None):
        """
        Sample a batch of images from the flow.
        """
        if z is None:
            z = self.flow.rsample(num_samples=num_samples)[0].unsqueeze(1)
            z = self.flow.unormalize(z, mn=self.zmin, mx=self.zmax)
        else:
            z = z.unsqueeze(1)
        return z

    def normy(self, y):
        return (y - self.ymean)/(self.ystd*2)

    def denormy(self, y):
        return y*self.ystd*2 + self.ymean

    def deltay(self, X):
        y = self.neuralNet.Xtoy(X)
        return y

    def gaussianyNew(self, X, samples=100):#Cov(X)
        ### y are the samples from the posterior with dimension (Number of instances, Number of Perturbations, gridDimx, gridDimy)
        mean = self.neuralNet.Xtoy(X)
        self.neuralNet.diagSigma = self.neuralNet.XtodiagSigma(X)
        y = mean.flatten(-2) + (torch.sqrt(torch.pow(10, self.neuralNet.diagSigma)).unsqueeze(1) * torch.randn(samples, self.neuralNet.diagSigma.size(-1)).unsqueeze(0))
        std = torch.std(y, dim=1)
        return [mean, std.reshape(-1, 1, self.pde.sgrid.size(-2), self.pde.sgrid.size(-1)), y.reshape(y.size(0), y.size(1), self.pde.sgrid.size(-2), self.pde.sgrid.size(-1))]

    def gaussiany(self, X, samples=100, yFMode=False):
        ### y are the samples from the posterior with dimension (Number of instances, Number of Perturbations, gridDimx, gridDimy)
        if not yFMode:
            mean = self.neuralNet.Xtoy(X)
        else:
            mean = self.pde.shapeFunc.cTrialSolutionParallel(self.neuralNet.Xtoy(X, rbfCoeff=True)[:100] \
                                                      + torch.einsum('...ij,...j->...i', self.pde.shapeFunc.aInvBCO, 1 * self.neuralNet.yF).unsqueeze(1))
        y = mean.flatten(-2) + (torch.sqrt(torch.pow(10, self.neuralNet.diagSigma)) * torch.randn(samples, self.neuralNet.diagSigma.size(-1))).unsqueeze(0)
        std = torch.std(y, dim=1)
        return [mean, std.reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))]

    def entropyMvnShortAndStableDiagonal(self):
        manualEntropyStable = 0.5 * torch.sum(torch.log(torch.pow(10, self.neuralNet.diagSigma)))
        return manualEntropyStable

    def loglikelihood_y_withoutConstant(self, data, ypred, X=None):
        return torch.sum(- self.entropyMvnShortAndStableDiagonal() \
        - 0.5 * torch.einsum('...i,...i->...', torch.einsum('...i,...ij->...j', (data-ypred).flatten(-3), torch.diag(1/torch.pow(10, self.neuralNet.diagSigma))), (data-ypred).flatten(-3)))

    def loglikelihood_y(self, data, ypred, X=None):
        return torch.sum(- data.flatten(-3).size(-1)/2 * (torch.log(2*torch.tensor(torch.pi))) - self.entropyMvnShortAndStableDiagonal() \
        - 0.5 * torch.einsum('...i,...i->...', torch.einsum('...i,...ij->...j', (data-ypred).flatten(-3), torch.diag(1/torch.pow(10, self.neuralNet.diagSigma))), (data-ypred).flatten(-3)))

    def loglikelihood_y_noisy(self, data, ypred, X=None, idx=None):
        # Originally the constant term 0.5 * data.flatten().size(0) * torch.log(torch.tensor(self.sigmaNoise)**2) but and now we removed the **2 from self.sigmaNoise
        # It doesn't matter because this is a constant term and doesn't affect the inference
        # self.sigmaNoise is \sigma_n^2
        if idx is None:
            return - torch.sum((data-ypred)**2)/(2*self.var_n)
        else:
            data = data.flatten()[idx]
            ypred = ypred.flatten()[idx]
            return - torch.sum((data-ypred)**2)/(2*self.var_n)

    def entropyMvnShortAndStableDiagonalNew(self):#Cov(X)
        manualEntropyStable = 0.5 * torch.sum(torch.log(torch.pow(10, self.neuralNet.diagSigma)), dim=-1)
        return manualEntropyStable

    def loglikelihood_yNew(self, data, ypred, X=None):#Cov(X)
        if X is not None:
            self.neuralNet.diagSigma = self.neuralNet.XtodiagSigma(X) ### Only when diagSigma = Cov(X) !!!
        return torch.sum(- self.entropyMvnShortAndStableDiagonal() \
        - 0.5 * torch.einsum('...i,...i->...', torch.einsum('...i,...i->...i', (data-ypred).flatten(-3), 1/torch.pow(10, self.neuralNet.diagSigma)), (data-ypred).flatten(-3)))

    def mu_tu(self, z):
        u = 0.03*z
        return u

    def closedFormUpdateForSigmaDiag(self, data, meanPrediction):
        with torch.no_grad():
            self.neuralNet.diagSigma.copy_(torch.log10((torch.sum((data-meanPrediction)**2, dim=0).flatten()/data.size(0))))
        return

    def loglikelihood_uOld(self, data, z1, sigma=0.01):
        u = self.mu_tu(z1) ## scaled fixed model
        return (-(data-u)**2/2/sigma**2).sum()

    def loglikelihood_u(self, data, z, sigma=0.001):
        u = self.neuralNet.uGivenzNN(z) ## scaled fixed model
        return (-(data-u)**2/2/sigma**2).sum()

    def loglikelihood_r(self, x, y, lamb=1.0,):
        indices1 = torch.randperm((self.pde.shapeFuncsDim+1)**2)[:self.randResBatchSize]
        phi = torch.eye(((self.pde.shapeFuncsDim+1)**2)).unsqueeze(0).expand(x.size(0), -1, -1)[:, indices1, :]
        absres = torch.abs(self.pde.calcSingleResGeneralParallel(x, y, phi))
        likelihood = - ((self.pde.shapeFuncsDim+1)**2)*lamb/2.* torch.mean(torch.sum(absres, dim=0), dim=0)
        self.log('Absolute Residual', torch.mean(absres))
        return likelihood

    def loglikelihood_xGivenz(self, data, z):
        if self.idx is not None:
            x = self.forward(z.flatten().unsqueeze(0).unsqueeze(0)).flatten(-3)
            x = torch.clamp(x, min=10**(-6), max=(1-10**(-6)))
            data = torch.where(data.flatten().unsqueeze(0).unsqueeze(0) > self.pde.phaseHigh, 1., 0.).flatten(-3)
            likelihood = torch.sum(data * torch.log(x) + (1-data)*torch.log(1-x))
        else:
            likelihood = self.xDecoder._get_likelihood(data.flatten().unsqueeze(0).unsqueeze(0), z.flatten().unsqueeze(0).unsqueeze(0))
        return likelihood

    def loglikelihood_yGivenz(self, data, z):
        X = self.ztoXDecoder.forward(z[..., :60], mn=self.Xmin, mx=self.Xmax)
        h = self.neuralNet.Xtoy(X)
        likelihood = self.loglikelihood_y_noisy(data, h, X=X, idx=self.idx)
        return likelihood

    def loglikelihood_rGivenz(self, data, z):
        X = self.ztoXDecoder.forward(z, mn=self.Xmin, mx=self.Xmax)
        h = self.neuralNet.Xtoy(X)
        x = self.xDecoder(z)
        x = torch.bernoulli(x)
        x[x == 0] = 0.1
        x = x.reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
        likelihood = self.loglikelihood_r(x, h)
        return likelihood

    def loglikelihood_z(self, z):
        z = self.flow.normalize(z, mn=self.zmin, mx=self.zmax)
        z = z.unsqueeze(0)### This is required only when using flow models
        likelihood = torch.sum(self.flow.log_probability(z))
        return likelihood

    def logPosteriorForward(self, z):
        data = self.xobserved
        log_likelihood = self.loglikelihood_xGivenz(data, z)
        log_prior = self.loglikelihood_z(z)
        logPosterior = (log_likelihood + log_prior)
        return logPosterior

    def logPosteriorInverse(self, z):
        data = self.yobserved
        log_likelihood = self.loglikelihood_yGivenz(data, z)
        log_prior = self.loglikelihood_z(z)
        logPosterior = (log_likelihood + log_prior) 
        return logPosterior

    def logPosteriorCondition(self, z):
        data = self.uobserved
        log_likelihood = self.loglikelihood_u(data, z)
        log_prior = self.loglikelihood_z(z)
        logPosterior = +(log_likelihood + log_prior) 
        return logPosterior


    def ForwardProblemHMC(self, y_true, x_obs, num_samples=1000, step_size=0.001, L = 3, XTrue=None, burn=100, SNR_in_dB=None, observationGrid=None, path=None, pde=None):
        x_obs = x_obs.to(torch.float32)

        if observationGrid is None:
            self.idx = None
        else:
            self.idx = observationGrid

        self.xobserved = x_obs 
        z = torch.randn(self.dimz)*0.001
        z.requires_grad_(True)

        samples = hamiltorch.sample(log_prob_func=self.logPosteriorForward, params_init=z, num_samples=num_samples,
                        step_size=step_size, num_steps_per_sample=L,desired_accept_rate=0.75,sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn)

        zEncoder = [self.neuralNet(x_obs.unsqueeze(0).unsqueeze(0), give_X=True, KLE=False,  mn=self.zmin, mx=self.zmax)[0].squeeze(0).squeeze(0)]
        z = torch.stack((zEncoder+samples))
        z = z.clone().detach()
        with torch.no_grad():
            xlist = []
            Xlist = []
            ylist = []
            if self.pde.pde_type == 'helmholz':
                x_obs = torch.where(x_obs<0.5, 20., 1.)

            for i in range(z.size(0)//100):
                X = self.ztoXDecoder.forward(z[(100*i):(100*i+100)][..., :60], mn=self.Xmin, mx=self.Xmax)
                y = self.neuralNet.Xtoy(X)
                x = self.xDecoder.forward(z[(100*i):(100*i+100)])
                x = torch.bernoulli(x)
                if self.pde.pde_type == 'helmholz':
                    x[x == 0] = 20.
                else:
                    x[x == 0] = 0.1
                xlist.append(x)
                Xlist.append(X)
                ylist.append(y)

            x = torch.cat(xlist, dim=0)
            X = torch.cat(Xlist, dim=0)
            y = torch.cat(ylist, dim=0)

        ### Generating from optimal z

        if X is None:
            resForward = {'x_obs':x_obs, 'yTrue':y_true.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
                    , 'Xmean':X.mean(0).squeeze(0), 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTrue':X.mean(0).squeeze(0), 'z_samples': z}
        else:
            resForward = {'x_obs':x_obs, 'yTrue':y_true.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
                    , 'Xmean':X.mean(0).squeeze(0), 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTrue':XTrue, 'z_samples': z}
        if path is None:
            path = './results/data/resFprward.pt'
        else:
            path = path
        torch.save(resForward, path)
        torch.save(resForward, './results/data/resForward.pt')

        return z

    def InverseProblemHMC(self, y_obs, x_true, num_samples=1000, step_size=0.001, L = 3, XTrue=None, burn=100, SNR_in_dB=None, observationGrid=None, path=None, pde=None, desiredAcceptRate=0.75, keepLastN=None):
        yTrue = y_obs
        if SNR_in_dB is not None:
            y_obs, self.var_n = self.add_noise_dB(y_obs, SNR_in_dB=SNR_in_dB, return_sigma=True)
        else:
            y_obs, self.var_n = self.add_noise_dB(y_obs, SNR_in_dB=40, return_sigma=True)
        if observationGrid is None:
            self.idx = None
        else:
            self.idx = observationGrid

        self.yobserved = y_obs.to(torch.float32) 
        z = torch.randn(self.dimz)*0.001
        z.requires_grad_(True)

        samples = hamiltorch.sample(log_prob_func=self.logPosteriorInverse, params_init=z, num_samples=num_samples,
                          step_size=step_size, num_steps_per_sample=L,desired_accept_rate=desiredAcceptRate,sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn)

        if keepLastN is None:
            z = torch.stack(samples)
        else:
            z = torch.stack(samples)[-keepLastN:]

        z = z.clone().detach()
        with torch.no_grad():
            xlist = []
            Xlist = []
            ylist = []
            if self.pde.pde_type == 'helmholz':
                x_true = torch.where(x_true<0.5, 20., 1.)

            for i in range(z.size(0)//100):
                X = self.ztoXDecoder.forward(z[(100*i):(100*i+100)][..., :60], mn=self.Xmin, mx=self.Xmax).detach()
                y = self.neuralNet.Xtoy(X).detach()
                x = self.xDecoder.forward(z[(100*i):(100*i+100)]).detach()
                x = torch.bernoulli(x).detach()
                if self.pde.pde_type == 'helmholz':
                    x[x == 0] = 20.
                else:
                    x[x == 0] = 0.1
                xlist.append(x)
                Xlist.append(X)
                ylist.append(y)

            x = torch.cat(xlist, dim=0)
            X = torch.cat(Xlist, dim=0)
            y = torch.cat(ylist, dim=0)

        Xmean = X.mean(0).squeeze(0)
        ### Generating from optimal z

        yFenics = self.pde.solve_reference(x.mean(0).reshape(-1, self.pde.sgrid.size(-1)).clone().detach().cpu(), rhs=self.pde.rhs, uBc=self.pde.uBc, options=self.pde.optionsMod).to(self.device)
        if XTrue is None and self.idx is None:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
                      , 'Xm':Xmean, 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':Xmean, 'z_samples': z, 'yFenics':yFenics}
        elif XTrue is None and self.idx is not None:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
                      , 'Xm':Xmean, 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':Xmean, 'z_samples': z, 'obs_indices': self.idx, 'yFenics':yFenics}
        elif XTrue is not None and self.idx is not None:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
            , 'Xm':X.mean(0).squeeze(0), 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':XTrue, 'z_samples': z, 'obs_indices': self.idx, 'yFenics':yFenics}
        else:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
            , 'Xm':X.mean(0).squeeze(0), 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':XTrue, 'z_samples': z, 'yFenics':yFenics}
        if path is None:
            path = './results/data/resInverse.pt'
        else:
            path = path
        torch.save(resInverse, path)
        torch.save(resInverse, './results/data/resInverse.pt')

        return z

    def InverseProblemMAP(self, y_obs, x_true, XTrue=None, lr=1e-2, SNR_in_dB=None, observationGrid=None, pde=None, iterMax=1000, path=None):
        yTrue = y_obs
        if SNR_in_dB is not None:
            y_obs, self.var_n = self.add_noise_dB(y_obs, SNR_in_dB=SNR_in_dB, return_sigma=True)
        else:
            y_obs, self.var_n = self.add_noise_dB(y_obs, SNR_in_dB=40, return_sigma=True)
        if observationGrid is None:
            self.idx = None
        else:
            self.idx = observationGrid

        self.yobserved = y_obs.to(torch.float32) ## Data for the inverse problem
        z = torch.randn(self.dimz)*0.1
        z.requires_grad_(True)

        optimizer = torch.optim.Adam([{'params': z}], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.9995    # excellent for 10k iterations
        )
        for i in range(iterMax):
            loss = -self.logPosteriorInverse(z)/self.dimz
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print('Iteration:', i, ', Loss:',loss.item())

        z = z.repeat(100, 1)
        with torch.no_grad():
            xlist = []
            Xlist = []
            ylist = []
            if self.pde.pde_type == 'helmholz':
                x_true = torch.where(x_true<0.5, 20., 1.)

            for i in range(z.size(0)//100):
                X = self.ztoXDecoder.forward(z[(100*i):(100*i+100)][..., :60], mn=self.Xmin, mx=self.Xmax).detach()
                y = self.neuralNet.Xtoy(X).detach()
                x = self.xDecoder.forward(z[(100*i):(100*i+100)]).detach()
                x = torch.bernoulli(x).detach()
                if self.pde.pde_type == 'helmholz':
                    x[x == 0] = 20.
                else:
                    x[x == 0] = 0.1
                xlist.append(x)
                Xlist.append(X)
                ylist.append(y)

            x = torch.cat(xlist, dim=0)
            X = torch.cat(Xlist, dim=0)
            y = torch.cat(ylist, dim=0)

        Xmean = X.mean(0).squeeze(0)
        ### Generating from optimal z

        yFenics = self.pde.solve_reference(x.mean(0).reshape(-1, self.pde.sgrid.size(-1)).clone().detach().cpu(), rhs=self.pde.rhs, uBc=self.pde.uBc, options=self.pde.optionsMod).to(self.device)
        if XTrue is None and self.idx is None:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
                      , 'Xm':Xmean, 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':Xmean, 'z_samples': z, 'yFenics':yFenics}
        elif XTrue is None and self.idx is not None:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
                      , 'Xm':Xmean, 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':Xmean, 'z_samples': z, 'obs_indices': self.idx, 'yFenics':yFenics}
        elif XTrue is not None and self.idx is not None:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
            , 'Xm':X.mean(0).squeeze(0), 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':XTrue, 'z_samples': z, 'obs_indices': self.idx, 'yFenics':yFenics}
        else:
            resInverse = {'xTrue':x_true, 'y_obs':y_obs.squeeze(0).squeeze(0), 'yTrue': yTrue.squeeze(0).squeeze(0), 'xmean':x.mean(0).reshape(-1, self.pde.sgrid.size(-1)), 'xstd':x.std(0).reshape(-1, self.pde.sgrid.size(-1))
            , 'Xm':X.mean(0).squeeze(0), 'ymean':y.mean(0).squeeze(0), 'ystd':y.std(0).squeeze(0), 'XTr':XTrue, 'z_samples': z, 'yFenics':yFenics}
        if path is None:
            path = './results/data/resInverse.pt'
        else:
            path = path
        torch.save(resInverse, path)
        torch.save(resInverse, './results/data/resInverse.pt')

        return z

    def ConditioningProblemHMC(self, u_obs, z_init, num_samples=1000, step_size=0.1, L = 5):
        self.uobserved = u_obs.to(torch.float32)
        z_init = torch.ones(self.dimz)*0.
        z_init.requires_grad_(True)

        # Sample using HMC
        samples = hamiltorch.sample(log_prob_func=self.logPosteriorCondition, params_init=z_init, num_samples=num_samples,
                               step_size=step_size, num_steps_per_sample=L,desired_accept_rate=0.75,sampler=hamiltorch.Sampler.HMC_NUTS,burn=400)

        z = torch.stack(samples)

        vf_samples = 1./(1.+torch.exp(-self.neuralNet.uGivenzNN(z)))
        vf_mean_model = vf_samples.mean()
        vf_std_model = vf_samples.std()
        vf_z, vf_x, vf_X, vf_y = self.sample(num_samples=100, z=z)
        vf_pred = torch.sum(torch.where(vf_x>0.9, 1., 0.).flatten(-2)/vf_x.flatten(-2).size(-1), dim=-1)
        vf_mean = vf_pred.mean()
        vf_std = vf_pred.std()
        zmean=z.mean(0)
        zstd=z.std(0)

        resCondition = {'vfmeanPred': vf_mean_model, 'vfstdPred': vf_std_model, 'vfmeanActual': vf_mean, 'vfstdActual': vf_std, 'xGenerated': vf_x, 'u_obs':1./(1.+torch.exp(-u_obs))}
        path = './results/data/resCondition.pt'
        torch.save(resCondition, path)

        return z

    def logPosteriorConditionTesting(self, z):
        mu = torch.arange(z.size(0))
        variances = torch.ones(z.size(0))*20

        # Compute log prob of multivariate normal with diagonal covariance
        log_prob = -0.5 * torch.sum(((z-mu) ** 2) / variances)
        return log_prob

    def hmcTest(self, u_obs, z_init, num_samples=1000, step_size=0.1, L = 5):
        self.uobserved = u_obs.to(torch.float32)
        z_init = torch.randn(21)*0.01
        z_init.requires_grad_(True)

        # Sample using HMC
        samples = hamiltorch.sample(log_prob_func=self.logPosteriorConditionTesting, params_init=z_init, num_samples=num_samples,
                               step_size=step_size, num_steps_per_sample=L,desired_accept_rate=0.75,sampler=hamiltorch.Sampler.HMC_NUTS,burn=200)

        params_hmc = torch.stack(samples)

        return params_hmc

    def Reconstruction(self, batch, true_x=True):
        with torch.no_grad():
            x = batch[0][..., :-1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))
            u = batch[0][..., -1]
            ydata = batch[1].reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))

            gksig = self.neuralNet(x, give_X=True, KLE=False,  mn=self.zmin, mx=self.zmax)[0]
            gksig = gksig.unsqueeze(1)

            z = gksig.squeeze(1).squeeze(1)

            X = self.ztoXDecoder.forward(gksig.squeeze(1).squeeze(1)[..., :60], mn=self.Xmin, mx=self.Xmax)
            print(X)
            h = self.neuralNet.Xtoy(X)
            y = self.gaussiany(X)

            h = y[0]
            std = y[1]
            EnvMetric = calcUncertaintyMetric(h.flatten(-3), ydata.flatten(-3), std.flatten(-3), 2)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
            print("Enveloping Score:", EnvelopScore)

            if not true_x:
                x = self.xDecoder.forward(z)
                x = torch.bernoulli(x)
                x[x == 0] = 0.1
                x = x.reshape(-1, 1, self.pde.sgrid.size(-1), self.pde.sgrid.size(-1))

            return z.unsqueeze(1), x, X, y

    def add_noise_dB(self, u_ref, SNR_in_dB, return_sigma=False):
        # u_ref is torch tensor!
        # change SNR in dB to SNR ## SNR=10 --> 10 db, SNR=100 --> 20 db, SNR=10 --> 10 db, SNR=3 --> 4.7712 db, SNR=2 --> 3.01 db, SNR=1 --> 0. db,
        SNR = 10.0 ** (SNR_in_dB / 10.0)
        u_ref_flat = u_ref.flatten()
        Sq_u_ref_flat = torch.pow(u_ref_flat, 2)
        SqSigma = torch.mean(Sq_u_ref_flat)
        Sigma = torch.sqrt(SqSigma) / SNR
        # generate noise
        noise = torch.distributions.Normal(torch.tensor(0.0, device=u_ref.device), Sigma).sample(u_ref.size())
        # add noise
        u_obs = u_ref + noise
        if return_sigma:
            peak_u = torch.max(u_ref_flat)
            print("The noise applied of {} dB is equivalent to a SNR of {:.2f} which "
                "means sigma = {:.3g}. peak u is {:.3g}".format(SNR_in_dB, SNR, Sigma, peak_u))
            return u_obs, Sigma
        else:
            return u_obs
