### Importing Libraries ###
import numpy as np
import torch
import torch.multiprocessing as mp
from joblib import Parallel, delayed
import os
import time
### Importing Manual Modules ###
from model.ProbModels.cgCnnMvn import probabModel
from model.pde.pdeForm2D import pdeForm
from utils.PostProcessing import postProcessing
from utils.tempData import storingData
from input import *
from utils.variousFunctions import calcRSquared, calcEpsilon, makeCGProjection, setupDevice, get_tensor_size, list_gpu_tensors, create_latent_grid, interpolate_latents, patched_getitem, image_to_binary_tensor, autocorrelation, clean_torch_file
from utils.saveEvaluationDataset import saveDatasetAll, importDatasetAll
from model.pde.pdeTrueSolFenics import solve_pde, solve_pde_Helmholz
import warnings
if True:
    warnings.filterwarnings("ignore")
    print('Be careful All the warnings are ignored in this run! Not suggested when debugging!')
import hamiltorch
import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



### Device Selection ###
device = setupDevice(cudaIndex, device, dataType)


### Constructing Post Processing Instance ###
post = postProcessing(path='./results/data/', displayPlots=display_plots)
post.save([['numOfTestSamples', np.array(numOfTestSamples)]])

### Reading Existing Dataset (if it exists) ###
if (not createNewCondField) and not importDatasetOption:
    sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = post.readTestSample(numOfTestSamples)
    torch.save(sampX, './model/pde/RefSolutions/condFields/' + 'sampX.dat')

### Importing Dataset ###
if importDatasetOption and not createNewCondField:
    if os.path.exists(saveDatasetName):
        print("Dataset:"+saveDatasetName+" exists. Importing...")
        sampCond, sampSolFenics, sampX, sampYCoeff, sampSol, gpEigVals, gpEigVecs = importDatasetAll(datapath=saveDatasetName, device=device)
        torch.save(sampX, './model/pde/RefSolutions/condFields/' + 'sampX.dat')
        torch.save(gpEigVals, './model/pde/RefSolutions/condFields/' + 'gpEigVals.dat')
        torch.save(gpEigVecs, './model/pde/RefSolutions/condFields/' + 'gpEigVecs.dat')
    else:
        raise ValueError("Dataset:"+saveDatasetName+" does not exist. Ignoring the command")
    



### Definition and Form of the PDE ###
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp, createNewCondField, device, post, rhs=rhs, reducedDim=reducedDim,
              options=options)
#pde.uBc = 1. # only for training Helmholz

### Create unlabeled dataset ###
#pde.createDu(N=100000, keep=True)
#pde.createDl(N=20000, keep=True)
pathUnlabeled = './data/unlabeledData.pth'
pathLabeled = './data/labeledData.pth'
#pathLabeled = './data/labeledData10000_65.pth'



### Calculation of the Coarse-Grained Projection ###
ProjectionIsTheTrueSolution = True
if ProjectionIsTheTrueSolution and (not createNewCondField) and compareWithCGProjection:
    xTest, yTest, yProj, yProjTotal = makeCGProjection(pde, sampX, sampSolFenics, sampYCoeff, load=loadProjections)


### Save the Reference Solution Data ###
if importDatasetOption and not createNewCondField:
    pde.saveFields(sampSol, sampSolFenics, sampYCoeff, sampCond, sampX)



### Initialization of Variables ###
tempData = storingData()






### Simple importance Sampling Test ###
RSquaredHistoryAvg = torch.zeros(100)

### External Loop for averaging out the randomless of the fixed 10 residuals ###
for kkk in range(externalIter):
    
    residual_history = []
    sigmaHistory = []
    movAvgResHist = []
    residuals_history = []
    hist_elbo = []
    hist_iter = []
    sigmaEvolution = []
    varNormEvolution = []
    XfullHist = []
    yEvolwMean = []
    yEvolwStd = []

    likelihood = []
    logProb = []
    entropy = []
    totalElbo = []

    RSquaredHistory = torch.zeros(100)
    
    progress_perc = 0
    t = time.time()


    ### Creating instance of the Probabilistic Model ###
    samples = probabModel(pde, poly_pow=poly_pow, stdInit=stdInit, gradLr=gradLr, lr=lr, sigma_r=sigma_r, sigma_w=sigma_w, yFMode=yFMode, randResBatchSize=randResBatchSize, reducedDim=reducedDim, dimz=60)
    samples.neuralNet.train()
    samples.flow.train()
    samples.xDecoder.train()
    samples.ztoXDecoder.train()



    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./checkpoints"
    torch.cuda.manual_seed_all(42) 
    generator = torch.Generator(device=device)



    #dx = torch.load(pathUnlabeled, map_location=device).unsqueeze(1).flatten(-2)[:200].to(dtype=torch.float32)
    #dxy = torch.load(pathLabeled, map_location=device).unsqueeze(-3).flatten(-2)[:, :10000, :, :].to(dtype=torch.float32)
    #dxy = torch.load(pathLabeled, map_location=device).unsqueeze(-3).flatten(-2)[:, :1000, :, :].repeat(1, 10, 1, 1).to(dtype=torch.float32)
    #dxy90 = torch.load('./data/labeledData20000_129_VF90.pth', map_location=device).unsqueeze(-3).flatten(-2)[:, :7000, :, :].to(dtype=torch.float32)
    #dxy07 = torch.load('./data/labeledData_l2_VF50.pth', map_location=device).unsqueeze(-3).flatten(-2)[:, :10000, :, :].to(dtype=torch.float32)
    #dxy025 = torch.load('./data/labeledData20000_129.pth', map_location=device).unsqueeze(-3).flatten(-2)[:, :20000, :, :].to(dtype=torch.float32)
    dxy025 = torch.load('./data/labeledData20000_129.pth', map_location=device).unsqueeze(-3).flatten(-2)[:, :200, :, :].to(dtype=torch.float32)
    #dxy025 = torch.load('./data/labeledData20000Helm_129.pth', map_location=device).unsqueeze(-3).flatten(-2)[:, :10000, :, :].to(dtype=torch.float32)
    dxyvo = torch.load('./data/labeledData20000_129.pth', map_location=device).unsqueeze(-3).flatten(-2)[:, :20000, :, :].to(dtype=torch.float32)

    #dxy = torch.cat((dxy025, dxy07), dim=1)
    dxy = dxy025
    #idx = torch.randperm(dxy.size(1))
    #dxy = dxy[:, idx, ...]
    ymax = dxy[1].max()
    ymin = dxy[1].min()
    samples.ymin = ymin
    samples.ymax = ymax
    samples.ystd = dxy[1].std()
    samples.ymean = dxy[1].mean()
    #dxy[1] = samples.normy(dxy[1])
    #dxy = torch.load(pathLabeled, map_location=device).unsqueeze(-3).flatten(-2)[:, :1000, :, :].repeat(1, 10, 1, 1).to(dtype=torch.float32)
    dxy_un = dxyvo[0, :1000]
    ### Augmenting labels of extra features (cheap to compute)
    u = torch.sum(torch.where(dxy[0]>0.9, 1., 0.).flatten(-2)/dxy[0].flatten(-2).size(-1), dim=-1)
    u = - torch.log(1/u -1)
    dxyu = torch.cat((dxy[0], u.reshape(-1, 1, 1)), dim=-1).to(dtype=torch.float32)
    
    u = torch.sum(torch.where(dxyvo[0]>0.9, 1., 0.).flatten(-2)/dxyvo[0].flatten(-2).size(-1), dim=-1)
    u = - torch.log(1/u -1)
    dxyu_vo = torch.cat((dxyvo[0], u.reshape(-1, 1, 1)), dim=-1).to(dtype=torch.float32)


    # Instantiate the dataset
    #dataset = torch.utils.data.TensorDataset(dx, labels) #Unlabeled
    dataset = torch.utils.data.TensorDataset(dxyu, dxy[1]) #Labeled
    # Split the dataset into training, validation, and test sets
    train_size = int(0.5 * len(dataset))  # 70% for training
    val_size = int(0.3 * len(dataset))  # 15% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining for test

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
    #torch.utils.data.TensorDataset.__getitem__ = patched_getitem

    # Create DataLoader for each set
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=250, shuffle=True, drop_last=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=False, drop_last=True, generator=generator)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False, drop_last=True, generator=generator)

    unlabeled_dxy = dxy_un  # Assuming [:200] was used for labeled
    unlabeled_dataset = torch.utils.data.TensorDataset(unlabeled_dxy, torch.zeros(unlabeled_dxy.size(0)))

    ### VO dataset ###
    virtualObservables = torch.zeros(dxyvo.size(1)) 
    vodataset = torch.utils.data.TensorDataset(dxyu_vo, virtualObservables) # Virtual Observables
    # Split the dataset into training, validation, and test sets
    votrain_size = int(0.5 * len(vodataset))  # 70% for training
    voval_size = int(0.3 * len(vodataset))  # 15% for validation
    votest_size = len(vodataset) - votrain_size - voval_size  # Remaining for test

    votrain_set, voval_set, votest_set = torch.utils.data.random_split(vodataset, [votrain_size, voval_size, votest_size], generator=generator)
    ### VO dataset ###

    


    def train_model(model, generator, train_set, val_loader=None, utrain_set=None, model_name="genPANIS", batchsize=100):
        # Create a PyTorch Lightning trainer
        """
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                            devices=1,
                            max_epochs=400,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                        LearningRateMonitor("epoch")],
                            check_val_every_n_epoch=399)
        """
        
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                            devices=1,
                            max_epochs=1000,
                            callbacks=[ModelCheckpoint(every_n_epochs=999),
                                        LearningRateMonitor("epoch")])
                            #,check_val_every_n_epoch=399)
        
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True, drop_last=False, generator=generator)
        unlabeled_loader = torch.utils.data.DataLoader(utrain_set, batch_size=batchsize*25, shuffle=True, generator=generator)
        vo_loader = torch.utils.data.DataLoader(votrain_set, batch_size=batchsize*10, shuffle=True, generator=generator)
        model.unlabeled_loader = unlabeled_loader
        model.unlabeled_iter = iter(unlabeled_loader)
        model.vo_loader = vo_loader
        model.vo_iter = iter(vo_loader)
        result = None

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + ".ckpt")
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_26/checkpoints/epoch=18-step=380.ckpt'
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_112_dim60_VF50_VF10_VF90_iter500_flow12_latentz/checkpoints/epoch=498-step=19960.ckpt'

        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_142_final_data10000_iter10000_flow12/checkpoints/epoch=998-step=39960.ckpt'
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_144_final_l100_u0_iter500/checkpoints/epoch=498-step=4990.ckpt'
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_146_final_l100_u10000_iter500/checkpoints/epoch=498-step=4990.ckpt'
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_148_final_l100_u1000_iter500/checkpoints/epoch=498-step=4990.ckpt'
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_160/checkpoints/epoch=98-step=1980.ckpt'
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_162_final_l1000/checkpoints/epoch=998-step=39960.ckpt'
        #pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_162_final_l1000/checkpoints/epoch=998-step=39960.ckpt'
        pretrained_filename = './checkpoints/genPANIS/lightning_logs/version_142_final_data10000_iter10000_flow12/checkpoints/epoch=998-step=39960.ckpt'
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            ckpt = torch.load(pretrained_filename, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            #trainer.fit(model, train_data_loader, val_loader)
            #result = ckpt.get("result", None)
            #print("Continue Training", model_name)
            #trainer.fit(flow, train_data_loader, val_loader)
        else:
            print("Start training", model_name)
            #trainer.fit(model, train_data_loader, val_loader)
            trainer.fit(model, train_data_loader)



        return model, result
    
    batchsize=250 #250 normally
    train_model(samples, generator=generator, train_set=train_set, val_loader=val_loader, utrain_set=unlabeled_dataset, batchsize=batchsize)
    samples.neuralNet.eval()
    samples.flow.eval()
    samples.xDecoder.eval()
    samples.ztoXDecoder.eval()
    samples.to(device)
    
    #### Generating Samples and Predictions ####
    generated_z = []
    generated_x = []
    generated_X = []
    generated_y_mean = []
    generated_y_std = []
    generated_u = []
    
    #D = 20
    D = batchsize
    indicesRec = val_set.indices
    reconDatax = dxyu[indicesRec]
    reconDatay = dxy[1][indicesRec]
    for i in range(20):
        #batch_z, batch_x, batch_X, y = samples.Reconstruction(batch=[reconDatax[(D*i):(D*i+D)], reconDatay[(D*i):(D*i+D)]], true_x=False)
        batch_z, batch_x, batch_X, y = samples.sample(num_samples=D)
        generated_z.append(batch_z)
        generated_x.append(batch_x)
        generated_X.append(batch_X)
        generated_y_mean.append(y[0])
        generated_y_std.append(y[1])
        #generated_u.append(1/(1.+torch.exp(-(samples.flow.polyCoeff[0] + samples.flow.polyCoeff[1]*batch_z[...,0, 0]+ samples.flow.polyCoeff[2]*batch_z[...,0, 0]**2))))
        generated_u.append(1/(1.+torch.exp(-samples.mu_tu(batch_z[...,0, 0]))))
    
    generated_z = torch.cat(generated_z, dim=0)
    generated_x = torch.cat(generated_x, dim=0)
    generated_X = torch.cat(generated_X, dim=0)
    generated_y_mean = torch.cat(generated_y_mean, dim=0)
    generated_y_std = torch.cat(generated_y_std, dim=0)
    generated_u = torch.cat(generated_u, dim=0)

    
    z_interp = create_latent_grid(dimz=generated_z.size(-1), blueprint_z=generated_z[0])
    z_interp = torch.stack((interpolate_latents(generated_z[0].squeeze(0), generated_z[1].squeeze(0)),
                            interpolate_latents(generated_z[1].squeeze(0), generated_z[2].squeeze(0)),
                            interpolate_latents(generated_z[2].squeeze(0), generated_z[3].squeeze(0)),
                             interpolate_latents(generated_z[3].squeeze(0), generated_z[4].squeeze(0)),
                              interpolate_latents(generated_z[4].squeeze(0), generated_z[5].squeeze(0))), dim=0)
    z_interp, x_interp, X_interp, y_interp = samples.sample(num_samples=100, z=z_interp)
    y_interp = y_interp[0]
    genz1_50 = torch.cat((generated_z[0, 0, :10], generated_z[4, 0, 10:]), dim=0)
    #genz1_50 = generated_z[4].squeeze(0)
    zInt = interpolate_latents(generated_z[0].squeeze(0), genz1_50)
    zInt, xInt, XInt, yInt = samples.sample(num_samples=100, z=zInt.squeeze(1))
    zInt = {'z':zInt.squeeze(1), 'x':xInt.squeeze(1), 'y': yInt[0].squeeze(1), 'X':XInt.squeeze(1)}
    torch.save(zInt, './zInt.pt')
    ### Calculate Reference Solution on generated sample
    if False:
        writingList = [['intGrid', pde.sgrid],
                    ['rbfGrid', pde.grid],
                    ['rbfGridW', pde.gridW],
                    ['gpEigVals', pde.gpEigVals],
                    ['gpEigVecs', pde.gpEigVecs]]
        post.save(writingList)
        #c_x = torch.nn.functional.interpolate(generated_x, size=(129, 129), mode='bilinear', align_corners=True)
        sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = pde.produceTestSample(Nx=numOfTestSamples, post=post, cx=generated_x)
        #sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = pde.produceTestSample(Nx=numOfTestSamples, post=post, cx=1/generated_x**1.30103)
        torch.save(sampSolFenics, './data/' + 'refSolution.pth')
        yRef = sampSolFenics.unsqueeze(1)
    else:
        yRef = torch.load('./data/' + 'refSolution.pth').unsqueeze(1)


    #DuX = torch.exp(samples.neuralNet.xtoXCnn(dx, KLE=False))
    #DuX = torch.exp(samples.neuralNet.xtoXCnn(dxy[0, :100].reshape(-1, 1, pde.sgrid.size(-1), pde.sgrid.size(-1)).to(dtype=torch.float64), KLE=False, mn=samples.Xmin, mx=samples.Xmax))
    DuX = torch.exp(samples.neuralNet.xtoXCnn(dxy[0, :100].reshape(-1, 1, pde.sgrid.size(-1), pde.sgrid.size(-1)), KLE=False, mn=samples.Xmin, mx=samples.Xmax))

    
    ### To be removed ###
    if False:
        generated_z2 = []

        
        #D = 20
        D = 1000
        indicesRec = val_set.indices
        reconDatax = dxyu[indicesRec]
        reconDatay = dxy[1][indicesRec]
        for i in range(1000):
            batch_z, batch_x, batch_X, y = samples.sample(num_samples=D)
            generated_z2.append(batch_z)

        generated_z2 = torch.cat(generated_z2, dim=0)
        generated_z = torch.cat((generated_z, generated_z2), dim=0)
    ### To be removed ###
    
    
print("Training Finished.")


### generating samples by using the flow model
#generated_x = samples.xDecoder(generated_z).reshape(-1, 1, pde.sgrid.size(-1), pde.sgrid.size(-1))
#generated_x = torch.bernoulli(generated_x)
#generated_x[generated_x == 0] = 0.1


### Saving samples from distribution ###
#reconSamples = {'z': generated_z, 'x': generated_x, 'y': generated_y_mean}
#torch.save(reconSamples, './experiments/latentz/recongenSamplesVF10.pth')
### Saving samples from distribution ###

W = samples.xDecoder.fc1.weight.data
W = W.t().reshape(-1, pde.sgrid.size(-1), pde.sgrid.size(-1))
b = samples.xDecoder.fc1.bias.data.reshape(pde.sgrid.size(-1), pde.sgrid.size(-1))

### Drawing Samples by using the Approx. Posterior (given x) ###
#samples.generateFromApproxPosterior(batch=[dxyu[:100], dxy[1][:100]])
### Drawing Samples by using the Approx. Posterior (given x) ###



### Solving the Forward, Inverse, Condition_u Problem ###
#z, resForward = samples.ForwardProblemMAP(y_true=sampSolFenics[1].unsqueeze(0).unsqueeze(0), x_obs=sampCond[1], iterMax=200, lr=1e-3)
#z, resInverse = samples.InverseProblemMAP(y_obs=sampSolFenics[1].unsqueeze(0).unsqueeze(0), x_true=sampCond[1], iterMax=200, lr=1e-3)
#z = samples.InverseProblemHMC(y_obs=sampSolFenics[1].unsqueeze(0).unsqueeze(0), x_true=sampCond[1], num_samples=500)
#z = samples.ForwardProblemHMC(y_true=generated_y_mean[0].unsqueeze(0), x_obs=generated_x[0].squeeze(0), num_samples=6000, XTrue=generated_X[0].squeeze(0), burn=2000)

## u=0.4055 --> vf=0.6


#dataVF50Ref = torch.load('./experiments/diffSNR_compPINO/darcy/inverseProblemPair.pt')
#dataVF50Ref = dxy[:, 0].reshape(2, 65, 65)
#z = torch.load('./testInv.pt')
#torch.save(z, './results/data/resInverse.pt')
#z = z['z_samples']
#z = samples.InverseProblemHMC(y_obs=dataVF50Ref[1].unsqueeze(0).unsqueeze(0), x_true=dataVF50Ref[0], num_samples=150, burn=40, SNR_in_dB=10)
#z = samples.ForwardProblemHMC(y_true=dataVF50Ref[1].unsqueeze(0).unsqueeze(0), x_obs=dataVF50Ref[0], num_samples=150, burn=20, pde='helmholz')
#z = samples.InverseProblemHMC(y_obs=dataVF10[1, 0].unsqueeze(0).unsqueeze(0), x_true=dataVF10[0, 0], num_samples=6000, burn=1000)
#z = samples.InverseProblemHMC(y_obs=data_tum[1].unsqueeze(0).unsqueeze(0), x_true=data_tum[0], num_samples=130, burn=20, SNR_in_dB=20)
#z = samples.InverseProblemHMC(y_obs=sampSolFenics[2].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32), x_true=sampCond[2].to(dtype=torch.float32), num_samples=200, XTrue=generated_X[2].squeeze(0), burn=100)
#z = samples.ForwardProblemHMC(y_true=sampSolFenics[2].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32), x_obs=sampCond[2].to(dtype=torch.float32), num_samples=120, XTrue=generated_X[2].squeeze(0), burn=10)

SESSION_DIR = './experiments/demo'
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(SESSION_DIR + '/figs', exist_ok=True)

t = 't'
### Paste your experiment pipeline here ###
data = dxy[:, -1].reshape(-1, 129, 129)
data = torch.load('./experiments/diffSNR_compPINO/darcy/testPair.pt', map_location=device)
z = samples.ForwardProblemHMC(y_true=data[1].unsqueeze(0).unsqueeze(0), x_obs=data[0], num_samples=200, burn=100, L=3, step_size=0.001, SNR_in_dB=40, path=SESSION_DIR + '/f.pt')
z = samples.InverseProblemHMC(y_obs=data[1].unsqueeze(0).unsqueeze(0), x_true=data[0], num_samples=200, burn=100, SNR_in_dB=10, L=3, step_size=0.001, path=SESSION_DIR + '/Mi.pt')
post.fpath = SESSION_DIR + '/figs/'
post.plotForwardProblemResults(data_path=SESSION_DIR + '/f.pt')
post.plotInverseProblemResults(data_path=SESSION_DIR + '/Mi.pt')
### Paste your experiment pipeline here ###




### Saving Data for Post Processing ###
if createNewCondField:
    writingList = [['intGrid', pde.sgrid],
                   ['rbfGrid', pde.grid],
                   ['rbfGridW', pde.gridW],
                   ['gpEigVals', pde.gpEigVals],
                   ['gpEigVecs', pde.gpEigVecs]]
    post.save(writingList)
    sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = pde.produceTestSample(Nx=numOfTestSamples, post=post)
    x_testing_samples = sampX



if saveDatasetOption:
    saveDatasetAll(sampCond, sampSolFenics, sampX, sampYCoeff, sampSol, pde.gpEigVals, pde.gpEigVecs, createNewCondField, path=saveDatasetName)




writingList = [ ['intGrid',  pde.sgrid],
                ['rbfGrid',  pde.grid],
                ['gpEigVals', pde.gpEigVals],
                ['gpEigVecs', pde.gpEigVecs],
                #['gpEigValsDetailed', pde.gpEigValsDetailed],
                #['gpEigVecsDetailed', pde.gpEigVecsDetailed],
                ['generated_x', generated_x.clone().detach().cpu()],
                ['generated_z', generated_z.clone().detach().cpu()],
                ['generated_u', generated_u.clone().detach().cpu()],
                ['generated_X', generated_X.clone().detach().cpu()],
                #['Duz', gksig.clone().detach().cpu()],
                ['generated_y_mean', generated_y_mean.clone().detach().cpu()],
                ['generated_y_std', generated_y_std.clone().detach().cpu()],
                ['DuX', generated_X.clone().detach().cpu()],
                ['Dux', dxy[0][indicesRec].reshape(-1, 1, pde.sgrid.size(-1), pde.sgrid.size(-1)).clone().detach().cpu()],
                ['Duy', dxy[1][indicesRec].reshape(-1, 1, pde.sgrid.size(-1), pde.sgrid.size(-1)).clone().detach().cpu()],
                ['W', W.clone().detach().cpu()],
                ['b', b.clone().detach().cpu()],
                ['yFENICSTrue', yRef.clone().detach().cpu()],
                #['resForward', resForward.clone().detach().cpu()],
                #['resInverse', resInverse.clone().detach().cpu()],
                #['z_samples', z.clone().detach().cpu()],
                ['z_interp', z_interp.clone().detach().cpu()],
                ['x_interp', x_interp.clone().detach().cpu()],
                ['y_interp', y_interp.clone().detach().cpu()],
                ['outOfDistributionPrediction', outOfDistributionPrediction]
                #['analSolmean', analSolmean.clone().detach().cpu()],
                #['analSolstd', analSolstd.clone().detach().cpu()]
                ]

post.save(writingList)
if compareWithCGProjection:
    post.save([['yTest', yProj.detach().cpu()]])
else:
    post.save([['yTest', sampSolFenics.detach().cpu()]])


print("\n", "Post Processing Begins: \n")

post.producePlots()

