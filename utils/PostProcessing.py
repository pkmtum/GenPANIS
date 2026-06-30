### Importing Libraries ###
import numpy as np
import math
import random
import pandas as pd
import sys
from numpy.linalg import inv
import matplotlib.pyplot as plt
import re

### Import Pyro/Torch Libraries ###
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize



from torch import nn
import os
import logging
from torch.distributions import constraints

smoke_test = ('CI' in os.environ)
from torch.distributions import constraints
import time
#import fenics as df
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from textwrap import wrap
from model.pde.pdeForm2D import pdeForm
from model.pde.shapeFunctions.hatFunctions import rbfInterpolation
from utils.variousFunctions import calcRSquared, calcEpsilon, calc_pixel_accuracy
import shutil
from scipy.ndimage import zoom
import torch
from scipy.stats import norm


class postProcessing:
    def __init__(self, path='./results/data/', fpath='./results/figs/', displayPlots=True, cleanFigs=True):
        self.path = path
        self.fpath = fpath
        if not os.path.exists(fpath):
            # If it doesn't exist, create the folder
            os.makedirs(fpath)
        if not os.path.exists(path):
            # If it doesn't exist, create the folder
            os.makedirs(path)
        self.defaultReadingList = [
                ['intGrid', 1],
                ['rbfGrid', 1],
                ['xPred',  1],
                ['solCoeffMean', 1],
                ['solPred',  1],
                ['psiEvolution', 1],
                ['phiEvolution', 1],
                ['residualEvolution', 1],
                ['elboEvolution', 1],
                ['movAvgElbo', 1],
                ['movAvgRes', 1],
                ['elboMinMaxEvolution', 1],
                ['relativeImprovementOfPhi', 1],
                ['grads', 1],
                ['gradsNorm', 1],
                ['phiGradEvolution', 1],
                ['psiGradEvolution', 1],
                ['jointGradEvolution', 1],
                ['sigmaEvolution', 1],
                ['solUpLowPred', 1],
                ['solSamplesMean', 1],
                ['solSamplesStd', 1],
                ['sampSamplesMean', 1],
                ['sampSamplesStd', 1],
                ['gpEigVals', 1],
                ['gpEigVecs', 1],
                ['gpEigValsDetailed', 1],
                ['gpEigVecsDetailed', 1],
                ['sPCA', 1],
                ['vPCA', 1],
                ['uPCA', 1],
                ['princDirDecom', 1],
                ['pcaApprox', 1],
                ['timeArray', 1],
                ['samp0Evol', 1],
                ['fc1BiasGrad', 1],
                ['fc1WeightGrad', 1],
                ['fc3BiasGrad', 1],
                ['fc3WeightGrad', 1],
                ['fullPhiGrad', 1],
                ['fc1Bias', 1],
                ['fc1Weight', 1],
                ['fc3Bias', 1],
                ['fc3Weight', 1],
                ['fullPhi', 1],
                ['numOfTestSamples', 1],
                ['psi64', 1],
                ['elbo_x', 1],
                ['elbo_y', 1],
                ['elbo_residuals', 1],
                ['normOfAbsDiff', 1],
                ['RSquared', 1],
                ['elboW', 1]
            ]
        self.plotCounter = 0
        self.displayPlots = displayPlots
        if cleanFigs:
            for filename in os.listdir(self.fpath):
                if filename.endswith(".png"):
                    file_path = os.path.join(self.fpath, filename)
                    os.remove(file_path)
        self.removeOldData(path)
    
    def removeOldData(self, folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

   
    def useCpu(self):
        torch.set_default_dtype(torch.float64)
        torch.set_default_device('cpu')

    def producePlots(self):
        self.useCpu()
        plt.close()
        self.data = self.read()


        #self.plotMeanSampleAsHMC()
        #self.plotESSHistory()
        #self.plotGreedySolutionEvolution()
        #self.plotresInterHistory()
        #self.plotresPhiHistory()
        #self.plotUsedWeightFuncs()
        #self.plotHLoss()
        
        #self.plotTrueSolSurface()
        #self.plotxXpairs()
        #self.plotResidual()
        self.ploteffOfLab()
        #self.ploteffOfUn()
        #self.diffSNR_compPINO_forward()
        #self.diffSNR_compPINO()
        #self.partialObs_compPINO()

        self.plotUncertaintySection()
        self.plotForwardProblemResults()
        self.plotInverseProblemResults()
        self.plotLatentx()
        self.plotLatentxy()
        self.visualizePCA()
        #self.zsamplesChains()
        #self.visualizeReconstruction()
        #self.plotGeneratedSamplesLogisticPCA()
        self.plotGeneratedSamples()
        #self.plotVFLatentVariable()
        self.zhistorgrams()
        #self.zhistorgrams_section2()
        #self.zPriorandInvSamplesHistograms()
        #self.plotDistributionStatistics_z()

        
        self.plotLoss()
        #self.Xhistorgrams()
        #self.plotGeneratedSamples()
        self.plotDistributionStatistics()
        #self.plotDistributionStatisticsX()


        
        #self.plotRandShapeFuncs()
        #self.plotResidual()
        #self.plotElbo()
        #self.plotFullElboParts()
        #self.plotRSquaredHistory()
        #self.plotRSquaredHistoryAvg()
        
        
        #self.plotVarianceConvergence()

        #self.plotComp3Figs()

        
        
        #ducself.plotCGMap()

        
        #self.plotMeanRelativeErrorsSample()
        #self.plotEigenCumul()
        #self.plotUsedWeightFuncs()


    def save(self, writingList, fullTensors=True):
        if fullTensors == True:
            for i in range(0, len(writingList)):
                torch.save(writingList[i][1], self.path+writingList[i][0]+'.dat')

    def removeAllRefSolutions(self, folder_path):
        # Check if the folder exists
        if os.path.exists(folder_path):
            # List all files and subdirectories in the folder
            folder_contents = os.listdir(folder_path)

            # Remove all files within the folder
            for item in folder_contents:
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

            # Remove all subdirectories within the folder
            for item in folder_contents:
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(folder_path)
        else:
            print(f"The folder '{folder_path}' does not exist.")

    def recreateAllRefSolFolders(self, base_path, resolution):
        # List of folder paths
        self.data = self.read()
        Nx = int(self.data['numOfTestSamples'])
        gridResolutionFenics = 'grid'+str(resolution)+'Fenics'
        gridResolutionFenicsSampleNx = 'grid'+str(resolution)+'FenicsSample'+str(Nx)
        folder_paths = [gridResolutionFenics, 'condFields', gridResolutionFenicsSampleNx]

        # Create full folder paths by joining the base path with each folder name
        folder_paths = [os.path.join(base_path, folder_name) for folder_name in folder_paths]

        # Loop over each folder path and create the folder
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def read(self):
        readingDict = {}
        for file_name in os.listdir(self.path):
            key = file_name[:-4]
            tensor = torch.load(os.path.join(self.path, file_name), weights_only=False, map_location='cpu')
            if torch.is_tensor(tensor):
                readingDict[key] = tensor.cpu()
            elif isinstance(tensor, np.ndarray):
                readingDict[key] = torch.from_numpy(tensor).cpu()
            else:
                readingDict[key] = tensor
        return readingDict

    def smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    



    def plotESSHistory(self):
        ESSHistory = self.data['ESSHistory']

        nncols = 6
        num_rows = (len(ESSHistory) + nncols-1) // nncols
        num_cols = min(nncols, len(ESSHistory))


        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))

        # Flatten the axes array if num_cols > 1
        axes = axes.flatten() if num_cols > 1 else [axes]

        # Loop through each slice and plot using pcolormesh
        for i in range(len(ESSHistory)):
            ax = axes[i]
            if i < len(ESSHistory):
                pcm = ax.plot(torch.arange(1, len(ESSHistory[str(i+1)])+1), torch.tensor(ESSHistory[str(i+1)]).to('cpu'), 'k')
                ax.set_title(f'Loss History for iteration {i + 1}')
                ax.grid(True)
                #ax.set_ylim(y_min, y_max)

        #fig.colorbar(axes, orientation='vertical')
        # self.surf.set_zlabel('Solution Value: y', fontsize=10)
        plot_title = "History of the ESS for the $\psi$ updates"
        fig.suptitle(plot_title, fontsize=14)

        #plt.tight_layout()
        plt.tight_layout()
        plt.savefig(self.fpath + "ESSHistory.png", dpi=300, bbox_inches='tight')
        if self.displayPlots == True:
            plt.show()
        else:
            plt.close()

    def plotresInterHistory(self):
        resInterHistory = self.data['resInterHistory']

        nncols = 6
        num_rows = (len(resInterHistory) + nncols-1) // nncols
        num_cols = min(nncols, len(resInterHistory))


        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))

        # Flatten the axes array if num_cols > 1
        axes = axes.flatten() if num_cols > 1 else [axes]

        # Loop through each slice and plot using pcolormesh
        for i in range(len(resInterHistory)):
            ax = axes[i]
            if i < len(resInterHistory):
                pcm = ax.plot(torch.arange(1, len(resInterHistory[str(i+1)])+1), torch.tensor(resInterHistory[str(i+1)]).to('cpu'), 'g')
                ax.set_title(f'Loss History for iteration {i + 1}')
                ax.grid(True)
                plt.yscale('log')
                #ax.set_ylim(y_min, y_max)

        #fig.colorbar(axes, orientation='vertical')
        # self.surf.set_zlabel('Solution Value: y', fontsize=10)
        plot_title = "History of the ESS for the $\psi$ updates"
        fig.suptitle(plot_title, fontsize=14)
        plt.yscale('log')

        #plt.tight_layout()
        plt.tight_layout()
        plt.savefig(self.fpath + "resInterHistory.png", dpi=300, bbox_inches='tight')
        if self.displayPlots == True:
            plt.show()
        else:
            plt.close()
    
    def plotresPhiHistory(self):
        resPhiHistory = self.data['resPhiHistory']

        nncols = 6
        num_rows = (len(resPhiHistory) + nncols-1) // nncols
        num_cols = min(nncols, len(resPhiHistory))


        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))

        # Flatten the axes array if num_cols > 1
        axes = axes.flatten() if num_cols > 1 else [axes]

        # Loop through each slice and plot using pcolormesh
        for i in range(len(resPhiHistory)):
            ax = axes[i]
            if i < len(resPhiHistory):
                pcm = ax.plot(torch.arange(1, len(resPhiHistory[str(i+1)])+1), torch.tensor(resPhiHistory[str(i+1)]).to('cpu'), 'm')
                ax.set_title(f'Loss History for iteration {i + 1}')
                ax.grid(True)
                #ax.set_ylim(y_min, y_max)

        #fig.colorbar(axes, orientation='vertical')
        # self.surf.set_zlabel('Solution Value: y', fontsize=10)
        plot_title = "History of the difference of H for the $\psi$ updates"
        fig.suptitle(plot_title, fontsize=14)

        #plt.tight_layout()
        plt.tight_layout()
        plt.savefig(self.fpath + "resPhiHistory.png", dpi=300, bbox_inches='tight')
        if self.displayPlots == True:
            plt.show()
        else:
            plt.close()

    def plotHLoss(self):
        HLoss = self.data['HLossHistory']

        nncols = 6
        num_rows = (len(HLoss) + nncols-1) // nncols
        num_cols = min(nncols, len(HLoss))


        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))

        # Flatten the axes array if num_cols > 1
        axes = axes.flatten() if num_cols > 1 else [axes]

        # Loop through each slice and plot using pcolormesh
        for i in range(len(HLoss)):
            ax = axes[i]
            if i < len(HLoss):
                pcm = ax.plot(torch.arange(1, len(HLoss[str(i+1)])+1), torch.tensor(HLoss[str(i+1)]).to('cpu'), 'b')
                ax.set_title(f'Loss History for iteration {i + 1}')
                ax.grid(True)
                #ax.set_ylim(y_min, y_max)

        #fig.colorbar(axes, orientation='vertical')
        # self.surf.set_zlabel('Solution Value: y', fontsize=10)
        plot_title = "History of H for the $\psi$ updates"
        fig.suptitle(plot_title, fontsize=14)

        #plt.tight_layout()
        plt.tight_layout()
        plt.savefig(self.fpath + "HLoss.png", dpi=300, bbox_inches='tight')
        if self.displayPlots == True:
            plt.show()
        else:
            plt.close()


 
    def plotUsedWeightFuncs(self):
        phis = self.data['usedWeightFuncs']

        sgrid = self.data['intGrid']
        grid = self.data['rbfGrid']
        gridW = self.data['rbfGridW']
        nele = grid.size(dim=1) - 1
        dl = 1 / nele
        dlW = 1 / gridW.size(dim=1)
        ### Shape Function Object Construction
        #shapeFuncs = rbfInterpolation(grid, sgrid, 1 / dl ** 2, rhs=1.)
        shapeFuncs = rbfInterpolation(grid, gridW, gridW, sgrid, 1 / dl ** 2, 1 / dlW ** 2)

        weightFuncs = shapeFuncs.cWeighFuncParallel(phis).detach()
        nncols = 6
        num_rows = (weightFuncs.size(0) + nncols-1) // nncols
        num_cols = min(nncols, weightFuncs.size(0))

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))

        # Flatten the axes array if num_cols > 1
        axes = axes.flatten() if num_cols > 1 else [axes]

        # Loop through each slice and plot using pcolormesh
        for i in range(weightFuncs.size(0)):
            ax = axes[i]
            if i < weightFuncs.size(0):
                pcm = ax.pcolormesh(sgrid[0, :, :], sgrid[1, :, :], weightFuncs[i, :, :].t(), cmap='coolwarm')
                ax.set_title(f'Weighting Functions {i + 1}')
                ax.axis('off')
            else:
                ax.axis('off')
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(pcm, cax=cbar_ax)
        cbar.set_label('Magnitude of the Weighting Functions')
        #fig.colorbar(axes, orientation='vertical')
        # self.surf.set_zlabel('Solution Value: y', fontsize=10)
        plot_title = "Visualization of the Weighting Functions"
        fig.suptitle(plot_title, fontsize=14)

        #plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(self.fpath + "usedWeightFuncs.png", dpi=300, bbox_inches='tight')
        if self.displayPlots == True:
            plt.show()
        else:
            plt.close()




    def plotLoss(self):
        # Regex pattern to match versioned files
        directory = "./checkpoints/genPANIS/lightning_logs/"
        pattern = re.compile(r"version_(\d+)")

        # Find all matching files
        files = [f for f in os.listdir(directory) if pattern.match(f)]

        if not files:
            raise FileNotFoundError("No versioned CSV files found!")

        # Extract the highest version number
        latest_file = max(files, key=lambda f: int(pattern.search(f).group(1)))

        # Full path of the latest file
        file_path = os.path.join(directory, latest_file)
        print(f"Using latest file: {file_path}")

        file_path = os.path.join(file_path, 'metrics.csv')
        df = pd.read_csv(file_path)

        # Forward-fill epoch values
        df['epoch'] = df['epoch'].fillna(method='ffill')

        # Check if 'val_bpd' exists in the DataFrame
        if 'val_bpd' in df.columns:
            # Drop rows where both train_bpd and val_bpd are NaN
            df = df.dropna(subset=['train_bpd', 'val_bpd'], how='all')
            train_loss = df[['epoch', 'train_bpd']].dropna()
            val_loss = df[['epoch', 'val_bpd']].dropna()
        else:
            # If 'val_bpd' doesn't exist, drop rows where train_bpd is NaN
            df = df.dropna(subset=['train_bpd'], how='all')
            train_loss = df[['epoch', 'train_bpd']].dropna()
            val_loss = None

        # Plot
        plt.figure(figsize=(10, 5))

        # Plot training loss
        plt.plot(
            train_loss['epoch'].to_numpy(),
            train_loss['train_bpd'].to_numpy(),
            label="Training Loss",
            marker='o'
        )

        # Optionally plot validation loss
        # if val_loss is not None:
        #     plt.plot(
        #         val_loss['epoch'].to_numpy(),
        #         val_loss['val_bpd'].to_numpy(),
        #         label="Validation Loss",
        #         marker='s',
        #         linestyle='dashed'
        #     )

        plt.title("Training vs Validation Loss")
        plt.rcParams.update({'font.size': 12})
        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel("Epochs")
        plt.ylabel("ELBO")
        # plt.legend(["ELBO", "ELBO Moving Avg."])  # Uncomment and customize if needed

        plt.savefig(self.fpath + "loss.png", dpi=300, bbox_inches='tight')

        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotResidual(self):
        # Regex pattern to match versioned files
        directory = "./checkpoints/genPANIS/lightning_logs/"
        pattern = re.compile(r"version_(\d+)")

        # Find all matching files
        files = [f for f in os.listdir(directory) if pattern.match(f)]

        if not files:
            raise FileNotFoundError("No versioned CSV files found!")

        # Extract the highest version number
        latest_file = max(files, key=lambda f: int(pattern.search(f).group(1)))

        # Full path of the latest file
        file_path = os.path.join(directory, latest_file)

        print(f"Using latest file: {file_path}")
        file_path = file_path + '/metrics.csv'
        df = pd.read_csv(file_path)


        df = df.dropna(subset=['Absolute Residual'], how='all')
        train_loss = df[['epoch', 'Absolute Residual']].dropna()
            

        # Forward-fill epoch values
        df['epoch'] = df['epoch'].fillna(method='ffill')

            

        # Plot
        plt.figure(figsize=(10, 5))

        # Plot training loss
        plt.plot(train_loss['epoch'], train_loss['Absolute Residual'], label="Training Loss", marker='o', color='green')


        plt.title("Convergence of the Absolute Mean Residual")

        #elbo_x = self.data['elbo_x']
        #elbo_y = self.data['elbo_y']
        #elbo_residuals = self.data['elbo_residuals']
        
        
        plt.rcParams.update({'font.size': 12})


        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("Epochs")
        plt.ylabel("Absolute Mean Residual")
        plt.yscale('log')
        #plt.legend(["ELBO", "ELBO Moving Avg."])
        plt.savefig(self.fpath+"residual.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    
    def plotVarianceConvergence(self):
        sigma = self.data['sigmaEvolution']
        varNorm = self.data['varNormEvolution']
        #elbo_x = self.data['elbo_x']
        #elbo_y = self.data['elbo_y']
        #elbo_residuals = self.data['elbo_residuals']

        self.plotCounter += 1
        plt.figure(self.plotCounter)
        sviCycles = torch.linspace(0, sigma.size(dim=0), sigma.size(dim=0))
        #sviCycles_elbo_x = torch.linspace(0, elbo_x.size(dim=0), elbo_x.size(dim=0))
        plt.plot(sviCycles, (sigma), '-b')
        plt.plot(sviCycles, (varNorm), '-c')
        #plt.plot(sviCycles, (movAvgElbo), 'c')

        plt.grid(True)
        plt.yscale('log')
        plt.title("$\sigma$ and Norm($V$)")
        plt.xlabel("Number of iterations")
        plt.ylabel("$\sigma$ and Norm($V$)")
        #plt.legend(["ELBO", "ELBO Moving Avg."])
        plt.savefig(self.fpath+"sigmaConvergence.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
    
    def plotCovarianceMatrix(self):
        cov = self.data['covMatrix'].clone().detach().cpu()
        #elbo_x = self.data['elbo_x']
        #elbo_y = self.data['elbo_y']
        #elbo_residuals = self.data['elbo_residuals']
        
        
        if cov.size(-1) > 64:
            plt.matshow(torch.log10(cov[:64, :64]), cmap='coolwarm')
        else:
            plt.matshow(torch.log10(cov[:, :]), cmap='coolwarm')
        plt.colorbar()
        plt.title("Final Covariance Matrix of the Approx. Posterior")
        plt.savefig(self.fpath+"covMatrix.png", dpi=300)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        
    def plotFullElboParts(self):
        totalElbo = self.data['totalElbo']
        likelihood = self.data['likelihood']
        entropy = self.data['entropy']
        logProb = self.data['logProb']



        #sviCycles = torch.linspace(0, RSquared.size(dim=0), RSquared.size(dim=0))
        #RSquared = torch.where(RSquared < 0, torch.tensor(0.001), RSquared)
        plt.plot(torch.arange(totalElbo.size(0))+1, totalElbo, '-m')
        plt.plot(torch.arange(totalElbo.size(0))+1, likelihood, '-b')
        plt.plot(torch.arange(totalElbo.size(0))+1, entropy, '-r')
        plt.plot(torch.arange(totalElbo.size(0))+1, logProb, '-g')

        plt.grid(True)
        plt.title("Full Elbo Components Progress")
        plt.xlabel("Progress (In % points)")
        plt.ylabel("Elbo Units")
        plt.yscale('symlog')
        #plt.ylim(bottom=-3., top=1.0)
        plt.legend(["Full Elbo", "Likelihood", "Entropy", "Prior"])
        plt.savefig(self.fpath + "fullElboParts.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()




    def plotRSquaredHistory(self):
        RSquared = self.data['RSquaredHistory']



        sviCycles = torch.linspace(0, RSquared.size(dim=0), RSquared.size(dim=0))
        #RSquared = torch.where(RSquared < 0, torch.tensor(0.001), RSquared)
        plt.plot(sviCycles, RSquared, '-b')

        plt.grid(True)
        plt.title("Evolution of RSquared over many random simulations")
        plt.xlabel("Number of weighting Functions added")
        plt.ylabel("RSquared")
        #plt.yscale('symlog')
        plt.ylim(bottom=-3., top=1.0)
        #plt.legend(["RSquared"])
        plt.savefig(self.fpath + "RSquaredHistory.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        
    
    def plotRSquaredHistoryAvg(self):
        RSquared = self.data['RSquaredHistoryAvg']
        externalIter = self.data['externalIter'][0]
        sviCycles = torch.linspace(0, RSquared.size(dim=0), RSquared.size(dim=0))
        randRes = torch.linspace(1, externalIter, RSquared.size(dim=0))
        RSquared = torch.where(RSquared < 0.1, torch.tensor(0.1), RSquared)
        plt.plot(sviCycles, RSquared, '-c')

        plt.grid(True)
        plt.title("Evolution of RSquared over many random simulations")
        plt.xlabel("Number of weighting Functions added")
        plt.ylabel("RSquared")
        plt.yscale('log')
        plt.ylim(bottom=-4, top=1.0)
        #plt.legend(["RSquared"])
        plt.savefig(self.fpath + "RSquaredHistoryAvg.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotRelImp(self):
        relImp = self.data['relativeImprovementOfPhi']

        self.plotCounter += 1
        plt.figure(self.plotCounter)
        relImp = torch.FloatTensor(relImp)
        phiOptCycles = torch.linspace(0, relImp.size(dim=0), relImp.size(dim=0))
        plt.plot(phiOptCycles, relImp, '-m')
        plt.grid(True)
        plt.title("Relative Improvement to sqRes when GradOpt is applied for phi")
        plt.xlabel("Number of iterations")
        plt.ylabel("relImp")
        plt.legend(["relImp"])
        plt.savefig(self.fpath+"relImp.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def readRefSolution(self, x=None):
        if x is None:
            x = self.data['xPred']
        sgrid = self.data['intGrid']
        csv_data = []
        csv_dataFenics = []
        err = []
        for i in range(0, x.size(dim=0)):
            #csv_data.append(torch.from_numpy(np.loadtxt('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1))+
            #                            '/x='+"{:.1f}".format(x[i])+'.csv', delimiter=',')))
            csv_dataFenics.append(torch.load('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1)) +
                                            'Fenics/x='+"{:.1f}".format(x[i])+'.csv'))
            #err.append(torch.mean(csv_data[i]))
            #err[i] = torch.mean(abs(csv_data[i] - torch.reshape(csv_dataFenics[i], [-1])))
        return torch.stack(csv_dataFenics).detach().cpu()

    def readTestSample(self, Nx=1000):
            self.data = self.read()
            Nx = int(self.data['numOfTestSamples'])
            gridSize = self.data['intGrid'].size(1)
            rbfGridSize = self.data['rbfGrid'].size(1)
            numOfInputs = self.data['gpEigVals'].size(0)
            sol = torch.zeros(Nx, gridSize, gridSize)
            solFenics = torch.zeros(Nx, gridSize, gridSize)
            yCoeff = torch.zeros(Nx, int(rbfGridSize**2))
            cond = torch.zeros(Nx, gridSize, gridSize)
            x = torch.zeros(Nx, numOfInputs)
            #x = torch.zeros(Nx, 20) #### Needs parameterization here
            for i in range(0, Nx):
                # csv_data.append(torch.from_numpy(np.loadtxt('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1))+
                #                            '/x='+"{:.1f}".format(x[i])+'.csv', delimiter=',')))
                sol[i, :, :] = torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                 'FenicsSample'+str(Nx)+'/sol' + "{:}".format(i) + '.csv',  map_location='cpu')
                solFenics[i, :, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                 'FenicsSample'+str(Nx)+'/solFenics' + "{:}".format(i) + '.csv',  map_location='cpu'))
                yCoeff[i, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                           'FenicsSample' + str(Nx) + '/yCoeff' + "{:}".format(i) + '.csv',  map_location='cpu'))
                cond[i, :, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                      'FenicsSample' + str(Nx) + '/Cond' + "{:}".format(i) + '.csv',  map_location='cpu'))
                #x.append((torch.load('./model/pde/RefSolutions/grid' + str(51) +
                #                       'FenicsSample' + str(Nx) + '/x' + "{:}".format(i) + '.csv')))
                x[i, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                  'FenicsSample' + str(Nx) + '/x' + "{:}".format(i) + '.csv',  map_location='cpu'))


            return sol, solFenics, cond, x, yCoeff

    def readRefCondField(self, x=None):
        if x is None:
            x = self.data['xPred']
        sgrid = self.data['intGrid']
        csv_data = []
        csv_dataFenics = []
        err = []
        for i in range(0, x.size(dim=0)):
            #csv_data.append(torch.from_numpy(np.loadtxt('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1))+
            #                            '/x='+"{:.1f}".format(x[i])+'.csv', delimiter=',')))
            csv_dataFenics.append(torch.load('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1)) +
                                            'Fenics/Cond_x='+"{:.1f}".format(x[i])+'.csv'))
            #err.append(torch.mean(csv_data[i]))
            #err[i] = torch.mean(abs(csv_data[i] - torch.reshape(csv_dataFenics[i], [-1])))
        return torch.stack(csv_dataFenics).detach().cpu()


    def calcUncertaintyMetric(self, meanSol, meanRef, stdSol, sIndex):
        """
        :param meanSol: The mean prediction of the trained surrogate
        :param meanRef: The mean solution, calculated by a numerical solver (Reference solution)
        :param stdSol: The std at each point of the domain, calculated from actual samples from the posterior
        :param sIndex: The factor with which we multiply the standard deviation (e.g. +- 2 sigma)
        :return: The Envelope Metric
        """
        x = 1. - torch.abs(meanSol.detach().cpu() - meanRef.detach().cpu()) / (sIndex * (stdSol)+10**(-6))
        #mask = torch.logical_and(x>=-1., x<=1.)
        #torch.where(mask, torch.abs(x), 1 - torch.abs(x))
        return x

  
    def plotCGMap(self):
        if 'XCG.dat' in os.listdir(self.path):
            dataDriven = False
            if not dataDriven:
                solsamp, solFenicssamp, condsamp, xsamp, yCoeffsamp = self.readTestSample()
            rbfGrid = self.data['rbfGrid']
            #yTrue = self.data['yTest']
            sgrid = self.data['intGrid']
            rbfGridSize = rbfGrid.size(dim=1)**2
            X = self.data['XCG']
            Y = self.data['YCG']
            y = self.data['yCG']
            x = self.data['xCG']
            YFenics = self.data['YCGFenics']
            yTrueProj = self.data['yProjT']
            yTrue = solFenicssamp
            
            self.reducedDim = 9
            xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim))
            xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)))
            
            
            for i in range(0, 2):
                ymin=torch.min(yTrueProj[i, :, :])
                if torch.max(yTrue[i, :, :]) > torch.max(y[i, :, :]):
                    ymax=torch.max(yTrue[i, :, :])
                else:
                    ymax=torch.max(y[i, :, :])
                xmin=torch.min(torch.tensor(-3.))
                xmax=torch.max(torch.tensor(0.))
                #xmin = torch.min(torch.log10(X[i, :, :]))
                combined_tensor = torch.stack((yTrue, y, yTrueProj))
                ymax = torch.max(combined_tensor[:, i, :, :])
                ymax=torch.max(yTrue[i, :, :])
                fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))


                xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)))

                
                # plot x
                im1 = axs[0, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(x[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
                axs[0, 0].set_title('log_10(x)')

                # plot X
                im2 = axs[0, 1].pcolormesh(xx, yy, torch.log10(X[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
                axs[0, 1].set_title('log_10(X)')

                # plot Y
                im3 = axs[1, 0].pcolormesh(xx, yy, YFenics[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
                axs[1, 0].set_title('Y')

                im6 = axs[0, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
                axs[0, 2].set_title('log10(xTrue)')

                im8 = axs[0, 3].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
                axs[0, 3].set_title('log10(xTrue)')

                # plot y
                im4 = axs[1, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], y[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
                axs[1, 1].set_title('y')
                # plot y
                im7 = axs[1, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], yTrueProj[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
                axs[1, 2].set_title('yTrue Projection')

                im9 = axs[1, 3].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], yTrue[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
                axs[1, 3].set_title('yTrue')

                # add colorbars
                cbar1 = fig.colorbar(im1, ax=[axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]])
                cbar2 = fig.colorbar(im4, ax=[axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]])
                
                cbar1.ax.set_position([0.8, 0.55, 0.03, 0.3])
                cbar2.ax.set_position([0.8, 0.15, 0.03, 0.3])

                plt.savefig(self.fpath + "MappingPlots"+str(i)+".png", dpi=300, bbox_inches='tight')
                if self.displayPlots:
                    plt.show()
                else:
                    plt.close()
            tess = 't'

    def plotRandShapeFuncs(self):
        dataDriven = False
        if not dataDriven:
            solsamp, solFenicssamp, condsamp, xsamp, yCoeffsamp = self.readTestSample()
        rbfGrid = self.data['rbfGrid']
        #yTrue = self.data['yTest']
        sgrid = self.data['intGrid']
        rbfGridSize = rbfGrid.size(dim=1)**2
        X = self.data['XCG']
        Y = self.data['YCG']
        y = self.data['yCG']
        x = self.data['xCG']
        YFenics = self.data['YCGFenics']
        yTrueProj = self.data['yProjT']
        yTrue = solFenicssamp
        
        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim))
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)))
        randShapeFuncs = self.data['randShapeFuncs']
        
        for i in range(0, 1):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

            xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)))
            xx, yy = torch.meshgrid(torch.linspace(0, 1, randShapeFuncs.size(-2)), torch.linspace(0, 1, randShapeFuncs.size(-1)))
            axs.set_xticks([])
            axs.set_yticks([])
            # plot x
            im1 = axs.pcolormesh(sgrid[0, :, :], sgrid[1, :, :], randShapeFuncs, cmap='inferno')

            plt.savefig(self.fpath + "randShapeFuncs"+".png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()
        tess = 't'

    def zhistorgrams_section2(self):
        from scipy.stats import gaussian_kde
        import matplotlib.lines as mlines
        from scipy.ndimage import gaussian_filter

        use_kde = False

        # Load generated latent codes (shape: N x D)
        X = self.data['generated_z'].squeeze(1)  # Shape: (N, D)
        X = torch.load('./experiments/latentz/genSamples.pth')['z'].cpu().squeeze(1)
        encZ1 = torch.load('./experiments/latentz/reconSamplesVF50.pth')['z'].cpu().squeeze(1)
        encZ2 = torch.load('./experiments/latentz/reconSamplesVF10.pth')['z'].cpu().squeeze(1)
        encZ3 = torch.load('./experiments/latentz/reconSamplesVF90.pth')['z'].cpu().squeeze(1)

        # -------------------------------
        # Visualization parameters
        # -------------------------------
        dim = 4  # Number of dimensions to visualize
        xlim = [-80, 80]
        ylim = [-80, 80]

        # Font configuration
        plt.rcParams.update({
            'font.size': 18,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 16,
            'figure.titlesize': 24
        })

        fig, axes = plt.subplots(dim, dim, figsize=(30, 30))
        axes = axes.flatten()

        hist_index = 0
        for i in range(dim):
            for j in range(dim):
                offset = 0
                ax = axes[hist_index]
                hist_index += 1

                # --------------------------------------
                # Diagonal: 1D KDE marginal distributions
                # --------------------------------------
                if i == j:
                    pixel_i = X[:, i + offset].numpy()

                    # Generated z's (background filled histogram)
                    ax.hist(pixel_i, bins=100, color='lightblue', alpha=0.6, density=True, label='Generated')

                    # Smooth KDEs for the encoded z's
                    for encZ, color, label in zip(
                        [encZ1, encZ2],
                        ['red', 'green'],
                        ['VF50', 'VF10']
                    ):
                        enc_i = encZ[:, i + offset].numpy()
                        kde = gaussian_kde(enc_i, bw_method=0.3)
                        x_grid = np.linspace(xlim[0], xlim[1], 400)
                        ax.plot(x_grid, kde(x_grid), color=color, lw=2.0, alpha=0.8, label=label)

                    ax.set_xlim(xlim)
                    ax.set_xlabel(f'$z_{{{i+offset}}}$')
                    ax.set_ylabel('Density')
                    ax.set_title(f'$p(z_{{{i+offset}}})$')
                    continue

                # --------------------------------------
                # Off-diagonal: 2D KDE marginal contours
                # --------------------------------------
                pixel_i = X[:, i + offset].numpy()
                pixel_j = X[:, j + offset].numpy()

                # Background 2D histogram (colormap preserved)
                h = ax.hist2d(pixel_i, pixel_j, bins=200, cmap='Blues',
                            range=[xlim, ylim], density=True)

                # Overlay KDE contours for each encoded z
                for encZ, color in zip([encZ1, encZ2], ['red', 'green']):
                    enc_i = encZ[:, i + offset].numpy()
                    enc_j = encZ[:, j + offset].numpy()

                    if use_kde:
                        # ======================================================
                        # TRUE KDE (slow)
                        # ======================================================
                        kde = gaussian_kde(np.vstack([enc_i, enc_j]), bw_method=0.3)

                        xx, yy = np.mgrid[xlim[0]:xlim[1]:200j, ylim[0]:ylim[1]:200j]
                        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

                    else:
                        # ======================================================
                        # FAST ALTERNATIVE:
                        # Histogram smoothed with Gaussian filter
                        # ======================================================
                        H, xedges, yedges = np.histogram2d(
                            enc_i, enc_j,
                            bins=100,
                            range=[xlim, ylim]
                        )

                        # Smooth with Gaussian
                        H_smooth = gaussian_filter(H, sigma=2.0)

                        # Build a grid matching histogram centers
                        xx, yy = np.meshgrid(
                            0.5 * (xedges[:-1] + xedges[1:]),
                            0.5 * (yedges[:-1] + yedges[1:])
                        )

                        zz = H_smooth.T

                    # --------------------------------------
                    # Draw contour and label values
                    # --------------------------------------
                    CS = ax.contour(xx, yy, zz,
                                    colors=color,
                                    linewidths=1.2,
                                    alpha=0.6,
                                    levels=5)

                    #ax.clabel(CS, inline=True, fontsize=10, fmt="%.3e")

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(f'$z_{{{i+offset}}}$')
                ax.set_ylabel(f'$z_{{{j+offset}}}$')

                # Optional colorbar for density shading
                fig.colorbar(h[3], ax=ax, orientation='vertical')

        # -------------------------------
        # Shared legend
        # -------------------------------
        black_line = mlines.Line2D([], [], color='red', lw=2, label='VF50')
        green_line = mlines.Line2D([], [], color='green', lw=2, label='VF10')

        fig.legend(handles=[black_line, green_line],
                loc='upper right',
                bbox_to_anchor=(0.98, 0.02),
                frameon=True,
                fontsize=22,
                title='Dataset distribution approximations')

        # -------------------------------
        # Layout and output
        # -------------------------------
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle(
            r'1D and 2D Marginal Distributions of the learned prior $p_{\mathbf{\theta}}(\mathbf{z})$',
            fontsize=28,
            y=0.92
        )

        plt.savefig(self.fpath + "XhistogramsDetailed.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()







        from matplotlib import gridspec
        # -------------------------------
        # Load data
        # -------------------------------
        name = 'VF50'
        data = torch.load('./experiments/latentz/reconSamples'+name+'.pth')
        x_fields = data['x'].cpu().squeeze(1).numpy()  # shape: (N, 129, 129)
        y_fields = data['y'].cpu().squeeze(1).numpy()  # shape: (N, 129, 129)

        # -------------------------------
        # Parameters
        # -------------------------------
        n_per_fig = 3      # how many samples (columns) per figure
        n_figures = 1      # total figures
        x_cmap = 'viridis'
        y_cmap = 'coolwarm'

        # Compute common color limits for consistent visualization
        x_vmin, x_vmax = np.percentile(x_fields, [1, 99])
        y_vmin, y_vmax = np.percentile(y_fields, [1, 99])

        # -------------------------------
        # Generate figures
        # -------------------------------
        for fig_idx in range(n_figures):
            fig = plt.figure(figsize=(14, 8))
            # GridSpec: 2 rows, 3 columns + 1 column for colorbars
            gs = gridspec.GridSpec(2, n_per_fig + 1, width_ratios=[1, 1, 1, 0.05],
                                wspace=0.15, hspace=0.15)

            start = fig_idx * n_per_fig
            indices = np.arange(start, start + n_per_fig)

            im_x, im_y = None, None

            # ----------------------------
            # Top row: x-fields
            # ----------------------------
            for col, idx in enumerate(indices):
                ax = fig.add_subplot(gs[0, col])
                im_x = ax.imshow(x_fields[idx], cmap=x_cmap, origin='lower',
                                vmin=x_vmin, vmax=x_vmax)
                ax.axis('off')  # remove axes ticks and labels

            # Vertical colorbar for x-row (without label)
            cax_x = fig.add_subplot(gs[0, -1])
            fig.colorbar(im_x, cax=cax_x, orientation='vertical', ticklocation='right')

            # ----------------------------
            # Bottom row: y-fields
            # ----------------------------
            for col, idx in enumerate(indices):
                ax = fig.add_subplot(gs[1, col])
                im_y = ax.imshow(y_fields[idx], cmap=y_cmap, origin='lower',
                                vmin=y_vmin, vmax=y_vmax)
                ax.axis('off')

            # Vertical colorbar for y-row (without label)
            cax_y = fig.add_subplot(gs[1, -1])
            fig.colorbar(im_y, cax=cax_y, orientation='vertical', ticklocation='right')

            # ----------------------------
            # Tight layout and save
            # ----------------------------
            plt.tight_layout()
            plt.savefig(self.fpath + name+'.png', dpi=300, bbox_inches='tight')
        




        
    def zhistorgrams(self):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')

        # Load generated latent codes (shape: N x D)
        X = self.data['generated_z'].squeeze(1).detach().cpu()  # Shape: (N, D)

        # Set how many dimensions to visualize
        dim = 7
        dim2 = 7
        n_univariate = dim2 ** 2
        fig, axes = plt.subplots(dim, dim, figsize=(30, 30))  # dim x dim grid

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Compute per-dimension mean and std (for axis limiting)
        z_means = X.mean(dim=0).numpy()
        z_stds = X.std(dim=0).numpy()
        max_std = z_stds[:dim].max()  # max std over selected dims

        hist_index = 0
        for i in range(dim):  # Rows
            for j in range(dim):  # Columns
                ax = axes[hist_index]

                # Get data for z_i and z_j
                pixel_i = X[:, i].numpy()
                pixel_j = X[:, j].numpy()
                #pixel_i2 = X[-1000:, i].numpy()
                #pixel_j2 = X[-1000:, j].numpy()



                # Compute ±4σ limits
                #xlim = (z_means[i] - 4 * z_stds[i], z_means[i] + 4 * z_stds[i])
                #ylim = (z_means[j] - 4 * z_stds[j], z_means[j] + 4 * z_stds[j])
                #xlim = (z_means[i] - 4 * max_std, z_means[i] + 4 * max_std)
                #ylim = (z_means[j] - 4 * max_std, z_means[j] + 4 * max_std)
                xlim = [-40, 40]
                ylim = [-40, 40]

                # 2D histogram with axis limits
                h = ax.hist2d(pixel_i, pixel_j, bins=200, cmap='Blues', range=[xlim, ylim], density=True)
                #h2 = ax.hist2d(pixel_i2, pixel_j2, bins=200, cmap='Reds', range=[xlim, ylim], density=True)

                # Add red dot for first z and green dot for second z
                #ax.scatter(X[0, i].item(), X[0, j].item(), color='red', s=30, label='z1')
                #ax.scatter(X[1, i].item(), X[1, j].item(), color='green', s=30, label='z2')
                #ax.scatter(X[2, i].item(), X[2, j].item(), color='m', s=30, label='z3')

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                # Labeling
                fig.colorbar(h[3], ax=ax, orientation='vertical')
                #fig.colorbar(h2[3], ax=ax, orientation='vertical')
                ax.set_xlabel(f'$z_{{{i}}}$')
                ax.set_ylabel(f'$z_{{{j}}}$')
                ax.set_title(f'Marginal Distribution $p(z_{{{i}}}, z_{{{j}}})$')

                hist_index += 1

        # Adjust layout
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # Global title
        fig.suptitle(r'2D Marginal Distributions of the learned prior $p_{\mathbf{\theta}}(\mathbf{z})$', fontsize=24, y=0.92)

        # Save and show
        plt.savefig(self.fpath + "Xhistograms.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        
        # ==========================================================
        # 2) Univariate histograms for the first dim^2 latent dims
        # ==========================================================
        fig_uni, axes_uni = plt.subplots(dim2, dim2, figsize=(30, 30))
        axes_uni = axes_uni.flatten()

        for i in range(n_univariate):
            ax = axes_uni[i]
            if i < X.shape[1]:
                zi = X[:, i].numpy()
                counts, bins, _ = ax.hist(zi, bins=200, density=True, color='steelblue', alpha=0.6, label='Empirical')
                # Gaussian overlay
                # Compute mean and std for Gaussian overlay
                mu = np.mean(zi)
                sigma = np.std(zi)
                x = np.linspace(bins[0], bins[-1], 500)
                y = norm.pdf(x, mu, sigma)
                ax.plot(x, y, 'r-', linewidth=2, label=f'$\mathcal{{N}}({mu:.2f}, {sigma:.2f})$')

                ax.set_title(f'Univariate Distribution $p(z_{{{i}}})$')
                ax.set_xlabel(f'$z_{{{i}}}$')
                ax.set_ylabel('Frequency')
            else:
                # Hide extra axes if not enough dimensions
                ax.axis('off')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig_uni.suptitle(r'Univariate Marginals $p(z_i)$ of the learned prior $p_{\mathbf{\theta}}(\mathbf{z})$', fontsize=24, y=0.92)
        plt.savefig(self.fpath + "Xhistograms_1D.png", dpi=300, bbox_inches='tight')

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close(fig_uni)
    
    def zPriorandInvSamplesHistograms(self):

        # Load generated latent codes (shape: N x D)
        X = self.data['generated_z'].squeeze(1)  # Shape: (N, D)
        z_samples = torch.load('./results/data/resInverse.pt')['z_samples'].cpu()

        # Set how many dimensions to visualize
        dim = 5
        fig, axes = plt.subplots(dim, dim, figsize=(30, 30))  # dim x dim grid

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Compute per-dimension mean and std (for axis limiting)
        z_means = X.mean(dim=0).numpy()
        z_stds = X.std(dim=0).numpy()
        max_std = z_stds[:dim].max()  # max std over selected dims

        hist_index = 0
        for i in range(dim):  # Rows
            for j in range(dim):  # Columns
                ax = axes[hist_index]

                # Get data for z_i and z_j
                pixel_i = X[:, i].numpy()
                pixel_j = X[:, j].numpy()

                sample_i = z_samples[:, i].numpy()
                sample_j = z_samples[:, j].numpy()

                # Axis limits
                xlim = [-40, 40]
                ylim = [-40, 40]

                # Primary histogram (generated_z)
                h1 = ax.hist2d(pixel_i, pixel_j, bins=30, cmap='Blues', range=[xlim, ylim])

                # Overlay secondary histogram (z_samples) using magma colormap
                h2 = ax.hist2d(sample_i, sample_j, bins=30, cmap='magma', alpha=0.5, range=[xlim, ylim])

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                # Labeling
                fig.colorbar(h1[3], ax=ax, orientation='vertical')
                ax.set_xlabel(f'$z_{{{i}}}$')
                ax.set_ylabel(f'$z_{{{j}}}$')
                ax.set_title(f'Marginal Distribution $p(z_{{{i}}}, z_{{{j}}})$')

                hist_index += 1

        # Adjust layout
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # Global title
        fig.suptitle(r'2D Marginal Distributions: $p_{\theta}(\mathbf{z})$ (blue) and $q(\mathbf{z})$ (magma)', fontsize=24, y=0.92)

        # Save and show
        plt.savefig(self.fpath + "zPriorAndzSamplesHist.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def Xhistorgrams(self):
        #DuX = self.data['DuX']
        #empirical_mean_X = torch.mean(DuX.flatten(-2), dim=0)
        #empX_norm = DuX.flatten(-2) - empirical_mean_X
        #empCovX = torch.cov(empX_norm.t())
        #empL = torch.cholesky(empCovX)
        # Replace this with your actual dataset
        #X = DuX.flatten(-2)  # 1000 grayscale 4x4 images
        #X = empirical_mean_X + torch.einsum('ij,...j->...i', empL, torch.randn(1000, empirical_mean_X.size(-1)))
        # Step 1: Flatten the images (1000, 1, 4, 4) -> (1000, 16)
        X = torch.log(self.data['generated_X'].squeeze(1).flatten(-2))
        #X = torch.load('./generatedXflow.pt').squeeze(1).flatten(-2).detach().cpu()

        # Step 2: Plot pairwise histograms for the first 4 pixels (as an example)
        # You can change this to other pixel pairs if needed
        dim = 5
        fig, axes = plt.subplots(dim, dim, figsize=(30, 30))  # 3x3 grid of subplots

        # Flatten axes to easily iterate over them
        axes = axes.flatten()

        # Step 3: Loop through all pairs of pixels (for example, first 4 pixels)
        # We'll create a 2D histogram for each pair
        hist_index = 0
        for i in range(dim):  # First 4 pixels
            for j in range(dim):  # Pair them (i, j)
                ax = axes[hist_index]
                
                # Get data for the pair of pixels
                pixel_i = X[:, i].numpy()
                pixel_j = X[:, j].numpy()
                
                # Create 2D histogram (joint distribution) of the pair
                h = ax.hist2d(pixel_i, pixel_j, bins=30, cmap='Blues')
                
                # Set axis labels and title
                fig.colorbar(h[3], ax=ax, orientation='vertical')
                ax.set_xlabel(f'Pixel {i}')
                ax.set_ylabel(f'Pixel {j}')
                ax.set_xlim(-3, 1)
                ax.set_ylim(-3, 1)
                ax.set_title(f'Pairwise Histogram: Pixel {i} vs Pixel {j}')
                
                hist_index += 1

        # Adjust layout for better spacing
        plt.savefig(self.fpath + "Xhistograms"+str(i)+".png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        tess = 't'
    
    def plotDistributionStatisticsX(self):
        Dux = self.data['Dux'].squeeze(1)
        n = Dux.size(-1)
        Dux = Dux.flatten(-2)
        x = self.data['generated_x'].squeeze(1).flatten(-2)
        DuX = self.data['DuX']
        N = DuX.size(-1)
        DuX = DuX.flatten(-2)
        X = self.data['generated_X'].squeeze(1).flatten(-2)
        #DuX = self.data['DuX']
        #empirical_mean_X = torch.mean(DuX.flatten(-2), dim=0)
        #empX_norm = DuX.flatten(-2) - empirical_mean_X
        #empCovX = torch.cov(empX_norm.t())
        #empL = torch.cholesky(empCovX)
        # Replace this with your actual dataset
        #X = DuX.flatten(-2)  # 1000 grayscale 4x4 images
        #X = empirical_mean_X + torch.einsum('ij,...j->...i', empL, torch.randn(1000, empirical_mean_X.size(-1)))
        # Step 1: Flatten the images (1000, 1, 4, 4) -> (1000, 16)
        ## Statistics for x ##
        empMeanx = torch.mean(X, dim=0)
        empMeanDux = torch.mean(DuX, dim=0)
        empVarDux = torch.var(DuX, dim=0, unbiased=True)
        empVarx = torch.var(X, dim=0, unbiased=True)
        empCovx = torch.cov(X.t())
        empCovDux = torch.cov(DuX.t())
        #X = torch.load('./generatedXflow.pt').squeeze(1).flatten(-2).detach().cpu()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = min(empMeanDux.min(), empMeanx.min())
        vmax = max(empMeanDux.max(), empMeanx.max())

        im1 = axes[0].pcolormesh(empMeanDux.reshape(N, N), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(empMeanx.reshape(N, N), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("Generated")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(((empMeanDux - empMeanx)/empMeanDux.abs()).reshape(N, N).abs()*100, cmap='inferno')
        axes[2].set_title("Absolute Relative Error (%)")
        fig.colorbar(im3, ax=axes[2])
        fig.suptitle("Empirical Mean (10000 samples)", fontsize=18, fontweight='bold')

        plt.savefig(self.fpath + "empMeanCompX"+".png", dpi=300, bbox_inches='tight')


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = min(empVarDux.min(), empVarx.min())
        vmax = max(empVarDux.max(), empVarx.max())

        im1 = axes[0].pcolormesh(empVarDux.reshape(N, N), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(empVarx.reshape(N, N), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("Generated")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(((empVarDux - empVarx)/empVarDux).reshape(N, N).abs()*100, cmap='inferno')
        axes[2].set_title("Absolute Relative Error (%)")
        fig.colorbar(im3, ax=axes[2])
        fig.suptitle("Empirical Variance (10000 samples)", fontsize=18, fontweight='bold')
        plt.savefig(self.fpath + "empVarCompX"+".png", dpi=300, bbox_inches='tight')


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = min(empCovDux.min(), empCovx.min())
        vmax = max(empCovDux.max(), empCovx.max())

        im1 = axes[0].pcolormesh(empCovDux.reshape(N**2, N**2), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(empCovx.reshape(N**2, N**2), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("Generated")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(((empCovDux - empCovx)).reshape(N**2, N**2).abs()*100, cmap='inferno')
        axes[2].set_title("Absolute Relative Error (%)")
        fig.colorbar(im3, ax=axes[2])
        fig.suptitle("Empirical Covariance Matrix (10000 samples)", fontsize=18, fontweight='bold')
        plt.savefig(self.fpath + "empCovCompX"+".png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        tess = 't'


    def plotDistributionStatistics(self):
        Dux = self.data['Dux'].squeeze(1)
        n = Dux.size(-1)
        Dux = Dux.flatten(-2)
        x = self.data['generated_x'].squeeze(1).flatten(-2)
        DuX = self.data['DuX']
        N = DuX.size(-1)
        DuX = DuX.flatten(-2)
        X = self.data['generated_X'].squeeze(1).flatten(-2)
        #DuX = self.data['DuX']
        #empirical_mean_X = torch.mean(DuX.flatten(-2), dim=0)
        #empX_norm = DuX.flatten(-2) - empirical_mean_X
        #empCovX = torch.cov(empX_norm.t())
        #empL = torch.cholesky(empCovX)
        # Replace this with your actual dataset
        #X = DuX.flatten(-2)  # 1000 grayscale 4x4 images
        #X = empirical_mean_X + torch.einsum('ij,...j->...i', empL, torch.randn(1000, empirical_mean_X.size(-1)))
        # Step 1: Flatten the images (1000, 1, 4, 4) -> (1000, 16)
        ## Statistics for x ##
        empMeanx = torch.mean(x, dim=0)
        empMeanDux = torch.mean(Dux, dim=0)
        empVarDux = torch.var(Dux, dim=0, unbiased=True)
        empVarx = torch.var(x, dim=0, unbiased=True)
        empCovx = torch.cov(x.t())
        empCovDux = torch.cov(Dux.t())
        #X = torch.load('./generatedXflow.pt').squeeze(1).flatten(-2).detach().cpu()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = min(empMeanDux.min(), empMeanx.min())
        vmax = max(empMeanDux.max(), empMeanx.max())

        im1 = axes[0].pcolormesh(empMeanDux.reshape(n, n), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(empMeanx.reshape(n, n), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("Generated")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(((empMeanDux - empMeanx)/empMeanDux.abs()).reshape(n, n).abs()*100, cmap='inferno')
        axes[2].set_title("Absolute Relative Error (%)")
        fig.colorbar(im3, ax=axes[2])
        fig.suptitle(r"Empirical Mean $\left( \mathbf{m} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i, \ N=10000 \right)$", fontsize=18, fontweight='bold', y=1.08)

        plt.savefig(self.fpath + "empMeanComp"+".png", dpi=300, bbox_inches='tight')


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = min(empVarDux.min(), empVarx.min())
        vmax = max(empVarDux.max(), empVarx.max())

        im1 = axes[0].pcolormesh(empVarDux.reshape(n, n), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(empVarx.reshape(n, n), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("Generated")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(((empVarDux - empVarx)/empVarDux).reshape(n, n).abs()*100, cmap='inferno')
        axes[2].set_title("Absolute Relative Error (%)")
        fig.colorbar(im3, ax=axes[2])
        fig.suptitle(r"Empirical Variance $\left( \mathbf{v} = \frac{1}{N-1} \sum_{i=1}^N \left( \mathbf{x}_i - \mathbf{m} \right)^2, \ N=10000 \right)$", fontsize=18, fontweight='bold', y=1.08)
        plt.savefig(self.fpath + "empVarComp"+".png", dpi=300, bbox_inches='tight')


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = min(empCovDux.min(), empCovx.min())
        vmax = max(empCovDux.max(), empCovx.max())

        im1 = axes[0].pcolormesh(empCovDux.reshape(n**2, n**2), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(empCovx.reshape(n**2, n**2), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("Generated")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(((empCovDux - empCovx)).reshape(n**2, n**2).abs()*100, cmap='inferno')
        axes[2].set_title("Absolute Relative Error (%)")
        fig.colorbar(im3, ax=axes[2])
        fig.suptitle(r"Empirical Covariance $\left( \mathbf{c} = \frac{1}{N-1} \sum_{i=1}^N \left( \mathbf{x}_i - \mathbf{m} \right) \left( \mathbf{x}_i - \mathbf{m} \right)^T, \ N=10000 \right)$", fontsize=18, fontweight='bold', y=1.08)
        plt.savefig(self.fpath + "empCovComp"+".png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        tess = 't'

    def plotDistributionStatistics_z(self):
        Dux = self.data['Duz'].squeeze(1)
        n = 10
        x = self.data['generated_z'].squeeze(1)

        #DuX = self.data['DuX']
        #empirical_mean_X = torch.mean(DuX.flatten(-2), dim=0)
        #empX_norm = DuX.flatten(-2) - empirical_mean_X
        #empCovX = torch.cov(empX_norm.t())
        #empL = torch.cholesky(empCovX)
        # Replace this with your actual dataset
        #X = DuX.flatten(-2)  # 1000 grayscale 4x4 images
        #X = empirical_mean_X + torch.einsum('ij,...j->...i', empL, torch.randn(1000, empirical_mean_X.size(-1)))
        # Step 1: Flatten the images (1000, 1, 4, 4) -> (1000, 16)
        ## Statistics for x ##
        empMeanx = torch.mean(x, dim=0)
        empMeanDux = torch.mean(Dux, dim=0)
        empVarDux = torch.var(Dux, dim=0, unbiased=True)
        empVarx = torch.var(x, dim=0, unbiased=True)
        empCovx = torch.cov(x.t())
        empCovDux = torch.cov(Dux.t())
        #X = torch.load('./generatedXflow.pt').squeeze(1).flatten(-2).detach().cpu()

        # Plot Empirical Means
        plt.figure(figsize=(10, 5))
        plt.plot(empMeanDux, label="Ground Truth Mean", color='blue', linewidth=2)
        plt.plot(empMeanx, label="Generated Mean", color='green', linestyle='--', linewidth=2)
        plt.plot((empMeanDux - empMeanx).abs(), label="Absolute Error", color='red', linestyle=':', linewidth=2)
        plt.title("Empirical Mean Comparison (10000 samples)", fontsize=16, fontweight='bold')
        plt.xlabel("Dimension")
        plt.ylabel("Mean Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.fpath + "empMeanz.png", dpi=300)

        # Plot Empirical Variances
        plt.figure(figsize=(10, 5))
        plt.plot(empVarDux, label="Ground Truth Variance", color='blue', linewidth=2)
        plt.plot(empVarx, label="Generated Variance", color='green', linestyle='--', linewidth=2)
        plt.plot((empVarDux - empVarx).abs(), label="Absolute Error", color='red', linestyle=':', linewidth=2)
        plt.title("Empirical Variance Comparison (10000 samples)", fontsize=16, fontweight='bold')
        plt.xlabel("Dimension")
        plt.ylabel("Variance Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.fpath + "empVarz.png", dpi=300)


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = min(empCovDux.min(), empCovx.min())
        vmax = max(empCovDux.max(), empCovx.max())

        im1 = axes[0].pcolormesh(empCovDux.reshape(n**2, n**2), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(empCovx.reshape(n**2, n**2), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("Generated")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(((empCovDux - empCovx)).reshape(n**2, n**2).abs(), cmap='inferno')
        axes[2].set_title("Absolute Relative Error (%)")
        fig.colorbar(im3, ax=axes[2])
        fig.suptitle("Empirical Covariance Matrix (10000 samples)", fontsize=18, fontweight='bold')
        plt.savefig(self.fpath + "empCovz"+".png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        tess = 't'

    def plotForwardProblemResults(self, data_path='./results/data/resForward.pt'):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        device = torch.device('cpu')
        resFor = torch.load(data_path, map_location=device, weights_only=False)
        resForg = resFor
        if False:
            resFor = torch.load('./experiments/diffSNR_compPINO/darcy/f.pt', map_location=device, weights_only=False)
            resFor['ymean'] = torch.load('./experiments/diffSNR_compPINO_physics/darcy/pforwardfull.pt', map_location=device, weights_only=False).cpu()
        #resFor = torch.load('./experiments/latentz/flow12f.pt')
        X = torch.exp(resForg['Xmean']).detach().cpu()
        xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

        xmax = resForg['x_obs'].detach().cpu().max()
        xmin = resForg['x_obs'].detach().cpu().min()
        EnvMetric = self.calcUncertaintyMetric(resFor['ymean'].detach().cpu(), meanRef=resFor['yTrue'].detach().cpu(),
                                                        stdSol=resFor['ystd'].detach().cpu(), sIndex=2)
        EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
        EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))

        # Relative L2 error for y
        ytrue = resFor['yTrue'].detach().cpu()
        ymean = resFor['ymean'].detach().cpu()
        rel_l2_error = torch.norm(ytrue - ymean) / torch.norm(ytrue)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 3 rows now

        # First row
        im0 = axes[0, 0].pcolormesh(resForg['x_obs'].t().detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0, 0].set_title(r"Observed $\mathbf{x}$ (Fully Observed, No Noise)")
        fig.colorbar(im0, ax=axes[0, 0])

        im4 = axes[0, 1].pcolormesh(xx, yy, X.cpu(), cmap='viridis', vmin=0., vmax=1.)
        axes[0, 1].set_title(r"Mean of Inferred $\mathbf{X}$")
        fig.colorbar(im4, ax=axes[0, 1])

        im1 = axes[0, 2].pcolormesh(resFor['yTrue'].t().detach().cpu(), cmap='coolwarm')
        axes[0, 2].set_title(r"Ground Truth $\mathbf{y}$")
        fig.colorbar(im1, ax=axes[0, 2])

        # Second row
        im2 = axes[1, 0].pcolormesh(resFor['xmean'].t().detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[1, 0].set_title(r"Mean of Inferred $\mathbf{x}$")
        fig.colorbar(im2, ax=axes[1, 0])

        im3 = axes[1, 1].pcolormesh(resFor['ymean'].t().detach().cpu(), cmap='coolwarm')
        axes[1, 1].set_title(r"Mean of Inferred $\mathbf{y}$")
        fig.colorbar(im3, ax=axes[1, 1])

        abs_err = torch.abs(ytrue - ymean).t()
        im_err = axes[1, 2].pcolormesh(abs_err, cmap='inferno')
        axes[1, 2].set_title(r"Absolute Error: $|\mathbf{y}_{True} - \mathbf{y}|$")
        fig.colorbar(im_err, ax=axes[1, 2])
        axes[1, 2].text(0.5, -0.15, f'Rel L2: {rel_l2_error.item():.3e}', 
                    transform=axes[1, 2].transAxes, ha='center', va='top', fontsize=10)

        # Third row — plot same xstd in all three columns
        xstd = resFor['xstd'].t().detach().cpu()
        ystd = resFor['ystd'].t().detach().cpu()


        im_std_0 = axes[2, 0].pcolormesh(xstd, cmap='jet')
        axes[2, 0].set_title(r"Std. Dev. of Inferred $\mathbf{x}$")
        fig.colorbar(im_std_0, ax=axes[2, 0])

        im_std_1 = axes[2, 1].pcolormesh(xx, yy, X.cpu(), cmap='viridis', vmin=0., vmax=1.)
        axes[2, 1].set_title(r"Ground-Truth $\mathbf{X}$")
        fig.colorbar(im_std_1, ax=axes[2, 1])

        im_std_2 = axes[2, 2].pcolormesh(ystd, cmap='plasma')
        axes[2, 2].set_title(r"Std. Dev. of Inferred $\mathbf{y}$ ("+f'{EnvelopScore.item():.4f})')
        fig.colorbar(im_std_2, ax=axes[2, 2])

        fig.suptitle("Forward Problem Results", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        plt.savefig(self.fpath + "resForward.png", dpi=300, bbox_inches='tight')




        z_samples = resFor['z_samples'].detach().cpu()
        #resCondition = torch.load('./results/data/resCondition.pt')

        plt.figure(figsize=(12, 6))

        # Plot the evolution of the first 5 dimensions
        for i in range(5):
            plt.plot(z_samples[:, i], label=f'z[{i}]')

        plt.title(r"Evolution of $z_i$, $i=1, 2, ..., 5$")
        plt.xlabel("Sample Index")
        plt.ylabel(r"$z_i$")
        plt.legend()
        plt.grid(True)

        plt.savefig(self.fpath + "zchainsForward.png", dpi=300)

        z = resFor['z_samples'].detach().cpu()
        z_delta = z[0]
        # Prepare the figure
        fig, axs = plt.subplots(5, 5, figsize=(15, 15))
        xlim = [-40, 40]
        ylim = [-40, 40]

        for i in range(5):      # row index (z_i)
            for j in range(5):  # column index (z_j)
                ax = axs[i, j]
                x = z[:, j].numpy()
                y = z[:, i].numpy()
                x_d = z_delta[j].item()
                y_d = z_delta[i].item()
                
                if i != j:
                    ax.hist2d(x, y, bins=40, cmap='plasma')
                    ax.plot(x_d, y_d, 'ro')  # red dot
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    
                else:
                    ax.hist(x, bins=40, color='gray')
                    ax.axvline(x_d, color='red', linestyle='--')  # red vertical line for 1D

                if i == 4:
                    ax.set_xlabel(f'z{j+1}')
                else:
                    ax.set_xticks([])

                if j == 0:
                    ax.set_ylabel(f'z{i+1}')
                else:
                    ax.set_yticks([])

        plt.tight_layout()
        plt.show()
        plt.savefig(self.fpath + "histograms_deltaEncoder.png", dpi=300)

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def plotInverseProblemResults(self, data_path='./results/data/resInverse.pt'):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        device = torch.device('cpu')
        resInv = torch.load(data_path, map_location=device, weights_only=False)
        #resInv = torch.load('./experiments/partialObs_compPINO/L5x5.pt')
        X = torch.exp(resInv['Xm']).detach().cpu()
        xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

        # Classification: threshold xmean to get binary field
        xtrue = resInv['xTrue'].detach().to('cpu')
        xmax = xtrue.max()
        xmin = xtrue.min()
        xmean_raw = resInv['xmean'].detach().to('cpu')
        xmean_classified = (xmean_raw > (xmax-xmin)/2+xmin).float()
        xmean_classified = xmean_classified * (xmax-xmin) + xmin # Converts to values 0.1 or 1.0

        classification_mask = (xtrue == xmin) | (xtrue == xmax)
        classification_correct = ((xmean_classified == xtrue) & classification_mask).sum()
        classification_total = classification_mask.sum()
        classification_accuracy = (classification_correct / classification_total).item()

        # Relative L2 error for y
        ytrue = resInv['yTrue'].detach().cpu()
        ymean = resInv['ymean'].detach().cpu()
        rel_l2_error = torch.norm(ytrue - ymean) / torch.norm(ytrue)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 3 rows now

        # First row
        im0 = axes[0, 0].pcolormesh(xtrue.t(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0, 0].set_title(r"Ground Truth $\mathbf{x}$")
        fig.colorbar(im0, ax=axes[0, 0])

        im4 = axes[0, 1].pcolormesh(xx, yy, X, cmap='viridis', vmin=0., vmax=1.)
        axes[0, 1].set_title(r"Mean of Inferred $\mathbf{X}$")
        fig.colorbar(im4, ax=axes[0, 1])

        im1 = axes[0, 2].pcolormesh(resInv['y_obs'].t().cpu(), cmap='coolwarm')
        axes[0, 2].set_title(r"Observed $\mathbf{y}$ (Fully Observed, No Noise)")
        fig.colorbar(im1, ax=axes[0, 2])

        if 'obs_indices' in resInv:
            obs_indices = resInv['obs_indices'].detach().cpu()
            obs_x = obs_indices[1]
            obs_y = obs_indices[0]
            xx_obs, yy_obs = torch.meshgrid(obs_x, obs_y, indexing='ij')
            axes[0, 2].plot(xx_obs.flatten(), yy_obs.flatten(), 'k*', markersize=8)

        # Second row
        im2 = axes[1, 0].pcolormesh(xmean_raw.t(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[1, 0].set_title(r"Mean of Inferred $\mathbf{x}$")
        fig.colorbar(im2, ax=axes[1, 0])
        axes[1, 0].text(0.5, -0.15, f'Acc: {classification_accuracy:.3f}', 
                    transform=axes[1, 0].transAxes, ha='center', va='top', fontsize=10)

        im3 = axes[1, 1].pcolormesh(ymean.t(), cmap='coolwarm')
        axes[1, 1].set_title(r"Mean of Inferred $\mathbf{y}$")
        fig.colorbar(im3, ax=axes[1, 1])

        abs_err = torch.abs(ytrue - ymean).t()
        im_err = axes[1, 2].pcolormesh(abs_err, cmap='inferno')
        axes[1, 2].set_title(r"Absolute Error: $|\mathbf{y}_{True} - \mathbf{y}|$")
        fig.colorbar(im_err, ax=axes[1, 2])
        axes[1, 2].text(0.5, -0.15, f'Rel L2: {rel_l2_error.item():.3e}', 
                    transform=axes[1, 2].transAxes, ha='center', va='top', fontsize=10)

        # Third row
        xstd = resInv['xstd'].t().cpu()
        ystd = resInv['ystd'].t().cpu()

        im_std_0 = axes[2, 0].pcolormesh(xstd, cmap='jet')
        axes[2, 0].set_title(r"Std. Dev. of Inferred $\mathbf{x}$")
        fig.colorbar(im_std_0, ax=axes[2, 0])

        im_std_1 = axes[2, 1].pcolormesh(xx, yy, X, cmap='viridis', vmin=0., vmax=1.)
        axes[2, 1].set_title(r"Ground-Truth $\mathbf{X}$")
        fig.colorbar(im_std_1, ax=axes[2, 1])

        im_std_2 = axes[2, 2].pcolormesh(ystd, cmap='plasma')
        axes[2, 2].set_title(r"Std. Dev. of Inferred $\mathbf{y}$")
        fig.colorbar(im_std_2, ax=axes[2, 2])

        fig.suptitle("Inverse Problem Results", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.fpath + "resInverse.png", dpi=300, bbox_inches='tight')




        z_samples = resInv['z_samples'].detach().cpu()
        #resCondition = torch.load('./results/data/resCondition.pt')

        plt.figure(figsize=(12, 6))

        # Plot the evolution of the first 5 dimensions
        for i in range(5):
            plt.plot(z_samples[:, i], label=f'z[{i}]')

        plt.title(r"Evolution of $z_i$, $i=1, 2, ..., 5$")
        plt.xlabel("Sample Index")
        plt.ylabel(r"$z_i$")
        plt.legend()
        plt.grid(True)

        plt.savefig(self.fpath + "zchainsInverse.png", dpi=300)

        z = resInv['z_samples'].detach().cpu()
        #z_delta = z[0]
        # Prepare the figure
        fig, axs = plt.subplots(5, 5, figsize=(15, 15))
        xlim = [-40, 40]
        ylim = [-40, 40]

        for i in range(5):      # row index (z_i)
            for j in range(5):  # column index (z_j)
                ax = axs[i, j]
                x = z[:, j].numpy()
                y = z[:, i].numpy()
                
                if i != j:
                    ax.hist2d(x, y, bins=40, cmap='plasma')
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    
                else:
                    ax.hist2d(x, y, bins=40, cmap='plasma')
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                if i == 4:
                    ax.set_xlabel(f'z{j+1}')
                else:
                    ax.set_xticks([])

                if j == 0:
                    ax.set_ylabel(f'z{i+1}')
                else:
                    ax.set_yticks([])

        plt.tight_layout()
        plt.show()
        plt.savefig(self.fpath + "histograms_deltaEncoder.png", dpi=300)

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def plotForwardResults(self, data_path='./results/data/resForward.pt', name='forward'):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        device = torch.device('cpu')
        resFor = torch.load(data_path, map_location=device, weights_only=False)
        resForg = resFor

        X = torch.exp(resForg['Xmean']).detach().cpu()
        xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

        xmax = resForg['x_obs'].detach().cpu().max()
        xmin = resForg['x_obs'].detach().cpu().min()
        EnvMetric = self.calcUncertaintyMetric(resFor['ymean'].detach().cpu(), meanRef=resFor['yTrue'].detach().cpu(),
                                                        stdSol=resFor['ystd'].detach().cpu(), sIndex=2)
        EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
        EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))

        ytrue = resFor['yTrue'].detach().cpu()
        ymean = resFor['ymean'].detach().cpu()
        rel_l2_error = torch.norm(ytrue - ymean) / torch.norm(ytrue)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        im0 = axes[0, 0].pcolormesh(resForg['x_obs'].t().detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0, 0].set_title(r"Observed $\mathbf{x}$ (Fully Observed, No Noise)")
        fig.colorbar(im0, ax=axes[0, 0])

        im4 = axes[0, 1].pcolormesh(xx, yy, X.cpu(), cmap='viridis', vmin=0., vmax=1.)
        axes[0, 1].set_title(r"Mean of Inferred $\mathbf{X}$")
        fig.colorbar(im4, ax=axes[0, 1])

        im1 = axes[0, 2].pcolormesh(resFor['yTrue'].t().detach().cpu(), cmap='coolwarm')
        axes[0, 2].set_title(r"Ground Truth $\mathbf{y}$")
        fig.colorbar(im1, ax=axes[0, 2])

        im2 = axes[1, 0].pcolormesh(resFor['xmean'].t().detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[1, 0].set_title(r"Mean of Inferred $\mathbf{x}$")
        fig.colorbar(im2, ax=axes[1, 0])

        im3 = axes[1, 1].pcolormesh(resFor['ymean'].t().detach().cpu(), cmap='coolwarm')
        axes[1, 1].set_title(r"Mean of Inferred $\mathbf{y}$")
        fig.colorbar(im3, ax=axes[1, 1])

        abs_err = torch.abs(ytrue - ymean).t()
        im_err = axes[1, 2].pcolormesh(abs_err, cmap='inferno')
        axes[1, 2].set_title(r"Absolute Error: $|\mathbf{y}_{True} - \mathbf{y}|$")
        fig.colorbar(im_err, ax=axes[1, 2])
        axes[1, 2].text(0.5, -0.15, f'Rel L2: {rel_l2_error.item():.3e}',
                    transform=axes[1, 2].transAxes, ha='center', va='top', fontsize=10)

        xstd = resFor['xstd'].t().detach().cpu()
        ystd = resFor['ystd'].t().detach().cpu()

        im_std_0 = axes[2, 0].pcolormesh(xstd, cmap='jet')
        axes[2, 0].set_title(r"Std. Dev. of Inferred $\mathbf{x}$")
        fig.colorbar(im_std_0, ax=axes[2, 0])

        im_std_1 = axes[2, 1].pcolormesh(xx, yy, X.cpu(), cmap='viridis', vmin=0., vmax=1.)
        axes[2, 1].set_title(r"Ground-Truth $\mathbf{X}$")
        fig.colorbar(im_std_1, ax=axes[2, 1])

        im_std_2 = axes[2, 2].pcolormesh(ystd, cmap='plasma')
        axes[2, 2].set_title(r"Std. Dev. of Inferred $\mathbf{y}$ (" + f'{EnvelopScore.item():.4f})')
        fig.colorbar(im_std_2, ax=axes[2, 2])

        fig.suptitle("Forward Problem Results", fontsize=18, fontweight='bold', y=1.02)
        fig.tight_layout()
        fig.savefig(self.fpath + f"resForward_{name}.png", dpi=300, bbox_inches='tight')

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def plotInverseResults(self, data_path='./results/data/resInverse.pt', name='inverse'):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        device = torch.device('cpu')
        resInv = torch.load(data_path, map_location=device, weights_only=False)

        X = torch.exp(resInv['Xm']).detach().cpu()
        xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

        xtrue = resInv['xTrue'].detach().to('cpu')
        xmax = xtrue.max()
        xmin = xtrue.min()
        xmean_raw = resInv['xmean'].detach().to('cpu')
        xmean_classified = (xmean_raw > (xmax - xmin) / 2 + xmin).float()
        xmean_classified = xmean_classified * (xmax - xmin) + xmin

        classification_mask = (xtrue == xmin) | (xtrue == xmax)
        classification_correct = ((xmean_classified == xtrue) & classification_mask).sum()
        classification_total = classification_mask.sum()
        classification_accuracy = (classification_correct / classification_total).item()

        ytrue = resInv['yTrue'].detach().cpu()
        ymean = resInv['ymean'].detach().cpu()
        rel_l2_error = torch.norm(ytrue - ymean) / torch.norm(ytrue)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        im0 = axes[0, 0].pcolormesh(xtrue.t(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0, 0].set_title(r"Ground Truth $\mathbf{x}$")
        fig.colorbar(im0, ax=axes[0, 0])

        im4 = axes[0, 1].pcolormesh(xx, yy, X, cmap='viridis', vmin=0., vmax=1.)
        axes[0, 1].set_title(r"Mean of Inferred $\mathbf{X}$")
        fig.colorbar(im4, ax=axes[0, 1])

        im1 = axes[0, 2].pcolormesh(resInv['y_obs'].t().cpu(), cmap='coolwarm')
        axes[0, 2].set_title(r"Observed $\mathbf{y}$ (Fully Observed, No Noise)")
        fig.colorbar(im1, ax=axes[0, 2])

        if 'obs_indices' in resInv:
            obs_indices = resInv['obs_indices'].detach().cpu()
            obs_x = obs_indices[1]
            obs_y = obs_indices[0]
            xx_obs, yy_obs = torch.meshgrid(obs_x, obs_y, indexing='ij')
            axes[0, 2].plot(xx_obs.flatten(), yy_obs.flatten(), 'k*', markersize=8)

        im2 = axes[1, 0].pcolormesh(xmean_raw.t(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[1, 0].set_title(r"Mean of Inferred $\mathbf{x}$")
        fig.colorbar(im2, ax=axes[1, 0])
        axes[1, 0].text(0.5, -0.15, f'Acc: {classification_accuracy:.3f}',
                    transform=axes[1, 0].transAxes, ha='center', va='top', fontsize=10)

        im3 = axes[1, 1].pcolormesh(ymean.t(), cmap='coolwarm')
        axes[1, 1].set_title(r"Mean of Inferred $\mathbf{y}$")
        fig.colorbar(im3, ax=axes[1, 1])

        abs_err = torch.abs(ytrue - ymean).t()
        im_err = axes[1, 2].pcolormesh(abs_err, cmap='inferno')
        axes[1, 2].set_title(r"Absolute Error: $|\mathbf{y}_{True} - \mathbf{y}|$")
        fig.colorbar(im_err, ax=axes[1, 2])
        axes[1, 2].text(0.5, -0.15, f'Rel L2: {rel_l2_error.item():.3e}',
                    transform=axes[1, 2].transAxes, ha='center', va='top', fontsize=10)

        xstd = resInv['xstd'].t().cpu()
        ystd = resInv['ystd'].t().cpu()

        im_std_0 = axes[2, 0].pcolormesh(xstd, cmap='jet')
        axes[2, 0].set_title(r"Std. Dev. of Inferred $\mathbf{x}$")
        fig.colorbar(im_std_0, ax=axes[2, 0])

        im_std_1 = axes[2, 1].pcolormesh(xx, yy, X, cmap='viridis', vmin=0., vmax=1.)
        axes[2, 1].set_title(r"Ground-Truth $\mathbf{X}$")
        fig.colorbar(im_std_1, ax=axes[2, 1])

        im_std_2 = axes[2, 2].pcolormesh(ystd, cmap='plasma')
        axes[2, 2].set_title(r"Std. Dev. of Inferred $\mathbf{y}$")
        fig.colorbar(im_std_2, ax=axes[2, 2])

        fig.suptitle("Inverse Problem Results", fontsize=18, fontweight='bold', y=1.02)
        fig.tight_layout()
        fig.savefig(self.fpath + f"resInverse_{name}.png", dpi=300, bbox_inches='tight')

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotForwardPaper(self, data_path='./results/data/resForward.pt', name='forward'):
        """Single-row forward-problem plot: x_obs, y_true, y_mean, |error|, y_std."""
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        device = torch.device('cpu')
        resFor = torch.load(data_path, map_location=device, weights_only=False)

        xmin = resFor['x_obs'].detach().cpu().min()
        xmax = resFor['x_obs'].detach().cpu().max()

        ytrue = resFor['yTrue'].detach().cpu()
        ymean = resFor['ymean'].detach().cpu()
        rel_l2_error = torch.norm(ytrue - ymean) / torch.norm(ytrue)

        EnvMetric = self.calcUncertaintyMetric(
            resFor['ymean'].detach().cpu(),
            meanRef=resFor['yTrue'].detach().cpu(),
            stdSol=resFor['ystd'].detach().cpu(),
            sIndex=2,
        )
        EnvelopScore = torch.mean(torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.)), dim=(0, 1))

        ystd = resFor['ystd'].t().detach().cpu()
        abs_err = torch.abs(ytrue - ymean).t()

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        im0 = axes[0].pcolormesh(resFor['x_obs'].t().detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0].set_title(r"Observed $\mathbf{x}$")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(resFor['yTrue'].t().detach().cpu(), cmap='coolwarm')
        axes[1].set_title(r"Ground Truth $\mathbf{y}$")
        fig.colorbar(im1, ax=axes[1])

        im3 = axes[2].pcolormesh(ymean.t(), cmap='coolwarm')
        axes[2].set_title(r"Mean of Inferred $\mathbf{y}$")
        fig.colorbar(im3, ax=axes[2])

        im4 = axes[3].pcolormesh(abs_err, cmap='inferno')
        axes[3].set_title(r"$|\mathbf{y}_{True} - \mathbf{y}|$")
        axes[3].text(0.5, -0.12, f'Rel L2: {rel_l2_error.item():.3e}',
                     transform=axes[3].transAxes, ha='center', va='top', fontsize=10)
        fig.colorbar(im4, ax=axes[3])

        im6 = axes[4].pcolormesh(ystd, cmap='plasma')
        axes[4].set_title(r"Std. Dev. of $\mathbf{y}$" + f' (ES={EnvelopScore.item():.3f})')
        fig.colorbar(im6, ax=axes[4])

        fig.suptitle("Forward Problem Results", fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(self.fpath + f"resForward_{name}.png", dpi=300, bbox_inches='tight')

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close(fig)

    def plotInversePaper(self, data_path='./results/data/resInverse.pt', name='inverse'):
        """Single-row inverse-problem plot: x_true, y_obs, x_mean, x_std."""
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        device = torch.device('cpu')
        resInv = torch.load(data_path, map_location=device, weights_only=False)

        xtrue = resInv['xTrue'].detach().to('cpu')
        xmax = xtrue.max()
        xmin = xtrue.min()
        xmean_raw = resInv['xmean'].detach().to('cpu')
        xmean_classified = (xmean_raw > (xmax - xmin) / 2 + xmin).float()
        xmean_classified = xmean_classified * (xmax - xmin) + xmin

        classification_mask = (xtrue == xmin) | (xtrue == xmax)
        classification_correct = ((xmean_classified == xtrue) & classification_mask).sum()
        classification_total = classification_mask.sum()
        classification_accuracy = (classification_correct / classification_total).item()

        xstd = resInv['xstd'].t().cpu()

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        im0 = axes[0].pcolormesh(xtrue.t(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0].set_title(r"Ground Truth $\mathbf{x}$")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(resInv['y_obs'].t().cpu(), cmap='coolwarm')
        axes[1].set_title(r"Observed $\mathbf{y}$")
        fig.colorbar(im1, ax=axes[1])
        if 'obs_indices' in resInv:
            obs_idx = resInv['obs_indices'].detach().cpu().long()
            W = resInv['y_obs'].cpu().shape[-1]
            row_idx = (obs_idx // W).float()
            col_idx = (obs_idx  % W).float()
            axes[1].plot(row_idx, col_idx, 'k*', markersize=5)

        im2 = axes[2].pcolormesh(xmean_raw.t(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[2].set_title(r"Mean of Inferred $\mathbf{x}$")
        axes[2].text(0.5, -0.12, f'Acc: {classification_accuracy:.3f}',
                     transform=axes[2].transAxes, ha='center', va='top', fontsize=10)
        fig.colorbar(im2, ax=axes[2])

        im5 = axes[3].pcolormesh(xstd, cmap='jet')
        axes[3].set_title(r"Std. Dev. of $\mathbf{x}$")
        fig.colorbar(im5, ax=axes[3])

        fig.suptitle("Inverse Problem Results", fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(self.fpath + f"resInverse_{name}.png", dpi=300, bbox_inches='tight')

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close(fig)

    def plotGeneratedSamples(self):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        X = self.data['generated_X'].squeeze(1)[:4].detach().cpu()
        x = self.data['generated_x'].squeeze(1)[:4].detach().cpu()
        Du = self.data['Dux'].squeeze(1)[:4].detach().cpu()
        generated_y = self.data['generated_y_mean'].squeeze(1)[:4].detach().cpu()
        Duy = self.data['Duy'].squeeze(1)[:4].detach().cpu()
        sgrid = self.data['intGrid'].detach().cpu()
        z = self.data['generated_z'].squeeze(1).detach().cpu()

        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim), indexing='ij')
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)), indexing='ij')

        for i in range(0, 1):
            fig = plt.figure(figsize=(22, 25))
            gs = GridSpec(nrows=5, ncols=6, figure=fig, width_ratios=[0.2, 1, 1, 1, 1, 0.05], hspace=0.4, wspace=0.2)

            fig.suptitle('Indicative instances/samples of various fields after training the model.', fontsize=24, fontweight='bold', y=0.9)

            xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

            # Reordered data_rows
            data_rows = [
                Du[i:i+4],          # new row 0 (was row 2)
                Duy[i:i+4],         # new row 1 (was row 4)
                x[i:i+4],           # new row 2 (was row 0)
                X[i:i+4],           # new row 3 (was row 1)
                generated_y[i:i+4]  # new row 4 (was row 3)
            ]

            row_labels = [
                '$\mathbf{x}$ from Dataset',            # new row 0
                '$\mathbf{y}$ from Dataset',         # new row 1
                'Generated $\mathbf{x}$', # new row 2
                'Generated $\mathbf{X}$',             # new row 3
                'Generated $\mathbf{y}$'           # new row 4
            ]

            yvmin = min(generated_y.min(), Duy.min())
            yvmax = max(generated_y.max(), Duy.max())

            for row_idx, row_data in enumerate(data_rows):
                vmin = row_data.min()
                vmax = row_data.max()

                # Choose colormap
                cmap = 'coolwarm' if row_idx in [1, 4] else 'viridis'

                # Add row label
                ax_label = fig.add_subplot(gs[row_idx, 0])
                ax_label.axis('off')
                ax_label.text(0.5, 0.5, row_labels[row_idx], ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)

                for col_idx in range(4):
                    ax = fig.add_subplot(gs[row_idx, col_idx + 1])
                    data = row_data[col_idx]

                    if row_idx in [0, 1, 2, 4]:  # Du, Generated Du, Generated x, Generated y use fine grid (sgrid)
                        grid_x, grid_y = sgrid[0], sgrid[1]
                    else:  # Coarse X uses xx, yy
                        grid_x, grid_y = xx, yy

                    if row_idx in [1, 4]:  # Shared color range for Generated Du and Generated y
                        im = ax.pcolormesh(grid_x, grid_y, data, cmap=cmap, vmin=yvmin, vmax=yvmax)
                    else:
                        im = ax.pcolormesh(grid_x, grid_y, data, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Add colorbar in the 6th column
                cax = fig.add_subplot(gs[row_idx, 5])
                fig.colorbar(im, cax=cax)

            plt.savefig(self.fpath + f"generatedSamples{i}.png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()
        torch.set_default_device(_prev_device)

    def plotVFLatentVariable(self):
        x = self.data['generated_x'].squeeze(1)
        z = self.data['generated_z'].squeeze(1)[..., 0]
        u = self.data['generated_u']
        uTrue = torch.sum(torch.where(x>0.9, 1., 0.).flatten(-2)/x.flatten(-2).size(-1), dim=-1)
        relDiff = (u-uTrue).abs()/uTrue
        meanRelDiff = relDiff.mean()
        sgrid = self.data['intGrid']

        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim), indexing='ij')
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)), indexing='ij')

        # Define bins based on z values (already 1D)
        z_min, z_max = z.min().item(), z.max().item()
        bin_edges = torch.linspace(z_min, z_max, 5)  # 4 intervals
        z_bins = [(bin_edges[i].item(), bin_edges[i+1].item()) for i in range(4)]

        # PLOTTING THE 4x4 GRID OF x INSTANCES
        fig, axs = plt.subplots(4, 4, figsize=(16, 14))
        fig.suptitle(
            "Indicative Generated Samples for different values of $z_1$, mean Relative Error " +
            f"= {meanRelDiff:.2f}", fontsize=16
        )

        for i, (low, high) in enumerate(z_bins):
            mask = (z >= low) & (z < high)
            idxs = mask.nonzero(as_tuple=True)[0]
            if len(idxs) >= 4:
                selected_idxs = random.sample(idxs.tolist(), 4)
            else:
                selected_idxs = random.choices(idxs.tolist(), k=4) if len(idxs) > 0 else [0]*4  # fallback

            for j in range(4):
                ax = axs[i, j]
                idx = selected_idxs[j]
                img = x[idx].cpu().numpy()
                u_val = u[idx].item()
                u_true_val = uTrue[idx].item()
                z_val = z[idx].item()
                rel_diff_val = relDiff[idx].item()

                pcm = ax.pcolormesh(img, cmap='viridis', shading='auto')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"$z_1$={z_val:.2f}, VF_true={u_true_val:.2f}\nVF_pred={u_val:.2f}, relDiff={rel_diff_val:.2%}",
                    fontsize=9
                )
                fig.colorbar(pcm, ax=ax)

                if j == 0:
                    ax.set_ylabel(f"$z_1$ ∈ [{low:.2f}, {high:.2f})", fontsize=12, labelpad=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

        plt.savefig(self.fpath + f"generatedxVF.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
        
        # SCATTER PLOT: z vs u and uTrue
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(z.cpu(), u.cpu(), label='VF_pred (u)', alpha=0.5, s=10, color='tab:blue')
        ax2.scatter(z.cpu(), uTrue.cpu(), label='VF_true (uTrue)', alpha=0.5, s=10, color='tab:orange')
        ax2.set_xlabel('$z_1$', fontsize=12)
        ax2.set_ylabel('VF', fontsize=12)
        ax2.set_title('Scatter Plot of $z_1$ vs Predicted and True VF', fontsize=14)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.fpath + "scatter_z_vs_VF.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()



    def plotGeneratedSamplesLogisticPCA(self):
        X = self.data['Dux'].squeeze(1)[:4]
        x = self.data['generated_x'].squeeze(1)[:4]
        sgrid = self.data['intGrid']

        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

        for i in range(0, 1):
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(nrows=2, ncols=5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05], hspace=0.4, wspace=0.2)

            data_rows = [
                x[i:i+4],  # row 0
                X[i:i+4],  # row 1
            ]

            for row_idx, row_data in enumerate(data_rows):
                vmin = row_data.min()
                vmax = row_data.max()
                cmap = 'viridis'

                for col_idx in range(4):
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    data = row_data[col_idx]

                    grid_x, grid_y = sgrid[0], sgrid[1] if row_idx == 0 else (xx, yy)

                    if row_idx == 0:
                        im = ax.pcolormesh(sgrid[0], sgrid[1], data, cmap=cmap, vmin=vmin, vmax=vmax)
                    else:
                        im = ax.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)

                    ax.set_xticks([])
                    ax.set_yticks([])

                # Add colorbar in the 5th column
                cax = fig.add_subplot(gs[row_idx, 4])
                fig.colorbar(im, cax=cax)
                cax.set_ylabel(f'Row {row_idx + 1}', fontsize=12)

            plt.savefig(self.fpath + f"generatedSamplesLPCA{i}.png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()



    def visualizeReconstruction(self):
        Wt = self.data['generated_x']  # shape: (100, 65, 65)
        #bt = self.data['bt']  # shape: (65, 65)

        #Wt = torch.cat((bt.unsqueeze(0), Wt), dim=0)  # prepend bt → shape: (101, 65, 65)

        # Select the first 64 images
        Wt = Wt[:64]

        # Meshgrid for plotting
        xx, yy = torch.meshgrid(torch.linspace(0, 1, 65), torch.linspace(0, 1, 65), indexing='ij')

        # Setup figure and axes
        fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(16, 16))
        vmin, vmax = Wt.min(), Wt.max()

        for i in range(64):
            row, col = divmod(i, 8)
            ax = axs[row, col]
            im = ax.pcolormesh(xx, yy, Wt[i], cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
            ax.axis('off')
            ax.set_title(f"Wt_{i}", fontsize=8)

        # Optional: add one shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
        plt.savefig(self.fpath + "visualizePCAReconstruction.png", dpi=300)

        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def zsamplesChains(self):
        z_samples = self.data['z_samples']
        resCondition = torch.load('./results/data/resCondition.pt')

        plt.figure(figsize=(12, 6))

        # Plot the evolution of the first 5 dimensions
        for i in range(5):
            plt.plot(z_samples[:, i], label=f'z[{i}]')

        plt.title("Evolution of First 5 z Dimensions Over Samples")
        plt.xlabel("Sample Index")
        plt.ylabel("z Value")
        plt.legend()
        plt.grid(True)

        plt.savefig(self.fpath + "zchains.png", dpi=300)

        if True:
            zmeanA = resCondition['vfmeanActual']  # replace with your actual zmean
            zstdA = resCondition['vfstdActual']   # replace with your actual zstd
            zmeanP = resCondition['vfmeanPred']  # replace with your actual zmean
            zstdP = resCondition['vfstdPred']   # replace with your actual zstd
            target = resCondition['u_obs']

            # Convert to float for plotting
            meanA = zmeanA.item()
            stdA = zstdA.item()
            meanP = zmeanP.item()
            stdP = zstdP.item()
            target_val = target.item()

            # X range for the Gaussian curve
            xA = np.linspace(meanA - 4*stdA, meanA + 4*stdA, 500)
            yA = norm.pdf(xA, loc=meanA, scale=stdA)
            xP = np.linspace(meanP - 4*stdP, meanP + 4*stdP, 500)
            yP = norm.pdf(xP, loc=meanP, scale=stdP)

            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(xA, yA, label=f'VF of Generated X from Posterior')
            plt.plot(xP, yP, label=f'VF directly from the HMC samples of $z_1$')
            plt.axvline(x=target_val, color='red', linestyle='--', label='Target Volume Fraction')

            plt.title(r"Distribution of VF after conditioning $\hat{u}=$"+f"{target:.2f}")
            plt.xlabel("Value")
            plt.ylabel("Probability Density")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(self.fpath + "vfTargetVsPred.png", dpi=300)

        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def visualizePCA(self):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        Wt = self.data['W'].detach().cpu()  # shape: (100, 65, 65)
        bt = self.data['b'].detach().cpu()  # shape: (65, 65)
        sgrid = self.data['intGrid'].detach().cpu()

        #Wt = torch.cat((bt.unsqueeze(0), Wt), dim=0)  # prepend bt → shape: (101, 65, 65)

        # Select the first 64 images
        #Wt = Wt[:10]
        Wt = Wt
        totalPlots = 16
        rows = 4

        # Meshgrid for plotting
        xx, yy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)), indexing='ij')

        # Setup figure and axes
        fig, axs = plt.subplots(nrows=rows, ncols=rows, figsize=(16, 16))
        vmin, vmax = Wt.min(), Wt.max()

        for i in range(totalPlots):
            row, col = divmod(i, rows)
            ax = axs[row, col]
            im = ax.pcolormesh(xx, yy, Wt[i], cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
            ax.axis('off')
            ax.set_title(f"Wt_{i}", fontsize=8)

        # Optional: add one shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
        plt.savefig(self.fpath + "visualizePCA.png", dpi=300)

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotLatentx(self, N=10, cmap='viridis'):
        """
        Plots an N x N grid of 2D field slices (e.g., 65x65 images) from a tensor.

        Args:
            x_tensor (torch.Tensor): Tensor of shape (N*N, H, W) to plot.
            N (int): Grid size (e.g., 10 means 10x10 grid).
            cmap (str): Colormap to use for plotting.
        """
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        x = self.data['x_interp'].squeeze(1)
        x_np = x.detach().cpu().numpy()  # Convert to NumPy
        Nx = 5
        Ny = 10
        fig, axes = plt.subplots(Nx, Ny, figsize=(20, 10))

        for i in range(Nx):
            for j in range(Ny):
                idx = i * Ny + j
                ax = axes[i, j]
                ax.imshow(x_np[idx], cmap=cmap)
                ax.axis('off')

        # Add a global title
        plt.suptitle("Latent Space Interpolations", fontsize=16)

        # Save and optionally show
        plt.savefig(self.fpath + "x_interp.png", dpi=300)

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def plotLatentxy(self, N=10, cmap='viridis'):
        """
        Plots an N x N grid of 2D field slices (e.g., 65x65 images) from a tensor.

        Args:
            x_tensor (torch.Tensor): Tensor of shape (N*N, H, W) to plot.
            N (int): Grid size (e.g., 10 means 10x10 grid).
            cmap (str): Colormap to use for plotting.
        """
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        zInt = torch.load('./zInt.pt', map_location='cpu', weights_only=False)
        x = zInt['x'].detach().cpu()
        y = zInt['y'].detach().cpu()
        X = torch.exp(zInt['X'].detach().cpu())

        x_np = x.numpy()
        y_np = y.numpy()
        X_np = X.numpy()

        Nx = 3  # Now 3 rows: x, y, X
        Ny = N  # Columns

        fig, axes = plt.subplots(Nx, Ny, figsize=(2 * Ny, 6), constrained_layout=True)

        # --- Row 0: x ---
        vmin_x = x_np.min()
        vmax_x = x_np.max()
        for j in range(Ny):
            ax = axes[0, j]
            pcm_x = ax.pcolormesh(x_np[j].T, cmap=cmap, shading='auto', vmin=vmin_x, vmax=vmax_x)
            ax.axis('off')
        cbar_x = fig.colorbar(pcm_x, ax=axes[0, :], orientation='vertical', fraction=0.02, pad=0.01)
        cbar_x.set_label('x field value', fontsize=12)

        # --- Row 1: y ---
        vmin_y = y_np.min()
        vmax_y = y_np.max()
        for j in range(Ny):
            ax = axes[1, j]
            pcm_y = ax.pcolormesh(y_np[j].T, cmap='coolwarm', shading='auto', vmin=vmin_y, vmax=vmax_y)
            ax.axis('off')
        cbar_y = fig.colorbar(pcm_y, ax=axes[1, :], orientation='vertical', fraction=0.02, pad=0.01)
        cbar_y.set_label('y field value', fontsize=12)

        # --- Row 2: X ---
        vmin_X = x_np.min()  # Use same color scale as x
        vmax_X = x_np.max()
        for j in range(Ny):
            ax = axes[2, j]
            pcm_X = ax.pcolormesh(X_np[j].T, cmap=cmap, shading='auto', vmin=vmin_X, vmax=vmax_X)
            ax.axis('off')
        cbar_X = fig.colorbar(pcm_X, ax=axes[2, :], orientation='vertical', fraction=0.02, pad=0.01)
        cbar_X.set_label('X field value', fontsize=12)

        # Title
        #plt.suptitle("Latent Space Interpolations:\nRow 1 = x, Row 2 = y, Row 3 = X", fontsize=16)

        # Save and display
        plt.savefig(self.fpath + "xy_interp.png", dpi=300)
        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()




    def plotxXpairs(self):
        dataDriven = False
        if not dataDriven:
            solsamp, solFenicssamp, condsamp, xsamp, yCoeffsamp = self.readTestSample()
            solsamp = solsamp.detach().cpu()
            solFenicssamp = solFenicssamp.detach().cpu()
            condsamp = condsamp.detach().cpu()
            xsamp = xsamp.detach().cpu()
            yCoeffsamp = yCoeffsamp.detach().cpu()

        rbfGrid = self.data['rbfGrid']
        #yTrue = self.data['yTest']
        sgrid = self.data['intGrid']
        rbfGridSize = rbfGrid.size(dim=1)**2
        X = self.data['XCG']
        Y = self.data['YCG']
        y = self.data['yCG']
        x = self.data['xCG']
        YFenics = self.data['YCGFenics']
        yTrueProj = self.data['yProjT']
        yTrue = solFenicssamp.clone().detach().cpu()
        
        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim), indexing='ij')
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)), indexing='ij')
        
        
        for i in range(0, 1):
            ymin=torch.min(yTrue[i, :, :])
            if torch.max(yTrue[i, :, :]) > torch.max(y[i, :, :]):
                ymax=torch.max(yTrue[i, :, :])
            else:
                ymax=torch.max(y[i, :, :])
            xmin=torch.min(torch.tensor(-3.))
            xmax=torch.max(torch.tensor(0.))
            combined_tensor = torch.stack((yTrue, y, yTrueProj))
            ymax = torch.max(combined_tensor[:, i, :, :])
            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))


            xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

            
            # plot x
            im1 = axs[0, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 0].set_title('log_10(x)')

            # plot X
            im2 = axs[0, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i+1, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 1].set_title('log_10(X)')

            im6 = axs[0, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i+2, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 2].set_title('log10(xTrue)')

            im8 = axs[0, 3].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i+3, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 3].set_title('log10(xTrue)')

            # plot Y
            im3 = axs[1, 0].pcolormesh(xx, yy, torch.log10(X[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 0].set_title('Y')

            # plot y
            im4 = axs[1, 1].pcolormesh(xx, yy, torch.log10(X[i+1, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 1].set_title('y')
            # plot y
            im7 = axs[1, 2].pcolormesh(xx, yy, torch.log10(X[i+2, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 2].set_title('yTrue Projection')

            im9 = axs[1, 3].pcolormesh(xx, yy, torch.log10(X[i+3, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 3].set_title('yTrue')

            # add colorbars
            cbar1 = fig.colorbar(im1, ax=[axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]])
            #cbar2 = fig.colorbar(im4, ax=[])
            
            cbar1.ax.set_position([0.78, 0.15, 0.03, 0.6])
            cbar1.set_label('$log_{10}(x)$',  fontsize=16)
            #cbar2.ax.set_position([0.8, 0.15, 0.03, 0.3])

            plt.savefig(self.fpath + "xXpairs"+str(i)+".png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()
        tess = 't'



    def gpExpansionExponentialParallel(self, x, gpEigVals, gpEigVecs, sgrid, fraction=0.):

        out = torch.einsum('i,...i,ij->...j', torch.sqrt(gpEigVals),
                                          x, gpEigVecs)
        out = torch.reshape(out, (*x.size()[:-1], sgrid.size(dim=1), sgrid.size(dim=1)))
        mask = out > fraction
        out = torch.where(mask, torch.tensor(0.1), torch.tensor(1.))
        #outNumpy = torch.reshape(out, [100, 32, 32]).cpu().numpy()
        return out


    def plotUncertaintySection(self, sIndex=2):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        sgrid = self.data['intGrid'].detach().cpu()
        sampSamplesMean = self.data['generated_y_mean'][:100].squeeze(1).detach().cpu()
        sampSamplesStd = self.data['generated_y_std'].squeeze(1).detach().cpu()
        yFENICSTrue = self.data['yFENICSTrue'].squeeze(1).detach().cpu()
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.clone()
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction

        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()

        yTrue = yFENICSTrue
        cond = self.data['generated_x'].squeeze(1)[:100].detach().cpu()

        RSquared = calcRSquared(yTrue, yPred)

        EnvMetric = []
        for i in range(0, yPred.size(dim=0)):
            EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                        stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
        EnvMetric = torch.stack(EnvMetric)
        EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
        EnvelopScore = torch.mean(EnvelopScore, dim=0)
        EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))

        columns = 5
        fig, axs = plt.subplots(columns, 2, figsize=(3 * columns, 3 * columns), num=self.plotCounter)
        c_x = cond

        for i in range(2):
            vmin = torch.min(yTrue[i, :, :])
            vmax = torch.max(yTrue[i, :, :])

            # Row 0: Conductivity
            mesh0 = axs[0, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                        c_x[i, :, :], cmap='jet', shading='auto')
            axs[0, i].set_title("Conductivity for Sample " + "{:.1f}".format(i))
            axs[0, i].set_aspect('equal')
            fig.colorbar(mesh0, ax=axs[0, i], orientation='horizontal', pad=0.05)

            # Row 1: Predicted Solution
            mesh1 = axs[1, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                        yPred[i, :, :], cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
            axs[1, i].set_title("Predicted Solution for Sample " + "{:.1f}".format(i))
            axs[1, i].set_aspect('equal')
            fig.colorbar(mesh1, ax=axs[1, i], orientation='horizontal', pad=0.05)

            # Row 2: True Solution
            mesh2 = axs[2, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                        yTrue[i, :, :], cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
            axs[2, i].set_title("True Solution for Sample " + "{:.1f}".format(i))
            axs[2, i].set_aspect('equal')
            fig.colorbar(mesh2, ax=axs[2, i], orientation='horizontal', pad=0.05)

            # Row 3: Envelope Metric
            mesh3 = axs[3, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                        EnvMetric[i, :, :], cmap='coolwarm', shading='auto', vmin=-1.0, vmax=1.0)
            axs[3, i].set_title("Envelope Metric " + "{:.1f}".format(i))
            axs[3, i].set_aspect('equal')
            fig.colorbar(mesh3, ax=axs[3, i], orientation='horizontal', pad=0.05)

            # Row 4: Uncertainty Section
            axs[4, i].plot(torch.linspace(0, 1.4142, yPred.size(-1)), torch.diag(yTrue[i, :, :]), 'r')
            axs[4, i].plot(torch.linspace(0, 1.4142, yPred.size(-1)), torch.diag(yPred[i, :, :]), '--b')
            axs[4, i].fill_between(
                torch.linspace(0, 1.4142, yPred.size(-1)),
                torch.diag(yPred[i, :, :]) + torch.diag(sIndex * sampSamplesStd[i, :, :]),
                torch.diag(yPred[i, :, :]) - torch.diag(sIndex * sampSamplesStd[i, :, :]),
                facecolor='blue', alpha=0.3
            )
            axs[4, i].grid(True)
            axs[4, i].set_title("Uncertainty Section " + "{:.1f}".format(i))

        # Compute and report relative errors
        relPANIS = calcEpsilon(yTrue=yTrue, yPred=yPred)

        # Load or generate PINO data
        yPINO = yTrue + torch.randn(yTrue.size())
        PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPINO_pureFNOVF50ttt.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")

        relPANIS = calcEpsilon(yTrue=yFENICSTrue, yPred=yPred)
        relPINO = calcEpsilon(yTrue=yFENICSTrue, yPred=yPINO)

        # Add super title and save
        fig.suptitle('For 100 different Samples $R^2$ = ' + f'{RSquared.item():.4f}' +
                    ' and EnvelopScore = ' + f'{EnvelopScore.item():.4f}' +
                    ' (Target = 0.95)' +
                    '\n relPANIS: ' + f'{relPANIS.item():.4f}' +
                    '\n relPINO: ' + f'{relPINO.item():.4f}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.fpath + "UncertaintySection.png", dpi=300, bbox_inches='tight')

        torch.set_default_device(_prev_device)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotMeanSampleAsCompetitors(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd']
        yFENICSTrue = self.data['yFENICSTrue'].detach().clone().cpu()
        RSquared = self.data['RSquared'][-1].clone().detach()
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
        RSquared = calcRSquared(yFENICSTrue, yPred)
            
        yPINO = yTrue + torch.randn(yTrue.size())
        PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPINO_pureFNOVF50ttt.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yFENICSTrue, yPINO)
        meanAbsErrorPINO = torch.mean(torch.abs(yPINO-yTrue))
        meanAbsErrorCG = torch.mean(torch.abs(yPred-yTrue))
        relativeL2ErrorPANIS = calcEpsilon(yTrue=yTrue, yPred=yPred)
        relativeL2ErrorPINO = calcEpsilon(yTrue=yTrue, yPred=yPINO)
        #print('PINO mean Absolute Error: '+f'{meanAbsErrorPINO.item():.3f}')
        #print('CG mean Absolute Error: '+f'{meanAbsErrorCG.item():.3f}')
        print('PINO relative L2 error: '+f'{relativeL2ErrorPINO.item():.5f}')
        print('PANIS relative L2 error: '+f'{relativeL2ErrorPANIS.item():.5f}')
        yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])

        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 4, figsize=(20, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):

            j = 30 + i
          
            

            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.



            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yFENICSTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            
            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPINO[j, :, :]-yFENICSTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)

            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanTrueFENICS = yFENICSTrue[j][yFENICSTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            #axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrueFENICS, 'c')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 3].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 3].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")

            axs[0, 1].set_title("Error FNO" + ' $R^2$=' +f'{RSquaredPINO.item():.3f}')
            axs[0, 2].set_title("Error PANIS" + ' $R^2$=' +f'{RSquared.item():.3f}')

            axs[0, 3].set_title('Solution Slice')

            axs[i, 3].legend(["Ground-truth", "FNO", "PANIS"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 2].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 3].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 5].collections[0], orientation='vertical')
            
            #fig.suptitle('Comparison on validation dataset with PINO')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        #fig.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(self.fpath + "comparingWithCompetitors.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def plotMeanSampleAsMyself(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd']
        RSquared = torch.tensor(self.data['RSquared'][-1])
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
            
        yTrue = self.data['yProjT']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        #PINOsavePath = "./PANISonl005mean.dat"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        
        

        RSquaredPINO = calcRSquared(yTrue, yPINO)
        yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        meanAbsErrorPINO = torch.mean(torch.abs(yPINO-yTrue))
        meanAbsErrorCG = torch.mean(torch.abs(yPred-yTrue))
        #print('CG without yF mean Absolute Error: '+f'{meanAbsErrorPINO.item():.3f}')
        #print('CG mean Absolute Error: '+f'{meanAbsErrorCG.item():.3f}')
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 4, figsize=(20, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):

            j =  51 + i 

            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.



            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            
            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPINO[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            #axs = fig.add_subplot(3, 4, i * 4 + 5)
            #axs[i, 4].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
            #                     EnvMetric[i, :, :], cmap='coolwarm', shading='auto', vmin=-1., vmax=1.)
            #axs = fig.add_subplot(3, 4, i * 4 + 6)
            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 3].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 3].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")

            axs[0, 2].set_title("Error with $\mathbf{y}_F$" + ' $R^2$=' +f'{RSquared.item():.3f}')
            axs[0, 1].set_title("Error without $\mathbf{y}_F$" + ' $R^2$=' +f'{RSquaredPINO.item():.3f}')

            axs[0, 3].set_title('Solution Slice')

            axs[i, 3].legend(["Ground-truth", "Our Model without $\mathbf{y}_F$", "Our Model with $\mathbf{y}_F$"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 2].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 3].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 5].collections[0], orientation='vertical')
            
            #fig.suptitle('Comparison on validation dataset with PINO')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        #fig.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(self.fpath + "comparingWithMyself.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
    
    def plotMeanSampleAsHMC(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean'][:2]
        sampSamplesStd = self.data['sampSamplesStd'][:2]
        yFENICSTrue = self.data['yFENICSTrue'][:2]
        RSquared = torch.tensor(self.data['RSquared'][-1])
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        yTrue = yTrue[:2]
        RSquared = calcRSquared(yTrue, yPred)
            
        #yTrue = self.data['yProjT']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        PINOsavePath = "./PANISonl005mean.dat"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
            yPINOstd = torch.load("./PANISonl005sigma.dat").detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")

        yPINO = self.data['analSolmean']
        yPINOstd = self.data['analSolstd']
        RSquaredPINO = calcRSquared(yTrue, yPINO)
        #yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        meanAbsErrorPINO = torch.mean(torch.abs(yPINO-yTrue))
        meanAbsErrorCG = torch.mean(torch.abs(yPred-yTrue))
        #print('CG without yF mean Absolute Error: '+f'{meanAbsErrorPINO.item():.3f}')
        #print('CG mean Absolute Error: '+f'{meanAbsErrorCG.item():.3f}')
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 4, figsize=(20, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):


            j =   i 



            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.
            vmax = torch.max(torch.abs(yPINO[j] - yTrue[j]))


            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            
            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPINO[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)

            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][:, yPred[j].size(0)//2]
            diagMean2 = yPINO[j][:, yPINO[j].size(0)//2]
            diagStd = sampSamplesStd[j][:, yPred[j].size(0)//2] * sIndex
            diagStd2 = yPINOstd[j][:, yPINO[j].size(0)//2] * sIndex
            diagMeanTrue = yTrue[j][:, yTrue[j].size(0)//2] #yTrue[j].flip(0).diagonal()
            diagMeanTrueFenics = yFENICSTrue[j][:, yFENICSTrue[j].size(0)//2] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][:, yPINO[j].size(0)//2] #yPINO[j].flip(0).diagonal()
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 3].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 3].fill_between(torch.linspace(0, 1., yPINO.size(-1)), diagMean2 + diagStd2,
                                                                                          diagMean2 - diagStd2,
                                                                                          facecolor='green', alpha=0.3)
            axs[i, 3].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")

            axs[0, 2].set_title("Error PANIS" + ' $R^2$=' +f'{RSquared.item():.3f}')
            axs[0, 1].set_title("Error HMC" + ' $R^2$=' +f'{RSquaredPINO.item():.3f}')

            axs[0, 3].set_title('Solution Slice')

            axs[i, 3].legend(["Full Ground-truth", "Closed-Form Gaussian", "PANIS"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 2].collections[0], orientation='vertical')


        plt.tight_layout()
        plt.savefig(self.fpath + "comparingWithHMC.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotMeanSampleAsPANIS(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd']
        yFENICSTrue = self.data['yFENICSTrue']
        RSquared = torch.tensor(self.data['RSquared'][-1])
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
            
        yTrue = self.data['yProjT']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        PINOsavePath = "./PANISonl005mean.dat"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
            yPINOstd = torch.load("./PANISonl005sigma.dat").detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yTrue, yPINO)
        #yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        meanAbsErrorPINO = torch.mean(torch.abs(yPINO-yTrue))
        meanAbsErrorCG = torch.mean(torch.abs(yPred-yTrue))
        #print('CG without yF mean Absolute Error: '+f'{meanAbsErrorPINO.item():.3f}')
        #print('CG mean Absolute Error: '+f'{meanAbsErrorCG.item():.3f}')
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 4, figsize=(20, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):


            j = 30 + i 



            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.
            vmax = torch.max(torch.abs(yPINO[j] - yTrue[j]))


            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            
            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPINO[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)

            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagMean2 = yPINO[j][yPINO[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagStd2 = yPINOstd[j][yPINO[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanTrueFenics = yFENICSTrue[j][yFENICSTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrueFenics, 'c')
            axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 3].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 3].fill_between(torch.linspace(0, 1., yPINO.size(-1)), diagMean2 + diagStd2,
                                                                                          diagMean2 - diagStd2,
                                                                                          facecolor='green', alpha=0.3)
            axs[i, 3].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")

            axs[0, 2].set_title("Error mPANIS" + ' $R^2$=' +f'{RSquared.item():.3f}')
            axs[0, 1].set_title("Error PANIS" + ' $R^2$=' +f'{RSquaredPINO.item():.3f}')

            axs[0, 3].set_title('Solution Slice')

            axs[i, 3].legend(["Full Ground-truth", "CG Ground-truth", "PANIS", "mPANIS"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 2].collections[0], orientation='vertical')


        plt.tight_layout()
        plt.savefig(self.fpath + "comparingWithPANIS.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotComp3Figs(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd'].clone().detach()
        #RSquared = self.data['RSquared'][-1].clone().detach()
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
            
        yTrue = self.data['yProjT']
        yTrueProj = yTrue
        yTrue = self.data['yFENICSTrue']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yTrue, yPINO)
        yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        meanAbsErrorPINO = torch.mean(torch.abs(yPINO-yTrue))
        meanAbsErrorCG = torch.mean(torch.abs(yPred-yTrue))
        #print('CG without yF mean Absolute Error: '+f'{meanAbsErrorPINO.item():.3f}')
        #print('CG mean Absolute Error: '+f'{meanAbsErrorCG.item():.3f}')
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 3, figsize=(16, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):

            #j =  73 + 8*i
            j = 0 + i

            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.



            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            

            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanTrueProj = yTrueProj[j][yTrueProj[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            axs[i, 2].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 2].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrueProj, '--g')
            #axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 2].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 2].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 2].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")
            axs[0, 1].set_title("Error PANIS" + ' $R^2$=' +f'{RSquared.item():.3f}')

            axs[0, 2].set_title('Solution Slice')

            axs[i, 2].legend(["Ground-truth", "CG-Projection" "PANIS"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')

            
            #fig.suptitle('Comparison on validation dataset with PINO')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        #fig.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(self.fpath + "comp3figs.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    

    def plotGreedySolutionEvolution(self, sIndex=2):
        sgrid = self.data['intGrid']
        yMean = self.data['yEvolwMean'].clone().detach()
        yStd = self.data['yEvolwStd'].clone().detach()
        RSquared = self.data['RSquared'][-1].clone().detach()
        intPoints = sgrid.size(dim=1)
        yPred = yMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
            
        yTrue = self.data['yProjT']
        yTrueProj = yTrue
        yTrue = self.data['yFENICSTrue']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yTrue, yPINO)
        yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        meanAbsErrorPINO = torch.mean(torch.abs(yPINO-yTrue))
        meanAbsErrorCG = torch.mean(torch.abs(yPred-yTrue))
        #print('CG without yF mean Absolute Error: '+f'{meanAbsErrorPINO.item():.3f}')
        #print('CG mean Absolute Error: '+f'{meanAbsErrorCG.item():.3f}')
        

        if True:
            EnvScore = []
            RSquared = []
            for k in range(0, yMean.size(0)):
                EnvMetric = []
                for i in range(0, yPred.size(dim=0)):
                    EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[k, i, :, :], meanRef=yTrue[i, :, :],
                                                                stdSol=yStd[k, i, :, :], sIndex=sIndex))
                EnvMetric = torch.stack(EnvMetric)
                EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
                EnvelopScore = torch.mean(EnvelopScore, dim=0)
                EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
                RSquared.append(calcRSquared(yTrue, yPred[k, :, :]))
                EnvScore.append(EnvelopScore)


        plt.rcParams.update({'font.size': 16})

        
        for i in range(2):

            columns = 1 + yMean.size(0)
            rows = 2
            fig, axs = plt.subplots(rows, columns, figsize=(5.3*(1+yMean.size(0)), 10), num=self.plotCounter)
            c_x = cond
            cax = []

            j = 0 + i

       
            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[0, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')
            
            axs[1, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 yTrue[j, :, :], cmap='coolwarm', shading='auto')
            
            fig.colorbar(axs[0, 0].collections[0], ax=axs[0, 0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[1, 0].collections[0], ax=axs[1, 0], orientation='vertical')

            for k in range(0, yMean.size(0)):
                axs[0, 1+k].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 yPred[k, j, :, :], cmap='coolwarm', shading='auto', vmin=torch.min(yTrue[j]), vmax=torch.max(yTrue[j]))
                
                
            


                diagMean = yPred[k, j][yTrue[j].size(0)//2, :]
                diagStd = yStd[k, j][yTrue[j].size(0)//2, :] * sIndex
                diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
                diagMeanTrueProj = yTrueProj[j][yTrueProj[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
                axs[1, 1+k].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
                #axs[1, 1+k].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrueProj, '--g')
                #axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
                axs[1, 1+k].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
                
                axs[1, 1+k].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                            diagMean - diagStd,
                                                                                            facecolor='blue', alpha=0.3)
                axs[1, 1+k].grid(True)
            
                #axs[1, 1+k].set_title("Input $\mathbf{x}$")
                #axs[1, 1+k].set_title("Error PANIS" + ' $R^2$=' +f'{RSquared[k].item():.3f}')

                #axs[1, 1+k].set_title('Solution Slice')
                axs[0, 1+k].set_title('Prediction for '+ str(k+1)+' residuals \n'+ ' $R^2$=' +f'{RSquared[k].item():.3f} \n' + ' $Enveloping Percentage: $=' +f'{EnvScore[k].item():.3f} \n')

                axs[1, 1+k].legend(["Ground-truth", "PANIS"])

                fig.colorbar(axs[0, 1+k].collections[0], orientation='vertical')
                #axs[1, 1+k].set_ylim(top=torch.max(diagMeanTrue + 3.))
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            

            
            #fig.suptitle('Comparison on validation dataset with PINO')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        #fig.subplots_adjust(bottom=0.35)
            plt.tight_layout()
            plt.savefig(self.fpath + "yEvolw"+str(i)+".png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()

    def diffSNR_compPINO_forward(self):
        test = 't'
        plt.rcParams.update({'font.size': 20})

        mode = 'exp1'
        pde = 'darcy'  # 'darcy' or 'helmholz'

        if mode == 'exp1':
            resInv = torch.load(f'experiments/diffSNR_compPINO/{pde}/g10000f.pt')
            resInvp = {
                'yPred': torch.load(f'experiments/diffSNR_compPINO/{pde}/p10000f.pt')[0]
            }
            resInvd = torch.load(f'experiments/diffSNR_compPINO/{pde}/d10000f.pt')

        xx, yy = torch.meshgrid(
            torch.linspace(0, 1, resInv['xstd'].size(-2)),
            torch.linspace(0, 1, resInv['xstd'].size(-1)),
            indexing='ij'
        )

        # Relative L2 errors
        rel_l2_error = calcEpsilon(
            resInv['yTrue'].detach().cpu(),
            resInv['ymean'].detach().cpu()
        )
        rel_l2_errorp = calcEpsilon(
            resInv['yTrue'].detach().cpu(),
            resInvp['yPred'].detach().cpu()
        )
        rel_l2_errord = calcEpsilon(
            resInv['yTrue'].detach().cpu(),
            resInvd['ymean'].detach().cpu()
        )

        # Shared std limits
        max_stdx = torch.max(torch.stack([
            resInvd['xstd'].max(),
            resInv['xstd'].max()
        ])).item()

        max_stdy = torch.max(torch.stack([
            resInvd['ystd'].max(),
            resInv['ystd'].max()
        ])).item()

        # Shared y color limits (means + truth)
        ymin = torch.min(torch.stack([
            torch.abs(resInv['yTrue'].detach().cpu() - resInvp['yPred'].detach().cpu()),
            torch.abs(resInv['yTrue'].detach().cpu() - resInvd['ymean'].detach().cpu()),
            torch.abs(resInv['yTrue'].detach().cpu() - resInv['ymean'].detach().cpu())
        ])).item()

        ymax = torch.max(torch.stack([
            torch.abs(resInv['yTrue'].detach().cpu() - resInvp['yPred'].detach().cpu()),
            torch.abs(resInv['yTrue'].detach().cpu() - resInvd['ymean'].detach().cpu()),
            torch.abs(resInv['yTrue'].detach().cpu() - resInv['ymean'].detach().cpu())
        ])).item()

        fig, axes = plt.subplots(1, 6, figsize=(30, 5))

        # Ground Truth
        im0 = axes[0].pcolormesh(
            xx, yy,
            resInv['yTrue'].detach().cpu(),
            cmap='coolwarm'
        )
        axes[0].set_title("Ground Truth")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(im0, ax=axes[0])

        # PINO
        im1 = axes[1].pcolormesh(
            xx, yy,
            torch.abs(resInv['yTrue'].detach().cpu() - resInvp['yPred'].detach().cpu()),
            cmap='plasma',
            vmin=ymin,
            vmax=ymax
        )
        axes[1].set_title(rf"Error, $ϵ={rel_l2_errorp:.1e}$")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(im1, ax=axes[1])

        # FunDPS Mean
        im2 = axes[2].pcolormesh(
            xx, yy,
            torch.abs(resInv['yTrue'].detach().cpu() - resInvd['ymean'].detach().cpu()),
            cmap='plasma',
            vmin=ymin,
            vmax=ymax
        )
        axes[2].set_title(rf"Mean error, $ϵ={rel_l2_errord:.1e}$")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        fig.colorbar(im2, ax=axes[2])

        # FunDPS Std
        im3 = axes[3].pcolormesh(
            xx, yy,
            resInvd['ystd'].cpu(),
            cmap='jet',
            vmin=0,
            vmax=max_stdy
        )
        axes[3].set_title("Standard Deviation")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        fig.colorbar(im3, ax=axes[3])

        # Ours Mean
        im4 = axes[4].pcolormesh(
            xx, yy,
            torch.abs(resInv['yTrue'].detach().cpu() - resInv['ymean'].detach().cpu()),
            cmap='plasma',
            vmin=ymin,
            vmax=ymax
        )
        axes[4].set_title(rf"Mean error, $ϵ={rel_l2_error:.1e}$")
        axes[4].set_xticks([])
        axes[4].set_yticks([])
        fig.colorbar(im4, ax=axes[4])

        # Ours Std
        im5 = axes[5].pcolormesh(
            xx, yy,
            resInv['ystd'].cpu(),
            cmap='jet',
            vmin=0,
            vmax=max_stdy
        )
        axes[5].set_title("Standard Deviation")
        axes[5].set_xticks([])
        axes[5].set_yticks([])
        fig.colorbar(im5, ax=axes[5])

        # Plot observed points if available
        if 'obs_indices' in resInv:
            obs_indices = resInv['obs_indices'].detach().cpu()
            obs_x = obs_indices[1]
            obs_y = obs_indices[0]
            xx_obs, yy_obs = torch.meshgrid(obs_x, obs_y, indexing='ij')
            axes[2].plot(xx_obs.flatten(), yy_obs.flatten(), 'k*', markersize=8)

        plt.tight_layout()

        # Vertical dashed separators
        fig_x = 0.515 * (axes[1].get_position().x1 + axes[2].get_position().x0)
        fig.lines.append(
            plt.Line2D([fig_x, fig_x], [0.05, 0.95],
                    transform=fig.transFigure,
                    color='k', linestyle='--', linewidth=2)
        )

        fig_x2 = 0.69 * axes[5].get_position().x1
        fig.lines.append(
            plt.Line2D([fig_x2, fig_x2], [0.05, 0.95],
                    transform=fig.transFigure,
                    color='k', linestyle='--', linewidth=2)
        )

        # Group labels
        y_top = axes[1].get_position().y1 + 0.12

        fig.text(
            (axes[0].get_position().x0 + axes[2].get_position().x1) / 2,
            y_top, 'PINO', ha='center', va='bottom',
            fontsize=20, fontweight='bold'
        )

        fig.text(
            (axes[2].get_position().x0 + axes[3].get_position().x1) / 2,
            y_top, 'FunDPS', ha='center', va='bottom',
            fontsize=20, fontweight='bold'
        )

        fig.text(
            (axes[4].get_position().x0 + axes[5].get_position().x1) / 2,
            y_top, 'GenPANIS', ha='center', va='bottom',
            fontsize=20, fontweight='bold'
        )

        plt.savefig(self.fpath + "diffSNR_compPINO.png", dpi=150, bbox_inches='tight')

        if self.displayPlots:
            plt.show()
        else:
            plt.close()



    def diffSNR_compPINO(self):
        test = 't'
        plt.rcParams.update({'font.size': 20})
        mode = 'exp1'
        pde = 'darcy' # 'darcy' or 'helmholz'
        if pde == 'darcy':
            xmax = 1.
            xmin = 0.
        else:
            xmax = 20.
            xmin = 0.
        if mode=='exp1':
            resInv = torch.load('experiments/diffSNR_compPINO/'+pde+'/Li.pt')
            resInvp = torch.load('experiments/diffSNR_compPINO/'+pde+'/pd20.pt')
            resInvd = torch.load('experiments/diffSNR_compPINO/'+pde+'/dsnr20.pt')
        elif mode=='exp4':
            resInv = torch.load('experiments/outDistPair/'+pde+'/snr100_outF.pt')
            resInvp = torch.load('experiments/outDistPair/'+pde+'/psnr100_outF.pt')
            resInvd = torch.load('experiments/outDistPair/'+pde+'/dsnr100_outF.pt')
        elif mode=='exp5':
            resInv = torch.load('experiments/outBc/'+pde+'/snr100_outBc.pt')
            resInvp = torch.load('experiments/outBc/'+pde+'/psnr100_outBc.pt')
            resInvd = torch.load('experiments/outBc/'+pde+'/dsnr100_outBc.pt')
        elif mode=='exp6':
            resInv = torch.load('experiments/outVf/'+pde+'/snr100_outVf.pt')
            resInvp = torch.load('experiments/outVf/'+pde+'/psnr100_outVf.pt')
            resInvd = torch.load('experiments/outVf/'+pde+'/dsnr100_outVf.pt')
        elif mode=='exp10':
            resInv = torch.load('experiments/diffLabeled/'+pde+'/snr100_data100.pt')
            resInvp = torch.load('experiments/diffLabeled/'+pde+'/psnr100_data100.pt')
            resInvd = torch.load('experiments/diffLabeled/'+pde+'/dsnr100_data100.pt')

        xx, yy = torch.meshgrid(torch.linspace(0, 1, resInv['xstd'].size(-2)),
                                torch.linspace(0, 1, resInv['xstd'].size(-1)),
                                indexing='ij')

        pixelAccuracy = calc_pixel_accuracy(resInv['xmean'].detach().cpu(), resInv['xTrue'].detach().cpu())
        pixelAccuracyp = calc_pixel_accuracy(resInvp['xPred'].detach().cpu(), resInv['xTrue'].detach().cpu())
        pixelAccuracyd = calc_pixel_accuracy(resInvd['xmean'].detach().cpu(), resInvd['xTrue'].detach().cpu())

        # Relative L2 error for y
        rel_l2_error = calcEpsilon(resInv['yTrue'].detach().cpu(), resInv['ymean'].detach().cpu())
        rel_l2_errorp = calcEpsilon(resInv['yTrue'].detach().cpu(), resInvp['yPred'].detach().cpu())
        rel_l2_errord = calcEpsilon(resInvd['yTrue'].detach().cpu(), resInvd['ymean'].detach().cpu())

        # Compute common color scale limits
        max_stdx = torch.max(torch.stack([
            resInvd['xstd'].max(), resInv['xstd'].max()
        ])).item()
        max_stdy = torch.max(torch.stack([
            resInvd['ystd'].max(), resInv['ystd'].max()
        ])).item()

        abs_err_d = torch.abs(resInvd['yTrue'] - resInvd['ymean']).cpu()
        abs_err_ours = torch.abs(resInv['yTrue'] - resInv['ymean']).cpu()
        max_abs_err = torch.max(torch.stack([abs_err_d.max(), abs_err_ours.max()])).item()


        fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # 1 row of 6 plots

        # Ground Truth
        im0 = axes[0].pcolormesh(xx, yy, resInv['xTrue'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0].set_title(r"Ground Truth")
        axes[0].set_xlabel('')
        axes[0].set_xticks([])  # remove x ticks
        axes[0].set_yticks([])  # remove y ticks
        fig.colorbar(im0, ax=axes[0])

        # PINO
        im2 = axes[1].pcolormesh(xx, yy, resInvp['xPred'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[1].set_title(r"Estimate, "+f'$PA= {pixelAccuracyp:.3f}$')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(im2, ax=axes[1])

        # FunDPS Mean
        im2 = axes[2].pcolormesh(xx, yy, resInvd['xmean'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[2].set_title(r"Mean, "+f'$PA= {pixelAccuracyd:.3f}$')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        fig.colorbar(im2, ax=axes[2])

        # FunDPS Std
        im_std_0 = axes[3].pcolormesh(xx, yy, resInvd['xstd'].cpu(), cmap='jet', vmin=0, vmax=max_stdx)
        axes[3].set_title(r"Standard Deviation")
        axes[3].set_xlabel('')
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        fig.colorbar(im_std_0, ax=axes[3])

        # Ours Mean
        im2 = axes[4].pcolormesh(xx, yy, resInv['xmean'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[4].set_title(r"Mean, "+f'$PA= {pixelAccuracy:.3f}$')
        axes[4].set_xticks([])
        axes[4].set_yticks([])
        fig.colorbar(im2, ax=axes[4])

        # Ours Std
        im_std_0 = axes[5].pcolormesh(xx, yy, resInv['xstd'].cpu(), cmap='jet', vmin=0, vmax=max_stdx)
        axes[5].set_title(r"Standard Deviation")
        axes[5].set_xlabel('')
        axes[5].set_xticks([])
        axes[5].set_yticks([])
        fig.colorbar(im_std_0, ax=axes[5])


        # Plot observed points if available (on FunDPS mean)
        if 'obs_indices' in resInv:
            obs_indices = resInv['obs_indices'].detach().cpu()
            obs_x = obs_indices[1]
            obs_y = obs_indices[0]
            xx_obs, yy_obs = torch.meshgrid(obs_x, obs_y, indexing='ij')
            axes[2].plot(xx_obs.flatten(), yy_obs.flatten(), 'k*', markersize=8)

        plt.tight_layout()
        # Add vertical dashed line between PINO and FunDPS
        fig_x = 0.52 * (axes[1].get_position().x1 + axes[2].get_position().x0)
        fig.lines.append(
            plt.Line2D(
                [fig_x, fig_x],
                [0.05, 0.95],
                transform=fig.transFigure,
                color='k',
                linestyle='--',
                linewidth=2
            )
        )
        # Vertical line between FunDPS (axes[2]) and Ours (axes[4])
        fig_x2 = 0.69*axes[5].get_position().x1
        fig.lines.append(
            plt.Line2D(
                [fig_x2, fig_x2],
                [0.05, 0.95],
                transform=fig.transFigure,
                color='k',
                linestyle='--',
                linewidth=2
            )
        )
        x0 = axes[2].get_position().x0
        x1 = axes[3].get_position().x1
        y_top = axes[1].get_position().y1 + 0.12  # slightly above the top of axes

        fig.text((x0 + x1)/2, y_top, 'FunDPS', ha='center', va='bottom', fontsize=20, fontweight='bold')
        x0 = axes[4].get_position().x0
        x1 = axes[5].get_position().x1
        y_top = axes[1].get_position().y1 + 0.12  # slightly above the top of axes

        fig.text((x0 + x1)/2, y_top, 'GenPANIS', ha='center', va='bottom', fontsize=20, fontweight='bold')
        x0 = axes[0].get_position().x0
        x1 = axes[2].get_position().x1
        y_top = axes[1].get_position().y1 + 0.12  # slightly above the top of axes

        fig.text((x0 + x1)/2, y_top, 'PINO', ha='center', va='bottom', fontsize=20, fontweight='bold')
        plt.savefig(self.fpath + "diffSNR_compPINO.png", dpi=150, bbox_inches='tight')

        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def partialObs_compPINO(self):
        test = 't'
        plt.rcParams.update({'font.size': 20})
        fixed = True
        pde = 'darcy' # 'darcy' or 'helmholz'
        if pde == 'darcy':
            xmax = 1.
            xmin = 0.
        else:
            xmax = 20.
            xmin = 0.
        if fixed==True:
            resInv = torch.load('experiments/partialObs_compPINO/'+pde+'/L11x11.pt')
            resInvp = torch.load('experiments/partialObs_compPINO/'+pde+'/p11x11.pt')
            resInvd = torch.load('experiments/partialObs_compPINO/'+pde+'/d11x11.pt')
        else:
            resInv = torch.load('experiments/randomObs/'+pde+'/Lr20.pt')
            resInvp = torch.load('experiments/randomObs/'+pde+'/pr20.pt')
            resInvd = torch.load('experiments/randomObs/'+pde+'/dr20.pt')

        xx, yy = torch.meshgrid(torch.linspace(0, 1, resInv['xstd'].size(-2)),
                                torch.linspace(0, 1, resInv['xstd'].size(-1)),
                                indexing='ij')

        pixelAccuracy = calc_pixel_accuracy(resInv['xmean'].detach().cpu(), resInv['xTrue'].detach().cpu())
        pixelAccuracyp = calc_pixel_accuracy(resInvp['xPred'].detach().cpu(), resInv['xTrue'].detach().cpu())
        pixelAccuracyd = calc_pixel_accuracy(resInvd['xmean'].detach().cpu(), resInvd['xTrue'].detach().cpu())

        # Relative L2 error for y
        rel_l2_error = calcEpsilon(resInv['yTrue'].detach().cpu(), resInv['ymean'].detach().cpu())
        rel_l2_errorp = calcEpsilon(resInv['yTrue'].detach().cpu(), resInvp['yPred'].detach().cpu())
        rel_l2_errord = calcEpsilon(resInvd['yTrue'].detach().cpu(), resInvd['ymean'].detach().cpu())

        # Compute common color scale limits
        max_stdx = torch.max(torch.stack([
            resInvd['xstd'].max(), resInv['xstd'].max()
        ])).item()
        max_stdy = torch.max(torch.stack([
            resInvd['ystd'].max(), resInv['ystd'].max()
        ])).item()

        abs_err_d = torch.abs(resInvd['yTrue'] - resInvd['ymean']).cpu()
        abs_err_ours = torch.abs(resInv['yTrue'] - resInv['ymean']).cpu()
        #abs_err_ours = torch.abs(resInv['yTrue'] - resInv['yFenics']).cpu()
        #abs_err_ours = resInv['yFenics'].cpu()
        max_abs_err = torch.max(torch.stack([abs_err_d.max(), abs_err_ours.max()])).item()

        fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # 1 row of 6 plots

        # Ground Truth
        im0 = axes[0].pcolormesh(xx, yy, resInv['xTrue'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[0].set_title(r"Ground Truth")
        axes[0].set_xlabel('')
        axes[0].set_xticks([])  # remove x ticks
        axes[0].set_yticks([])  # remove y ticks
        fig.colorbar(im0, ax=axes[0])

        # PINO
        im2 = axes[1].pcolormesh(xx, yy, resInvp['xPred'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[1].set_title(r"Estimate, "+f'$PA= {pixelAccuracyp:.3f}$')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(im2, ax=axes[1])

        # FunDPS Mean
        im2 = axes[2].pcolormesh(xx, yy, resInvd['xmean'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[2].set_title(r"Mean, "+f'$PA= {pixelAccuracyd:.3f}$')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        fig.colorbar(im2, ax=axes[2])

        # FunDPS Std
        im_std_0 = axes[3].pcolormesh(xx, yy, resInvd['xstd'].cpu(), cmap='jet', vmin=0, vmax=max_stdx)
        axes[3].set_title(r"Standard Deviation")
        axes[3].set_xlabel('')
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        fig.colorbar(im_std_0, ax=axes[3])

        # Ours Mean
        im2 = axes[4].pcolormesh(xx, yy, resInv['xmean'].detach().cpu(), cmap='viridis', vmin=xmin, vmax=xmax)
        axes[4].set_title(r"Mean, "+f'$PA= {pixelAccuracy:.3f}$')
        axes[4].set_xticks([])
        axes[4].set_yticks([])
        fig.colorbar(im2, ax=axes[4])

        # Ours Std
        im_std_0 = axes[5].pcolormesh(xx, yy, resInv['xstd'].cpu(), cmap='jet', vmin=0, vmax=max_stdx)
        axes[5].set_title(r"Standard Deviation")
        axes[5].set_xlabel('')
        axes[5].set_xticks([])
        axes[5].set_yticks([])
        fig.colorbar(im_std_0, ax=axes[5])

        """
        # Plot observed points if available (on FunDPS mean)
        if 'obs_indices' in resInv:
            obs_indices = resInv['obs_indices'].detach().cpu()
            obs_x = obs_indices[1]
            obs_y = obs_indices[0]
            xx_obs, yy_obs = torch.meshgrid(obs_x, obs_y, indexing='ij')
            axes[2].plot(xx_obs.flatten(), yy_obs.flatten(), 'k*', markersize=8)
        """

        # Plot observed points on Ground Truth input field
        if 'obs_indices' in resInv:
            nx, ny = resInv['xTrue'].shape[-2:]

            x_coords = torch.linspace(0, 1, nx)
            y_coords = torch.linspace(0, 1, ny)

            xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')

            obs_idx = resInv['obs_indices'].detach().cpu().long()

            axes[0].scatter(
                xx.flatten()[obs_idx],
                yy.flatten()[obs_idx],
                c='k',
                marker='*',
                s=80
            )

        plt.tight_layout()
        # Add vertical dashed line between PINO and FunDPS
        fig_x = 0.52 * (axes[1].get_position().x1 + axes[2].get_position().x0)
        fig.lines.append(
            plt.Line2D(
                [fig_x, fig_x],
                [0.05, 0.95],
                transform=fig.transFigure,
                color='k',
                linestyle='--',
                linewidth=2
            )
        )
        # Vertical line between FunDPS (axes[2]) and Ours (axes[4])
        fig_x2 = 0.69*axes[5].get_position().x1
        fig.lines.append(
            plt.Line2D(
                [fig_x2, fig_x2],
                [0.05, 0.95],
                transform=fig.transFigure,
                color='k',
                linestyle='--',
                linewidth=2
            )
        )
        x0 = axes[2].get_position().x0
        x1 = axes[3].get_position().x1
        y_top = axes[1].get_position().y1 + 0.12  # slightly above the top of axes

        fig.text((x0 + x1)/2, y_top, 'FunDPS', ha='center', va='bottom', fontsize=20, fontweight='bold')
        x0 = axes[4].get_position().x0
        x1 = axes[5].get_position().x1
        y_top = axes[1].get_position().y1 + 0.12  # slightly above the top of axes

        fig.text((x0 + x1)/2, y_top, 'GenPANIS', ha='center', va='bottom', fontsize=20, fontweight='bold')
        x0 = axes[0].get_position().x0
        x1 = axes[2].get_position().x1
        y_top = axes[1].get_position().y1 + 0.12  # slightly above the top of axes

        fig.text((x0 + x1)/2, y_top, 'PINO', ha='center', va='bottom', fontsize=20, fontweight='bold')
        plt.savefig(self.fpath + "partial_compPINO.png", dpi=300, bbox_inches='tight')



        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    
    def ploteffOfUn(self):
        test = 't'
        plt.rcParams.update({'font.size': 20})
        resInv = {
                k: v.clone().detach() if torch.is_tensor(v) else v
                for k, v in torch.load('./experiments/effOfUn/darcy/i20u_u10000.pt').items()
            }
        resInvp = {
                k: v.clone().detach() if torch.is_tensor(v) else v
                for k, v in torch.load('./experiments/effOfUn/darcy/i20u_u0.pt').items()
            }
        resInvd = {
                k: v.clone().detach() if torch.is_tensor(v) else v
                for k, v in torch.load('./experiments/effOfUn/darcy/i20u_u1000.pt').items()
            }

        xx, yy = torch.meshgrid(torch.linspace(0, 1, resInv['xstd'].size(-2)),
                                torch.linspace(0, 1, resInv['xstd'].size(-1)),
                                indexing='ij')

        pixelAccuracy = calc_pixel_accuracy(resInv['xmean'].detach().cpu(), resInv['xTrue'].detach().cpu())
        pixelAccuracyp = calc_pixel_accuracy(resInvp['xmean'].detach().cpu(), resInv['xTrue'].detach().cpu())
        pixelAccuracyd = calc_pixel_accuracy(resInvd['xmean'].detach().cpu(), resInvd['xTrue'].detach().cpu())

        # Relative L2 error for y
        rel_l2_error = calcEpsilon(resInv['yTrue'].detach().cpu(), resInv['ymean'].detach().cpu())
        rel_l2_errorp = calcEpsilon(resInv['yTrue'].detach().cpu(), resInvp['ymean'].detach().cpu())
        rel_l2_errord = calcEpsilon(resInvd['yTrue'].detach().cpu(), resInvd['ymean'].detach().cpu())

        rel2i = torch.stack((rel_l2_errorp, rel_l2_errord, rel_l2_error), dim=0)
        pxaci = torch.tensor([pixelAccuracyp, pixelAccuracyd, pixelAccuracy])
        # Compute common color scale limits
        max_stdx = torch.max(torch.stack([
            resInvd['xstd'].max(), resInv['xstd'].max(), resInvp['xstd'].max()
        ])).item()
        max_stdy = torch.max(torch.stack([
            resInvd['ystd'].max(), resInv['ystd'].max(), resInvp['ystd'].max()
        ])).item()

        abs_err_d = torch.abs(resInvd['yTrue'] - resInvd['ymean']).cpu()
        abs_err_ours = torch.abs(resInv['yTrue'] - resInv['ymean']).cpu()
        max_abs_err = torch.max(torch.stack([abs_err_d.max(), abs_err_ours.max()])).item()

        fig, axes = plt.subplots(1, 7, figsize=(35, 5))

        # ---- Remove all x/y ticks ----
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # ---- First row ----

        im0 = axes[0].pcolormesh(xx, yy, resInv['xTrue'].detach().cpu(),
                                cmap='viridis', vmin=0., vmax=1.)
        axes[0].set_title(r"Ground Truth")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(xx, yy, resInvp['xmean'].detach().cpu(),
                                cmap='viridis', vmin=0., vmax=1.)
        axes[1].set_title(r"Mean, " + f'$PA={pixelAccuracyp:.3f}$')
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].pcolormesh(xx, yy, resInvp['xstd'].detach().cpu(),
                                cmap='jet', vmin=0, vmax=max_stdx)
        axes[2].set_title(r"Standard Deviation")
        fig.colorbar(im2, ax=axes[2])

        im3 = axes[3].pcolormesh(xx, yy, resInvd['xmean'].detach().cpu(),
                                cmap='viridis', vmin=0., vmax=1.)
        axes[3].set_title(r"Mean, " + f'$PA={pixelAccuracyd:.3f}$')
        fig.colorbar(im3, ax=axes[3])

        im4 = axes[4].pcolormesh(xx, yy, resInvd['xstd'].cpu(),
                                cmap='jet', vmin=0, vmax=max_stdx)
        axes[4].set_title(r"Standard Deviation")
        fig.colorbar(im4, ax=axes[4])

        im5 = axes[5].pcolormesh(xx, yy, resInv['xmean'].detach().cpu(),
                                cmap='viridis', vmin=0., vmax=1.)
        axes[5].set_title(r"Mean, " + f'$PA={pixelAccuracy:.3f}$')
        fig.colorbar(im5, ax=axes[5])

        im6 = axes[6].pcolormesh(xx, yy, resInv['xstd'].cpu(),
                                cmap='jet', vmin=0, vmax=max_stdx)
        axes[6].set_title(r"Standard Deviation")
        fig.colorbar(im6, ax=axes[6])

        # ---- Observation markers (optional) ----
        if 'obs_indices' in resInv:
            obs_indices = resInv['obs_indices'].detach().cpu()
            obs_x = obs_indices[1]
            obs_y = obs_indices[0]
            xx_obs, yy_obs = torch.meshgrid(obs_x, obs_y, indexing='ij')
            axes[2].plot(xx_obs.flatten(), yy_obs.flatten(), 'k*', markersize=8)

        plt.tight_layout()

        # ================================
        # Vertical dashed separators
        # ================================
        # Between u0 and u1000
        x_sep1 = 0.55 * (axes[0].get_position().x1 + axes[1].get_position().x0)
        fig.lines.append(
            plt.Line2D([x_sep1, x_sep1], [0.05, 0.95],
                    transform=fig.transFigure,
                    linestyle='--', linewidth=2, color='k')
        )
        # Between u0 and u1000
        x_sep1 = 0.515 * (axes[2].get_position().x1 + axes[3].get_position().x0)
        fig.lines.append(
            plt.Line2D([x_sep1, x_sep1], [0.05, 0.95],
                    transform=fig.transFigure,
                    linestyle='--', linewidth=2, color='k')
        )

        # Between u1000 and u10000
        x_sep2 = 0.51 * (axes[4].get_position().x1 + axes[5].get_position().x0)
        fig.lines.append(
            plt.Line2D([x_sep2, x_sep2], [0.05, 0.95],
                    transform=fig.transFigure,
                    linestyle='--', linewidth=2, color='k')
        )

        # ================================
        # Group titles
        # ================================

        y_top = axes[0].get_position().y1 + 0.10

        # u0
        x0 = axes[1].get_position().x0
        x1 = axes[2].get_position().x1
        fig.text((x0 + x1) / 2, y_top, r'$N_u=0$',
                ha='center', va='bottom',
                fontsize=26, fontweight='bold')

        # u1000
        x0 = axes[3].get_position().x0
        x1 = axes[4].get_position().x1
        fig.text((x0 + x1) / 2, y_top, r'$N_u=1000$',
                ha='center', va='bottom',
                fontsize=26, fontweight='bold')

        # u10000
        x0 = axes[5].get_position().x0
        x1 = axes[6].get_position().x1
        fig.text((x0 + x1) / 2, y_top, r'$N_u=10000$',
                ha='center', va='bottom',
                fontsize=26, fontweight='bold')

        plt.savefig(self.fpath + "effOfUn_inverse.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        ###################################### FORWARD PROBLEM #######################################
        resInv = {
                k: v.clone().detach() if torch.is_tensor(v) else v
                for k, v in torch.load('./experiments/effOfUn/darcy/fu_u10000.pt').items()
            }
        resInvp = {
                k: v.clone().detach() if torch.is_tensor(v) else v
                for k, v in torch.load('./experiments/effOfUn/darcy/fu_u0.pt').items()
            }
        resInvd = {
                k: v.clone().detach() if torch.is_tensor(v) else v
                for k, v in torch.load('./experiments/effOfUn/darcy/fu_u1000.pt').items()
            }
        xx, yy = torch.meshgrid(torch.linspace(0, 1, resInv['xstd'].size(-2)),
                                torch.linspace(0, 1, resInv['xstd'].size(-1)),
                                indexing='ij')

        pixelAccuracy = calc_pixel_accuracy(resInv['xmean'].detach().cpu(), resInv['x_obs'].detach().cpu())
        pixelAccuracyp = calc_pixel_accuracy(resInvp['xmean'].detach().cpu(), resInv['x_obs'].detach().cpu())
        pixelAccuracyd = calc_pixel_accuracy(resInvd['xmean'].detach().cpu(), resInvd['x_obs'].detach().cpu())

        # Relative L2 error for y
        rel_l2_error = calcEpsilon(resInv['yTrue'].detach().cpu(), resInv['ymean'].detach().cpu())
        rel_l2_errorp = calcEpsilon(resInv['yTrue'].detach().cpu(), resInvp['ymean'].detach().cpu())
        rel_l2_errord = calcEpsilon(resInvd['yTrue'].detach().cpu(), resInvd['ymean'].detach().cpu())

        rel2f = torch.stack((rel_l2_errorp, rel_l2_errord, rel_l2_error), dim=0)
        pxacf = torch.tensor([pixelAccuracyp, pixelAccuracyd, pixelAccuracy])

        # Compute common color scale limits
        max_stdx = torch.max(torch.stack([
            resInvd['xstd'].max(), resInv['xstd'].max(), resInvp['xstd'].max()
        ])).item()
        max_stdy = torch.max(torch.stack([
            resInvd['ystd'].max(), resInv['ystd'].max(), resInvp['ystd'].max()
        ])).item()

        abs_err_d = torch.abs(resInvd['yTrue'] - resInvd['ymean']).cpu()
        abs_err_ours = torch.abs(resInv['yTrue'] - resInv['ymean']).cpu()
        max_abs_err = torch.max(torch.stack([abs_err_d.max(), abs_err_ours.max()])).item()

        fig, axes = plt.subplots(1, 7, figsize=(35, 5))

        # ---- Remove all x/y ticks ----
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # ================================
        # Single row (former second row)
        # ================================

        im0 = axes[0].pcolormesh(xx, yy, resInv['yTrue'].cpu(), cmap='coolwarm')
        axes[0].set_title(r"Ground Truth")
        fig.colorbar(im0, ax=axes[0])

        abs_err_pino = torch.abs(
            resInv['yTrue'].detach().cpu() - resInvp['ymean'].detach().cpu()
        )
        im1 = axes[1].pcolormesh(xx, yy, abs_err_pino, cmap='inferno')
        axes[1].set_title(
            r"Mean error, ϵ="
            + f'{rel_l2_errorp.item():.1e}'
        )
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].pcolormesh(
            xx, yy, resInvp['ystd'].detach().cpu(),
            cmap='jet', vmin=0, vmax=max_stdy
        )
        axes[2].set_title(r"Standard Deviation")
        fig.colorbar(im2, ax=axes[2])

        im3 = axes[3].pcolormesh(
            xx, yy, abs_err_d,
            cmap='inferno', vmin=0, vmax=max_abs_err
        )
        axes[3].set_title(
            r"Mean error, ϵ="
            + f'{rel_l2_errord.item():.1e}'
        )
        fig.colorbar(im3, ax=axes[3])

        im4 = axes[4].pcolormesh(
            xx, yy, resInvd['ystd'].cpu(),
            cmap='jet', vmin=0, vmax=max_stdy
        )
        axes[4].set_title(r"Standard Deviation")
        fig.colorbar(im4, ax=axes[4])

        im5 = axes[5].pcolormesh(
            xx, yy, abs_err_ours,
            cmap='inferno', vmin=0, vmax=max_abs_err
        )
        axes[5].set_title(
            r"Mean error, ϵ="
            + f'{rel_l2_error.item():.1e}'
        )
        fig.colorbar(im5, ax=axes[5])

        im6 = axes[6].pcolormesh(
            xx, yy, resInv['ystd'].cpu(),
            cmap='jet', vmin=0, vmax=max_stdy
        )
        axes[6].set_title(r"Standard Deviation")
        fig.colorbar(im6, ax=axes[6])

        plt.tight_layout()

        # ================================
        # Vertical dashed separators
        # ================================
        x_sep1 = 0.55 * (axes[0].get_position().x1 + axes[1].get_position().x0)
        fig.lines.append(
            plt.Line2D([x_sep1, x_sep1], [0.05, 0.95],
                    transform=fig.transFigure,
                    linestyle='--', linewidth=2, color='k')
        )
        # Between u0 and u1000
        x_sep1 = 0.515 * (axes[2].get_position().x1 + axes[3].get_position().x0)
        fig.lines.append(
            plt.Line2D([x_sep1, x_sep1], [0.05, 0.95],
                    transform=fig.transFigure,
                    linestyle='--', linewidth=2, color='k')
        )

        # Between u1000 and u10000
        x_sep2 = 0.51 * (axes[4].get_position().x1 + axes[5].get_position().x0)
        fig.lines.append(
            plt.Line2D([x_sep2, x_sep2], [0.05, 0.95],
                    transform=fig.transFigure,
                    linestyle='--', linewidth=2, color='k')
        )

        # ================================
        # Group titles
        # ================================
        y_top = axes[0].get_position().y1 + 0.10

        # u0
        x0 = axes[1].get_position().x0
        x1 = axes[2].get_position().x1
        fig.text((x0 + x1) / 2, y_top, r'$N_u=0$',
                ha='center', va='bottom',
                fontsize=26, fontweight='bold')

        # u1000
        x0 = axes[3].get_position().x0
        x1 = axes[4].get_position().x1
        fig.text((x0 + x1) / 2, y_top, r'$N_u=1000$',
                ha='center', va='bottom',
                fontsize=26, fontweight='bold')

        # u10000
        x0 = axes[5].get_position().x0
        x1 = axes[6].get_position().x1
        fig.text((x0 + x1) / 2, y_top, r'$N_u=10000$',
                ha='center', va='bottom',
                fontsize=26, fontweight='bold')

        plt.savefig(
            self.fpath + "effOfUn_forward.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        ###################################### FORWARD PROBLEM #######################################



        plt.rcParams.update({'font.size': 14})
        xu = torch.tensor([0, 1000, 10000], dtype=torch.int)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # =========================
        # Forward: L2 relative error
        # =========================
        color1 = 'tab:red'
        ax1.set_xlabel('$N_u$')
        ax1.set_ylabel('L2 Rel. Error (Forward Problem)', color=color1)
        ax1.plot(xu, rel2f, label='Forward L2 Rel. Error', color=color1, marker='o')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True)

        # =========================
        # Inverse: Pixel Accuracy
        # =========================
        ax2 = ax1.twinx()  # secondary y-axis
        color2 = 'tab:blue'
        ax2.set_ylabel('Pixel Accuracy (Inverse Problem)', color=color2)
        ax2.plot(xu, pxaci, label='Inverse Pixel Accuracy', color=color2, marker='o')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        """
        # Place legend inside plot at bottom center
        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc='lower center',         # inside axes
            bbox_to_anchor=(0.5, 0.05), # x=0.5 center, y=5% above bottom
            ncol=2,
            frameon=True,
            fontsize=12
        )
        """

        plt.tight_layout()
        plt.savefig(self.fpath + "error_plot_unlabeled.png", dpi=150, bbox_inches='tight')

        if self.displayPlots:
            plt.show()
        else:
            plt.close()



    def ploteffOfLab(self):
        _prev_device = torch.get_default_device()
        torch.set_default_device('cpu')
        test = 't'
        plt.rcParams.update({'font.size': 12})

        def detach_dict(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: detach_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [detach_dict(v) for v in obj]
            else:
                return obj
        # === Load inverse results (i) ===
        resInv100    = detach_dict(torch.load('./experiments/effOfLab/darcy/g100i.pt',    map_location='cpu', weights_only=False))
        resInv1000   = detach_dict(torch.load('./experiments/effOfLab/darcy/g1000i.pt',   map_location='cpu', weights_only=False))
        resInv10000  = detach_dict(torch.load('./experiments/effOfLab/darcy/g10000i.pt',  map_location='cpu', weights_only=False))

        resInv100p   = detach_dict(torch.load('./experiments/effOfLab/darcy/p100i.pt',    map_location='cpu', weights_only=False))
        resInv1000p  = detach_dict(torch.load('./experiments/effOfLab/darcy/p1000i.pt',   map_location='cpu', weights_only=False))
        resInv10000p = detach_dict(torch.load('./experiments/effOfLab/darcy/p10000i.pt',  map_location='cpu', weights_only=False))

        resInv100d   = detach_dict(torch.load('./experiments/effOfLab/darcy/d100i.pt',    map_location='cpu', weights_only=False))
        resInv1000d  = detach_dict(torch.load('./experiments/effOfLab/darcy/d1000i.pt',   map_location='cpu', weights_only=False))
        resInv10000d = detach_dict(torch.load('./experiments/effOfLab/darcy/d10000i.pt',  map_location='cpu', weights_only=False))

        # === Load forward results (f) ===
        resFor100    = detach_dict(torch.load('./experiments/effOfLab/darcy/g100f.pt',    map_location='cpu', weights_only=False))
        resFor1000   = detach_dict(torch.load('./experiments/effOfLab/darcy/g1000f.pt',   map_location='cpu', weights_only=False))
        resFor10000  = detach_dict(torch.load('./experiments/effOfLab/darcy/g10000f.pt',  map_location='cpu', weights_only=False))

        resFor100p   = detach_dict(torch.load('./experiments/effOfLab/darcy/p100f.pt',    map_location='cpu', weights_only=False))
        resFor1000p  = detach_dict(torch.load('./experiments/effOfLab/darcy/p1000f.pt',   map_location='cpu', weights_only=False))
        resFor10000p = detach_dict(torch.load('./experiments/effOfLab/darcy/p10000f.pt',  map_location='cpu', weights_only=False))

        resFor100d   = detach_dict(torch.load('./experiments/effOfLab/darcy/d100f.pt',    map_location='cpu', weights_only=False))
        resFor1000d  = detach_dict(torch.load('./experiments/effOfLab/darcy/d1000f.pt',   map_location='cpu', weights_only=False))
        resFor10000d = detach_dict(torch.load('./experiments/effOfLab/darcy/d10000f.pt',  map_location='cpu', weights_only=False))


        # === Pixel accuracy for inverse results ===
        pixelAccuracy100    = calc_pixel_accuracy(resInv100['xmean'].cpu(),    resInv100['xTrue'].cpu())
        pixelAccuracy1000   = calc_pixel_accuracy(resInv1000['xmean'].cpu(),   resInv1000['xTrue'].cpu())
        pixelAccuracy10000  = calc_pixel_accuracy(resInv10000['xmean'].cpu(),  resInv10000['xTrue'].cpu())

        pixelAccuracy100p   = calc_pixel_accuracy(resInv100p['xPred'].cpu(),   resInv100p['xTrue'].cpu())
        pixelAccuracy1000p  = calc_pixel_accuracy(resInv1000p['xPred'].cpu(),  resInv1000p['xTrue'].cpu())
        pixelAccuracy10000p = calc_pixel_accuracy(resInv10000p['xPred'].cpu(), resInv10000p['xTrue'].cpu())

        pixelAccuracy100d   = calc_pixel_accuracy(resInv100d['xmean'].cpu(),   resInv100d['xTrue'].cpu())
        pixelAccuracy1000d  = calc_pixel_accuracy(resInv1000d['xmean'].cpu(),  resInv1000d['xTrue'].cpu())
        pixelAccuracy10000d = calc_pixel_accuracy(resInv10000d['xmean'].cpu(), resInv10000d['xTrue'].cpu())


        # === Relative L2 errors for forward results (y) ===
        rel_l2_error100    = calcEpsilon(resFor100['yTrue'].cpu(),    resFor100['ymean'].cpu())
        rel_l2_error1000   = calcEpsilon(resFor1000['yTrue'].cpu(),   resFor1000['ymean'].cpu())
        rel_l2_error10000  = calcEpsilon(resFor10000['yTrue'].cpu(),  resFor10000['ymean'].cpu())

        rel_l2_error100p   = calcEpsilon(resFor100p[0].cpu(),   resFor100['yTrue'].cpu())
        rel_l2_error1000p  = calcEpsilon(resFor1000p[0].cpu(),  resFor1000['yTrue'].cpu())
        rel_l2_error10000p = calcEpsilon(resFor10000p[0].cpu(), resFor10000['yTrue'].cpu())

        rel_l2_error100d   = calcEpsilon(resFor100d['y_obs'].cpu(),   resFor100d['ymean'].cpu())
        rel_l2_error1000d  = calcEpsilon(resFor1000d['y_obs'].cpu(),  resFor1000d['ymean'].cpu())
        rel_l2_error10000d = calcEpsilon(resFor10000d['y_obs'].cpu(), resFor10000d['ymean'].cpu())


        # === Build vectors of results =================================================

        # Inverse vectors (g, p, d)
        inv_g = torch.tensor([
            pixelAccuracy100,
            pixelAccuracy1000,
            pixelAccuracy10000
        ]).cpu()

        inv_p = torch.tensor([
            pixelAccuracy100p,
            pixelAccuracy1000p,
            pixelAccuracy10000p
        ]).cpu()

        inv_d = torch.tensor([
            pixelAccuracy100d,
            pixelAccuracy1000d,
            pixelAccuracy10000d
        ]).cpu()

        # Forward vectors (g, p, d)
        for_g = torch.tensor([
            rel_l2_error100,
            rel_l2_error1000,
            rel_l2_error10000
        ]).cpu()

        for_p = torch.tensor([
            rel_l2_error100p,
            rel_l2_error1000p,
            rel_l2_error10000p
        ]).cpu()

        for_d = torch.tensor([
            rel_l2_error100d,
            rel_l2_error1000d,
            rel_l2_error10000d
        ]).cpu()

        x_vals = [100, 1000, 10000]

        # === Create figure with 2 subplots side-by-side ===
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # ---------------------- LEFT: Forward problem ----------------------
        axs[0].plot(x_vals, for_g,'b-o', label='GenPANIS')
        axs[0].plot(x_vals, for_p, 'g-o', label='PINO')
        axs[0].plot(x_vals, for_d, 'r-o', label='FunDPS')

        axs[0].set_xlabel("Number of labeled data")
        axs[0].set_ylabel("Relative L2 Error")
        axs[0].set_title("Forward Problem")

        axs[0].set_xscale("log")
        axs[0].set_yscale("log")   # <<< log scale on y-axis for forward
        axs[0].set_ylim(bottom=0.01)

        axs[0].legend()
        axs[0].grid(True, which='both')

        # ---------------------- RIGHT: Inverse problem ----------------------
        axs[1].plot(x_vals, inv_g, 'b-o', label='GenPANIS')
        axs[1].plot(x_vals, inv_p, 'g-o', label='PINO')
        axs[1].plot(x_vals, inv_d, 'r-o', label='FunDPS')

        axs[1].set_xlabel("Number of labeled data")
        axs[1].set_ylabel("Pixel Accuracy")
        axs[1].set_title("Inverse Problem")

        axs[1].set_xscale("log")   # inverse stays linear on y-axis
        axs[1].set_yscale("log")
        axs[1].set_ylim(top=1.0, bottom=0.5)

        axs[1].legend()
        axs[1].grid(True, which='both')

        # Improve layout
        plt.tight_layout()

        # Save figure
        plt.savefig('./results/figs/effOfLab.png', dpi=300)
        plt.close()
        torch.set_default_device(_prev_device)