import torch
import torch.nn as nn
import torch.nn as nn
from model.pde.numIntegrators.trapezoid import trapzInt2D, trapzInt2DParallel
import matplotlib.pyplot as plt


### Normal CNN
class nnmPANIS(nn.Module): # mPANIS
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams):
        super(nnmPANIS, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.


        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 32x32
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=4, stride=4)  # 16x16
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act3 = nn.Softplus()
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act4 = nn.Softplus()
        self.bn4 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)  # 32x32

        

        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.deconv1.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)

        
        
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)

    
   
    
    def xtoXCnn(self, x, printFlag=False):

        x = self.pde.gpExpansionExponentialParallel(x)
        x = torch.log(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.deconv1(x)
        x = self.act4(x)
        x = self.bn4(x)
        x = self.deconv2(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.deconv3(x)

        x = torch.exp(x)


        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        return x
        

    def forward(self, x):
        
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))


        x = self.Ytoy(x)    


        x = x.view(-1, 1, self.pde.NofShFuncs)

        return x
    
    def forwardMultiple(self, x, Navg):
        
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        x = x.repeat(1, Navg, 1).view(10*Navg, 1, -1)
        x = x.view(-1, self.reducedDim**2) +\
              (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                  (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2))) * self.yFMode
        

        x = self.Ytoy(x)    

        x = x.view(-1, Navg, self.pde.NofShFuncs)

        return x
    

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def Ytoy(self, x):
        x = torch.reshape(x, [x.size(0), -1, self.reducedDim, self.reducedDim])

        x = self.pde.shapeFunc.yCGtoYFG(x)

        

        return x
    
    
  
    def XtoY(self, x):
        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)
        return x
    

class nnPANISSigma(nn.Module): # PANIS
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams):
        super(nnPANISSigma, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.
        #self.Xfull = extraParams[3]

        
        # x -> X
        
        fb= 8
        ef1 = 3
        ef2 = 2
        fbs= 8
        ef1s = 2
        ef2s = 2
        

        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  #### k2s2 for reduceDim=17, k4s4 for reduceDim=9 for intGrid129
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(8, 24, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16
 
        self.deconv2 = nn.ConvTranspose2d(24, 8, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)  # 32x32

        self.sigmoid = nn.Sigmoid()

        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        
        
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)

    
    
   
    
    def xtoXCnn(self, x, printFlag=False):
        x = self.pde.gpExpansionExponentialParallel(x)


        
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.deconv2(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.deconv3(x)

        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        return x
        

    def forward(self, x):
        
        x = self.xtoXCnn(x)

 

        x = x.view(-1, 1, self.pde.NofShFuncs)

        return x
 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    

    





class nnPANIS(nn.Module): # PANIS
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams, dimz):
        super(nnPANIS, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.
        self.Xfull = extraParams[3]
        #self.diagSigma = extraParams[4]
        self.diagSigma = nn.Parameter(extraParams[4]) ### Activate this only if you train directly for sigmaDiag
        self.sigmaEncoder = nn.Parameter(extraParams[5])
        #self.sigmaEncoder = torch.tensor(-4.)
        #self.yWhole = nn.Parameter(extraParams[5])
        #self.yF = nn.Parameter(extraParams[5])
        #self.diagSigma = extraParams[4] ### Activate this if a NN is used to produce this tensor #Cov(X)
        #self.polyCoeff = nn.Parameter(torch.randn(3)) #Cond(u)
        #self.fcu1 = nn.Linear(50, 100) #Cond(u)
        #self.fcu2 = nn.Linear(100, 1) #Cond(u)
        
        # x -> X
        
        fb= 8
        ef1 = 3
        ef2 = 2
        fbs= 8
        ef1s = 2
        ef2s = 2
        

        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  #### k2s2 for reduceDim=17, k4s4 for reduceDim=9 for intGrid129
        #self.conv1b = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)

        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(8, 24, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16
 
        self.deconv2 = nn.ConvTranspose2d(24, 8, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)  # 32x32

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(17**2, dimz)
        self.fcxy = nn.Linear(17**2, dimz) ## This is irrelevant for the working version of GenPANIS


        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        
        
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)

    
    
    def xtoXCnn(self, x, printFlag=False, prior=False, KLE=True, mn=None, mx=None, y=None):
        if KLE:
            x = self.pde.gpExpansionExponentialParallel(x)

        #x = torch.nn.functional.interpolate(x, size=(65, 65), mode='bilinear', align_corners=True)
        

        if prior:
            x = torch.nn.functional.interpolate(x, size=(self.reducedDim, self.reducedDim), mode='bilinear', align_corners=True)
        else:
            x = torch.log(x)


            x = self.conv1(x)
            x = self.act1(x)
            x = self.bn1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.bn2(x)
            x = self.pool2(x)
            x = self.deconv2(x)
            x = self.act5(x)
            x = self.bn5(x)
            x = self.deconv3(x)

            #x = (torch.sigmoid(self.Xfull)*5-5.).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = (-torch.abs(self.Xfull)).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = (-torch.sqrt(torch.abs(self.Xfull))).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = - torch.abs(x)
            x = x.flatten(-2)

            if y is not None:
                y = self.conv1(y)
                y = self.act1(y)
                y = self.bn1(y)
                y = self.pool1(y)
                y = self.conv2(y)
                y = self.act2(y)
                y = self.bn2(y)
                y = self.pool2(y)
                y = self.deconv2(y)
                y = self.act5(y)
                y = self.bn5(y)
                y = self.deconv3(y)
                y = y.flatten(-2)
                x = self.fcxy(torch.cat((x, y), dim=-1))
            else:
                x = self.fc1(x)
        
        ### Testing basic probabilistic encoder
        x = x + torch.sqrt(torch.pow(10, self.sigmaEncoder)) * torch.randn_like(x)
        ### Testing basic probabilistic encoder


            
            


        #self.xbc = self.conv1b(x)
        #self.xbc = (self.xbc - torch.nn.functional.pad(self.xbc[..., 1:-1, 1:-1], pad=(1, 1, 1, 1), value=0.))

        #x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        #self.xbc = torch.reshape(self.xbc, [x.size(0), self.reducedDim, self.reducedDim])
        return x
    
    def logProbEncoder(self, z, x, printFlag=False, prior=False, KLE=True, mn=None, mx=None):
        z = z.unsqueeze(1)
        if KLE:
            x = self.pde.gpExpansionExponentialParallel(x)

        #x = torch.nn.functional.interpolate(x, size=(65, 65), mode='bilinear', align_corners=True)
        

        if prior:
            x = torch.nn.functional.interpolate(x, size=(self.reducedDim, self.reducedDim), mode='bilinear', align_corners=True)
        else:
            x = torch.log(x)


            x = self.conv1(x)
            x = self.act1(x)
            x = self.bn1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.bn2(x)
            x = self.pool2(x)
            x = self.deconv2(x)
            x = self.act5(x)
            x = self.bn5(x)
            x = self.deconv3(x)

            #x = (torch.sigmoid(self.Xfull)*5-5.).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = (-torch.abs(self.Xfull)).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = (-torch.sqrt(torch.abs(self.Xfull))).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = - torch.abs(x)
            x = x.flatten(-2)
            x = self.fc1(x)

            logProb = -x.size(-1)/2*(torch.log(2*torch.tensor(torch.pi)) + torch.log(torch.pow(10, self.sigmaEncoder))) - 0.5 * ((z - x).flatten(-2)**2).sum(-1)/torch.pow(10, self.sigmaEncoder)
        
        return logProb

    
    
    def xtoXCnnold(self, x, printFlag=False, prior=False, KLE=True, mn=None, mx=None):
        if KLE:
            x = self.pde.gpExpansionExponentialParallel(x)

        #x = torch.nn.functional.interpolate(x, size=(65, 65), mode='bilinear', align_corners=True)

        if prior:
            x = torch.nn.functional.interpolate(x, size=(self.reducedDim, self.reducedDim), mode='bilinear', align_corners=True)
        else:
            x = torch.log(x)


            x = self.conv1(x)
            x = self.act1(x)
            x = self.bn1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.bn2(x)
            x = self.pool2(x)
            x = self.deconv2(x)
            x = self.act5(x)
            x = self.bn5(x)
            x = self.deconv3(x)

            #x = (torch.sigmoid(self.Xfull)*5-5.).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = (-torch.abs(self.Xfull)).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = (-torch.sqrt(torch.abs(self.Xfull))).view(self.Xfull.size(0), 1, self.reducedDim, self.reducedDim)
            #x = - torch.abs(x)
            if mn is not None:
                #x = torch.clamp(x, min=mn, max=mx)
                x = self.sigmoid(x)*(mx-mn) + mn

            
            


        #self.xbc = self.conv1b(x)
        #self.xbc = (self.xbc - torch.nn.functional.pad(self.xbc[..., 1:-1, 1:-1], pad=(1, 1, 1, 1), value=0.))

        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        #self.xbc = torch.reshape(self.xbc, [x.size(0), self.reducedDim, self.reducedDim])
        return x
    
    def forward(self, x, permute=False, prior=False, give_X=False, KLE=False, mn=None, mx=None, y=None):
        
        x = self.xtoXCnn(x, KLE=KLE, mn=mn, mx=mx, y=y)
        X = x
        
        return x, X

    def forwardold(self, x, permute=False, prior=False, give_X=False, KLE=False, mn=None, mx=None):
        
        x = self.xtoXCnn(x, KLE=KLE, mn=mn, mx=mx)
        X = x
        x = torch.exp(x)

        

        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)

        #x = torch.nn.functional.pad(x[..., 1:-1, 1:-1], pad=(1, 1, 1, 1), value=0.) + self.xbc

        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))
        
        if permute:
            #sigma = torch.einsum('k,...k->...k', torch.pow(10, self.diagSigma/2), torch.randn(x.size(0), (self.reducedDim-2)**2))
            sigma = torch.pow(10, self.globalSigma) * torch.randn(x.size(0), (self.reducedDim-2)**2)
            #sigma += torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) 
            sigma = torch.nn.functional.pad(sigma.view(*sigma.size()[:-1], self.reducedDim - 2, self.reducedDim - 2), pad=(1, 1, 1, 1), value=0.)
            x = x + sigma
        
        ### Original PANIS
        x = self.Ytoy(x) 
        x = x.view(-1, 1, self.pde.NofShFuncs)
        x = self.pde.shapeFunc.cTrialSolutionParallel(x)
        ### Original PANIS
        ### Reduced PANIS ###
        #x = x[...,1:-1, 1:-1]
        #x = x.reshape(-1, (self.reducedDim-2)**2)
        ### Reduced PANIS ###

        ### PANIS + FNO ###
        
        #x = torch.nn.functional.interpolate(x.unsqueeze(1), size=(self.pde.sgrid.size(-1), self.pde.sgrid.size(-1)), mode='bilinear', align_corners=True).squeeze(1)
        if give_X:
            return X, x
        else:
            return x
    
    def forwardMultiple(self, x, permute=False, Navg=None):
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))
        """
        if permute:
            sigma = torch.einsum('k,...k->...k', torch.pow(10, self.diagSigma/2), torch.randn(x.size(0)*Navg, (self.reducedDim-2)**2))
            #sigma += torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0)*Navg, self.V.size(-1))) 
            sigma = torch.nn.functional.pad(sigma.view(*sigma.size()[:-1], self.reducedDim - 2, self.reducedDim - 2), pad=(1, 1, 1, 1), value=0.)
            x = x.view(x.size(0), 1, self.reducedDim, self.reducedDim).repeat(1, Navg, 1, 1) + sigma.view(x.size(0), Navg, self.reducedDim, self.reducedDim)
        else:
            x = x.view(x.size(0), 1, self.reducedDim, self.reducedDim).repeat(1, Navg, 1, 1)
        """

        x = self.Ytoy(x)    
        


        x = x.view(-1, Navg, self.pde.NofShFuncs)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def Ytoy(self, x):
        x = torch.reshape(x, [x.size(0), -1, self.reducedDim, self.reducedDim])

        #x = torch.nn.functional.interpolate(x, size=(self.pde.sgrid.size(-1), self.pde.sgrid.size(-1)), mode='bilinear', align_corners=True)

        #x = self.pde.shapeFunc.findRbfCoeffs(closedForm=True, y=x.squeeze(1), N=30000).unsqueeze(1)
        #t0 = torch.einsum('ij,...j->...i', torch.linalg.inv(self.pde.shapeFunc.aInvB.t() @ self.pde.shapeFunc.aInvB) @ self.pde.shapeFunc.aInvB.t(), x)
        #x = self.pde.shapeFunc.yCGtoYFG(t0.view(x.size(0), 1, self.reducedDim, self.reducedDim))
        x = self.pde.shapeFunc.yCGtoYFG(x)
        

        return x
    
  
    def Xtoy(self, x, rbfCoeff=False):
        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        x = torch.exp(x)

        

        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)

        #x = torch.nn.functional.pad(x[..., 1:-1, 1:-1], pad=(1, 1, 1, 1), value=0.) + self.xbc


        
        ### Original PANIS
        x = self.Ytoy(x) 
        x = x.view(-1, 1, self.pde.NofShFuncs)
        if not rbfCoeff:
            x = self.pde.shapeFunc.cTrialSolutionParallel(x)
        ### Original PANIS
        ### Reduced PANIS ###
        #x = x[...,1:-1, 1:-1]
        #x = x.reshape(-1, (self.reducedDim-2)**2)
        ### Reduced PANIS ###

        ### PANIS + FNO ###
        
        #x = torch.nn.functional.interpolate(x.unsqueeze(1), size=(self.pde.sgrid.size(-1), self.pde.sgrid.size(-1)), mode='bilinear', align_corners=True).squeeze(1)

        return x
    
    def XtodiagSigmaFCNN(self, x):
        x = x.flatten(-3)
        x = self.model_XtosigmaDiag(x)
        return x
    
    def center_crop(self, x, size):
        """Center crop a 4D tensor to a given spatial size (H, W)."""
        _, _, h, w = x.size()
        th, tw = size
        i = (h - th) // 2
        j = (w - tw) // 2
        return x[:, :, i:i+th, j:j+tw]
    
    def XtodiagSigma(self, x):
        x = x
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.center_crop(x, [65, 65]).flatten(-3)
        return x

    
    def forwardMultiple(self, x, permute=False, Navg=None):
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))
        """
        if permute:
            sigma = torch.einsum('k,...k->...k', torch.pow(10, self.diagSigma/2), torch.randn(x.size(0)*Navg, (self.reducedDim-2)**2))
            #sigma += torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0)*Navg, self.V.size(-1))) 
            sigma = torch.nn.functional.pad(sigma.view(*sigma.size()[:-1], self.reducedDim - 2, self.reducedDim - 2), pad=(1, 1, 1, 1), value=0.)
            x = x.view(x.size(0), 1, self.reducedDim, self.reducedDim).repeat(1, Navg, 1, 1) + sigma.view(x.size(0), Navg, self.reducedDim, self.reducedDim)
        else:
            x = x.view(x.size(0), 1, self.reducedDim, self.reducedDim).repeat(1, Navg, 1, 1)
        """

        x = self.Ytoy(x)    
        


        x = x.view(-1, Navg, self.pde.NofShFuncs)
        return x
    
    def uGivenzNN(self, x):
        x = self.fcu1(x)
        x = self.relu(x)
        x = self.fcu2(x)
        return x
    




class nnPANISLatentX(nn.Module): # PANIS with introduced latent X (Perturbations on X)
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams):
        super(nnPANISLatentX, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.

        
        # x -> X
        
        fb= 8
        ef1 = 3
        ef2 = 2
        fbs= 8
        ef1s = 2
        ef2s = 2
        

        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  #### k2s2 for reduceDim=17, k4s4 for reduceDim=9 for intGrid129
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(8, 24, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16
 
        self.deconv2 = nn.ConvTranspose2d(24, 8, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1)  # 32x32
        
        self.sigmoid = nn.Sigmoid()

        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        
        
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)

    
    
   
    
    def xtoXCnn(self, x, Navg=None, printFlag=False):
        x = self.pde.gpExpansionExponentialParallel(x)

        x = torch.log(x)

        
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.deconv2(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.deconv3(x)

        if Navg is not None:
            sigma_x = -5.5 + self.sigmoid(x[:, 1, :, :])*6
            xmean = x[:, 0, :, :].view(x.size(0), 1, x.size(-2), x.size(-1)).repeat(1, Navg, 1, 1)
            xsigma = sigma_x.view(x.size(0), 1, x.size(-2), x.size(-1)).repeat(1, Navg, 1, 1)
            x = (xmean + torch.pow(10, xsigma) * torch.randn_like(xmean)).view(x.size(0)*Navg, self.reducedDim, self.reducedDim)
        else:
            self.sigma_x = -5.5 + self.sigmoid(x[:, 1, :, :])*6
            self.xmean = x[:, 0, :, :]
            x = (self.xmean + torch.pow(10, self.sigma_x) * torch.randn_like(x[:, 1, :, :])).unsqueeze(1)
        #x = x + torch.pow(10, self.globalSigma/2.) * torch.randn_like(x)
        #x = x + torch.einsum('ij,j->i', torch.pow(10, self.V), torch.randn(self.V.size(-1))).repeat(x.size(0), 1).reshape(x.size(0), 1, self.reducedDim, self.reducedDim) \
        #      + torch.pow(10, self.globalSigma/2.) * torch.randn_like(x)
        x = torch.exp(x)



        

        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        return x
        

    def forward(self, x):
        
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))


        x = self.Ytoy(x)    

        x = x.view(-1, 1, self.pde.NofShFuncs)

        return x
    
    def forwardMultiple(self, x, Navg):
        
        with torch.no_grad():
         
            x = self.xtoXCnn(x, Navg=Navg)

            x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)


            ### For reduced covariance
            if self.yFMode:
                x = x.view(-1, self.reducedDim**2) +\
                    (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                        (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))


            x = self.Ytoy(x)    

            x = x.view(-1, Navg, self.pde.NofShFuncs)

        return x
 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def Ytoy(self, x):
        x = torch.reshape(x, [x.size(0), -1, self.reducedDim, self.reducedDim])


        x = self.pde.shapeFunc.yCGtoYFG(x)
        

        return x
    
  
    def XtoY(self, x):
        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)
        return x



class nnPANISwithLinear(nn.Module): # PANIS
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams):
        super(nnPANISwithLinear, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.

        
        # x -> X
        
        fb= 24
        ef1 = 3
        ef2 = 3
        fbs= 8
        ef1s = 2
        ef2s = 2
        

        
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(24)
        #self.pool1 = nn.AvgPool2d(kernel_size=2, stride=1)  #### k2s2 for reduceDim=17, k4s4 for reduceDim=9 for intGrid129
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(24, 72, kernel_size=5, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(72)
        #self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16
 
        self.deconv2 = nn.ConvTranspose2d(72, 24, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(24)
        self.deconv3 = nn.ConvTranspose2d(24, 1, kernel_size=3, stride=1, padding=1)  # 32x32
        self.act6 = nn.Softplus()
        self.bn6 = nn.BatchNorm2d(1)
        self.linear = nn.Linear(62**2, 33*33)

        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        
        
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)

    
    
   
    
    def xtoXCnn(self, x, printFlag=False):
        x = self.pde.gpExpansionExponentialParallel(x)

        x = torch.log(x)

        
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        #x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        #x = self.pool2(x)
        x = self.deconv2(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.deconv3(x)
        x = self.act6(x)
        #x = self.bn6(x)
        x = x.reshape(x.size(0), x.size(1), x.size(-1)**2)
        x = self.linear(x)
        

        #x = torch.nn.functional.interpolate(x, size=(33, 33), mode='bilinear', align_corners=True)

        x = torch.exp(x)



        

        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        return x
        

    def forward(self, x):
        
        x = self.xtoXCnn(x)


        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))


        x = self.Ytoy(x)    

        x = x.view(-1, 1, self.pde.NofShFuncs)

        return x
 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def Ytoy(self, x):
        x = torch.reshape(x, [x.size(0), -1, self.reducedDim, self.reducedDim])


        x = self.pde.shapeFunc.yCGtoYFG(x)
        

        return x
    
  
    def XtoY(self, x):
        x = self.pde.shapeFunc.solveCGPDE_dispatch(c_x=x, f=100., uBc=self.uBc)
        return x



class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=6, stride=2, padding=1)  # 32x32
        self.relu1 = torch.nn.Softplus()
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=6, stride=2, padding=1)  # 16x16
        self.relu2 = torch.nn.Softplus()
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=1)  # 8x8
        self.relu3 = torch.nn.Softplus()
        self.bn3 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        return x




# Define the decoder
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=1)  # 16x16
        self.relu1 = torch.nn.Softplus()
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=1)  # 32x32
        self.relu2 = torch.nn.Softplus()
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.deconv3 = torch.nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2, padding=1)  # 64x64
        self.relu3= torch.nn.Softplus()
        self.bn3 = torch.nn.BatchNorm1d(62**2)
        self.fc1 = torch.nn.Linear(62**2, 17**2)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.softplus2 = nn.Softplus()
        #self.fc1.bias.data.fill_(100)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.deconv3(x)

        ### New addition
        x = self.relu3(x)
        ###
        x = x.view(x.size(0), x.size(1), -1)  # This reshaping works only if x is 4D
        x = self.fc1(x)
        return x


# Combine the encoder and decoder into the autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.

    def forward(self, x):
        x = self.pde.gpExpansionExponentialParallel(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class FullyConnectedNN(nn.Module):
    def __init__(self, reducedDim=None, cgCnn=None, xtoXCnn=None, pde=None, extraParams=None, input_dim=512, hidden_dim1=100, hidden_dim2=100, hidden_dim3=100, hidden_dim4=100, output_dim=10**2):
        super(FullyConnectedNN, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.
        """
        # Layer 1: Input layer to Hidden layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.softplus1 = nn.SELU()
        """
        # Layer 2: Hidden layer 1 to Hidden layer 2
        self.fc2 = nn.Linear(input_dim, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.softplus2 = nn.Softplus()

        # Layer 3: Hidden layer 2 to Hidden layer 3
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.softplus3 = nn.Softplus()

        # Layer 4: Hidden layer 3 to Hidden layer 4
        self.fc1 = nn.Linear(hidden_dim3, hidden_dim4)
        self.bn1 = nn.BatchNorm1d(hidden_dim4)
        self.softplus4 = nn.Softplus()

        # Layer 5: Hidden layer 4 to Output layer
        self.fc5 = nn.Linear(hidden_dim4, output_dim)

        # Initialize layers
        # self.initialize_layer(self.fc1)
        self.initialize_layer(self.fc2)
        self.initialize_layer(self.fc3)
        self.initialize_layer(self.fc1)
        self.initialize_layer(self.fc5)

    def forward(self, x):
        x = x.view(x.size(0), x.size(-1))
        """
        x = self.fc1(x)
        x = self.softplus1(x)
        x = self.bn1(x)
        """
        x = self.fc2(x)
        x = self.softplus2(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.softplus3(x)
        x = self.bn3(x)

        x = self.fc1(x)
        x = self.softplus4(x)
        x = self.bn1(x)


        x = self.fc5(x)

        x = x.view(x.size(0), 1, x.size(-1))
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_layer(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)


