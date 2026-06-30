import torch
import torch.nn as nn
import torch.nn as nn
from model.pde.numIntegrators.trapezoid import trapzInt2D, trapzInt2DParallel
import matplotlib.pyplot as plt


class xDecoder(nn.Module): # PANIS
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams, dimz):
        super(xDecoder, self).__init__()
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
        self.diagSigma = extraParams[4]

        
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 17x17 -> 17x17
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(8)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # 8x8 -> 8x8
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(16)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 4x4 -> 4x4
        self.act3 = nn.Softplus()
        self.bn3 = nn.BatchNorm2d(32)

        # Deconvolution (upsampling) layers
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)  # 2x2 -> 4x4
        self.act4 = nn.Softplus()
        self.bn4 = nn.BatchNorm2d(16)

        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1)  # 4x4 -> 8x8
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(8)

        self.deconv4 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)  # 8x8 -> 17x17

        self.sigmoid = nn.Sigmoid()

        #self.fc1 = nn.Linear(dimz, 4225)
        self.fc1 = nn.Linear(dimz, 16641)

        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        torch.nn.init.xavier_uniform_(self.deconv4.weight)

        # Print the number of parameters
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x

    def forwardold(self, x):

        # First layer block
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)

        # Second layer block
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)

        # Third layer block
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)

        # Deconvolution layers for upsampling
        x = self.deconv2(x)
        x = self.act4(x)
        x = self.bn4(x)

        x = self.deconv3(x)
        x = self.act5(x)
        x = self.bn5(x)

        x = self.deconv4(x)
        #x = torch.exp(x)
        x = self.sigmoid(x)

        return x
    
    def _get_likelihood(self, data, X):
        x = self.forward(X).flatten(-3)
        x = torch.clamp(x, min=10**(-6), max=(1-10**(-6)))
        data = torch.where(data > self.pde.phaseHigh, 1., 0.).flatten(-3)
        # Binomial Likelihood
        likelihood = torch.sum(data * torch.log(x) + (1-data)*torch.log(1-x)) #/torch.prod(torch.tensor(xbin.size()[1:]))  
        return likelihood


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
