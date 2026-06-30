import torch
import torch.nn as nn

class ztoXDecoder(nn.Module):
    def __init__(self, dimz):
        super(ztoXDecoder, self).__init__()
        
        
        #self.fc1 = nn.Linear(dimz-150, 129**2)
        self.fc1 = nn.Linear(dimz-0, 129**2)
        #self.fc1 = nn.Linear(dimz, 129**2)
        """
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(289)

        self.fc2 = nn.Linear(289, 289)
        #self.act2 = nn.Softplus()


        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(16)
        #self.conv1b = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)

        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act3 = nn.Softplus()
        self.bn3 = nn.BatchNorm2d(16)
 
        self.deconv2 = nn.ConvTranspose2d(24, 8, kernel_size=3, stride=1, padding=1)  # 32x32
        self.act4 = nn.Softplus()
        self.bn4 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)  # 32x32

        self.sigmoid = nn.Sigmoid()

        self.init_size = 4 # 17 // 4 ≈ 4
        self.fc = nn.Sequential(
            nn.Linear(200, 256 * 4 * 4),
            nn.ReLU()
        )

        # Decoder / generator-like structure to upscale to 17x17
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=1, padding=1),  # 16x16 -> 17x17
            #nn.Tanh()  # If image range is [-1, 1], you can also use Sigmoid if [0, 1]
        )
        """

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  #### k2s2 for reduceDim=17, k4s4 for reduceDim=9 for intGrid129
        #self.conv1b = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)

        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(48)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16
 
        self.deconv2 = nn.ConvTranspose2d(48, 16, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)  # 32x32

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        #self.fc1 = nn.Linear(289, 200)

        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        
        
        #numOfPars = self.count_parameters()
        #print("Number of NN Parameters: ", numOfPars)

    def forwardnew(self, x, mn, mx):
        out = self.fc(x)
        out = out.view(x.size(0), 256, self.init_size, self.init_size)
        img = self.decoder(out)
        img = self.sigmoid(img)*(mx-mn) + mn
        return img


    def forwardLinear(self, x, mn=-5., mx=0.): ### Simple Linear Layer

        x = self.fc1(x)  # [batch_size, 8*8*16]
        
        #x = self.act1(x)
        #x = self.bn1(x)

        #x = self.fc2(x)


        #x = x.view(-1, 1, 17, 17)  # Reshape into 3D for ConvTranspose
        #x = self.conv1(x)
        #x = self.act2(x)
        #x = self.bn2(x)
        #x = self.conv2(x)
        
        """
        x = self.act3(x)
        x = self.bn3(x)
        x = self.deconv2(x)
        x = self.act4(x)
        x = self.bn4(x)
        x = self.deconv3(x)
        """
        
        x = self.sigmoid(x)*(mx-mn) + mn

        return x  
    
    def forward(self, x, mn=-5., mx=0.): ### PANISLike
        # First layer block

        x = self.fc1(x)
        x = self.sigmoid(x)
        x = x.reshape(-1, 1, 129, 129)
        #x = x.reshape(-1, 1, 65, 65)
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

        x = self.sigmoid(x)*(mx-mn) + mn
        return x