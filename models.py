import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
   
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 100, 8, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 32, 8, stride=1, padding=0), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv1 = nn.Conv2d(1, 16, 3, padding=1) 
#         #self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
#         self.conv2=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
#         self.conv3=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv4=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
#         self.conv5=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv6=nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1)
#         self.conv7=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv8=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv9 = nn.Conv2d(16, 4, 3, padding=1)
#         #self.conv9=nn.Conv2d(in_channels=32, out_channels=100, kernel_size=8, stride=1, padding=1),

#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)         
#         #self.t_conv1=nn.ConvTranspose2d(in_channels=100, out_channels=32, kernel_size=8, stride=2, padding=1)
#         self.t_conv2=nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1)
#         self.t_conv3=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.t_conv4=nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)
#         self.t_conv5=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.t_conv6=nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
#         self.t_conv7=nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.t_conv8=nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=1, padding=1)
#         self.t_conv9 = nn.ConvTranspose2d(16, 1, 2, stride=2)
#         #self.t_conv9=nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=1, padding=1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv9(x))
#         x = self.pool(x)

#         x = F.relu(self.t_conv1(x))
#         x = F.sigmoid(self.t_conv9(x))
#         return x