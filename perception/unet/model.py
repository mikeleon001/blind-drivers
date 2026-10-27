import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetSegmentation(nn.Module):
    """
    u-net semantic lane segmentation, so in theory, pixels should be classified wihtout needing a dedicated cnn backbone for it
    """
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #padding=1, border of 0s, [B, 1, 28, 28] -> [B, 16, 28, 28]
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), #changes from [B, 16, 28, 28] to [B, 16, 28, 28]? refines existing feature maps
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #[B, 16, 28, 28] -> [B, 16, 14, 14], same number of feature maps, different sized feature maps (//2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), #[B, 16, 14, 14] -> [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), #[B, 32, 14, 14] -> [B, 32, 14, 14] #refining existing feature maps
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #[B, 32, 14, 14] -> [B, 32, 7, 7] #further downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), #[B, 32, 7, 7] -> [B, 64, 7, 7]
            nn.BatchNorm2d(64), #[B, 64, 7, 7] -> [B, 64, 7, 7], normalizes via subtraction of mean and division by sqrt of variance, adjusts data range to ensure stability
            nn.ReLU() 
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), #[B,64, 7, 7] -> [B, 64, 14, 14] 
            nn.Conv2d(64, 32, kernel_size=3, padding=1) #[B, 64, 14, 14] -> [B, 32, 14, 14]
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), #[B 64, 14, 14] -> [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), #[B 32 14 14]
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), #[B, 32, 28, 28]
            nn.Conv2d(32, 16, kernel_size=3, padding=1) #[B, 32, 28, 28] -> [B, 16, 28, 28]
        )
        
        #dec2 starts at 32 because we concat up2 with enc1 which is [B, 32, 28 28]
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1), #[B 32, 28, 28] -> [B, 16, 28, 28]
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), #[B 16 28 28] -> [B 16 28 28]
            nn.ReLU()
        )

        self.out = nn.Conv2d(16, 2, kernel_size=1) #[B, 16, 28, 28] -> [B, 1, 28, 28] (should be 128, 1, 28, 28 in theory)
        #output in theory should be Bx1xHxWx2 (class count, lane or not lane?)
        
    def forward(self, x):
        """
        forward takes in batched, potentially noised, data of size [B, 1, 28, 28] atleast for MNIST, eventually patches from VALIS output etc

        Input:
            xt: noisy image batch [B, 1, 28, 28]
            t: timesteps [B]
        
        returns: [B, 1, 28, 28] of predicted noise images passed through U-Net 
        """
        skip_connections = []

        x1 = self.enc1(x)
        skip_connections.append(x1)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        skip_connections.append(x2)
        x = self.pool2(x2)
        
        x = self.bottleneck(x)# + t_embed

        u1 = self.up1(x) #[B 32 14 14]
        u1 = torch.cat([u1, skip_connections[1]], dim=1) #[B 32 14 14] + [B 32 14 14] along dim=1, [B 64 14 14]
        u1 = self.dec1(u1) #[B 64, 14, 14] -> [B 32 14 14]

        u2 = self.up2(u1) 
        u2 = torch.cat([u2, skip_connections[0]], dim=1) #[B 16 28 28] + [B 16 28 28] = [B 32 28 28]
        u2 = self.dec2(u2) #[B 16 28 28]

        out = self.out(u2) #[B 16 28 28] -> [B 1 28 28]

        return out
    

