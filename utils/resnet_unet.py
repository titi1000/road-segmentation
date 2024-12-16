import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class ResNetUnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        base_model = models.resnet34(pretrained=True)

        self.initial = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            base_model.bn1,
            base_model.relu
        )
        self.enc1 = base_model.layer1
        self.enc2 = base_model.layer2
        self.enc3 = base_model.layer3
        self.enc4 = base_model.layer4

        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = self.final_conv(d1)
        return torch.sigmoid(out)