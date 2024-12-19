import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = Block(in_channels, out_channels, dropout_prob)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = Block(in_channels, out_channels, dropout_prob)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)

class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x)

class UNet(nn.Module):
    def __init__(self, BOTTLENECK, n_channels, n_class, dropout_prob=0.5):
        super().__init__()

        self.block = Block(n_channels, BOTTLENECK//16, dropout_prob)
        self.enc1 = Encoder(BOTTLENECK//16, BOTTLENECK//8, dropout_prob)
        self.enc2 = Encoder(BOTTLENECK//8, BOTTLENECK//4, dropout_prob)
        self.enc3 = Encoder(BOTTLENECK//4, BOTTLENECK//2, dropout_prob)
        self.enc4 = Encoder(BOTTLENECK//2, BOTTLENECK, dropout_prob)
        self.dec1 = Decoder(BOTTLENECK, BOTTLENECK//2, dropout_prob)
        self.dec2 = Decoder(BOTTLENECK//2, BOTTLENECK//4, dropout_prob)
        self.dec3 = Decoder(BOTTLENECK//4, BOTTLENECK//8, dropout_prob)
        self.dec4 = Decoder(BOTTLENECK//8, BOTTLENECK//16, dropout_prob)
        self.out = OutConvolution(BOTTLENECK//16, n_class)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        return self.out(x)
