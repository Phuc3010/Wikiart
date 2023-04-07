import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect'):
        super(SNConv2d, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           bias=bias, padding_mode=padding_mode))
    
    def forward(self, x):
        return self.conv(x)
    
class SNConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect') -> None:
        super().__init__()
        self.conv = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                                     padding, bias=bias, padding_mode=padding_mode))
    
    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, kernel_size=3, stride=1, padding=1, down=True):
        super().__init__()
        self.conv = nn.Sequential(
            SNConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride,padding=padding
                       ) if down else 
            SNConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return x + self.block(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            SNConv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Critic(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Critic, self).__init__()
        self.initial = nn.Sequential(
            SNConv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        self.final = SNConv2d(in_channels+1, 1, kernel_size=4, stride=1, padding=1)
        self.model = nn.Sequential(*layers)

    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        x = self.minibatch_std(x)
        x = self.final(x)
        return x

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True):
        super(GenBlock, self).__init__()
        self.conv = nn.Sequential(
            SNConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True))
    
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(Generator, self).__init__()
        features = init_features
        self.pointwise = lambda in_channels, out_channels: nn.Sequential(
            SNConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                    bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.encoder1 = GenBlock(in_channels, features, use_norm=False)
        self.pool1 = nn.PixelUnshuffle(2)
        self.enc_point1 = self.pointwise(features*4, features*2)
        self.encoder2 = GenBlock(features*2, features * 2)
        self.pool2 = nn.PixelUnshuffle(2)
        self.enc_point2 = self.pointwise(features*8, features*4)
        self.encoder3 = GenBlock(features * 4, features * 4)
        self.pool3 = nn.PixelUnshuffle(2)
        self.enc_point3 = self.pointwise(features*16, features*8)
        self.encoder4 = GenBlock(features * 8, features * 8)
        self.pool4 = nn.PixelUnshuffle(2)
        self.enc_point4 = self.pointwise(features*32, features*16)
        
        
        self.bottleneck = GenBlock(features*16, features*16)
        self.residual = ResidualBlock(features*16)
        self.upconv4 = SNConvTranspose2d(
            features * 16, features * 8, kernel_size=3, stride=2  
        )
        self.upconv4 = nn.PixelShuffle(2)
        self.point4 = self.pointwise(features*4, features*8)
        self.decoder4 = GenBlock((features * 8) * 2, features * 8)
        self.upconv3 = SNConvTranspose2d(
            features * 8, features * 4, kernel_size=3, stride=2
        )
        self.upconv3 = nn.PixelShuffle(2)
        self.point3 = self.pointwise(features*2, features*4)
        self.decoder3 = GenBlock((features * 4) * 2, features * 4)
        self.upconv2 = SNConvTranspose2d(
            features * 4, features * 2, kernel_size=3, stride=2
        )
        self.upconv2 = nn.PixelShuffle(2)
        self.point2 = self.pointwise(features, features*2)
        self.decoder2 = GenBlock((features * 2) * 2, features * 2)
        self.upconv1 = SNConvTranspose2d(
            features * 2, features, kernel_size=3, stride=2
        )
        self.upconv1 = nn.PixelShuffle(2)
        self.point1 = self.pointwise(features//2, features)
        self.decoder1 = GenBlock(features * 2, features)

        self.conv = SNConv2d(
            in_channels=features, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.enc_point1(self.pool1(enc1)))
        enc3 = self.encoder3(self.enc_point2(self.pool2(enc2)))
        enc4 = self.encoder4(self.enc_point3(self.pool3(enc3)))
        residual_connect = self.residual(self.bottleneck(self.enc_point4(self.pool4(enc4))))

        dec4 = self.upconv4(residual_connect)
        dec4 = self.point4(dec4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = self.point3(dec3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = self.point2(dec2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.point1(dec1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        return torch.tanh(self.conv(dec1))