import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm_constant = (2 / (in_channels * kernel_size * kernel_size)) ** 0.5
        self.bias = self.conv.bias

        # Intializing the weights
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # Returning the output layer
        return self.conv(x * self.norm_constant)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return batch/torch.sqrt(torch.mean(batch**2, dim =1, keepdim= True) + 1e-8)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = DSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = DSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = PixelNorm()  # A single instance of both activation and normalization used as they do not have learnable parameters
        self.act = nn.LeakyReLU(0.2)
        self.block = nn.Sequential(
            self.conv1,
            self.act,
            self.norm,
            self.conv2,
            self.act,
            self.norm,
        )

    def forward(self, x):
        return self.block(x)


class ConvBlockDisc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = DSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = DSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = PixelNorm()  # A single instance of both activation and normalization used as they do not have learnable parameters
        self.act = nn.LeakyReLU(0.2)
        self.block = nn.Sequential(
            self.conv1,
            self.act,
            self.norm,
            self.conv2,
            self.act,
            self.norm,
        )

    def forward(self, x):
        return self.block(x)

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class Generator(nn.Module):
    def __init__(self, latent_size, in_channels):
        super().__init__()

        self.start = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(latent_size, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            DSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initialRGB = DSConv2d(in_channels, 3, 1, 1, 0)

        self.blocks = nn.ModuleList()
        self.RGB = nn.ModuleList([self.initialRGB])

        # Initializing the module list
        for i in range(len(factors) - 1):
            self.blocks.append(ConvBlock(int(in_channels * factors[i]), int(in_channels * factors[i + 1])))
            self.RGB.append(DSConv2d(int(in_channels * factors[i + 1]), 3, 1, 1, 0))

    def fade_in(self, prev_img, new_img, alpha):
        img = torch.tanh(prev_img * (1 - alpha) + new_img * (alpha))
        return img

    def forward(self, batch, alpha, steps):

        if steps == 0:
            return self.initialRGB(self.start(batch))
        else:
            out = self.start(batch)
            for step in range(steps):
                upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
                out = self.blocks[step](upscaled)

            upscaled_img = self.RGB[steps-1](upscaled)  # Check the steps or steps-1 once
            final_img = self.RGB[steps](out)

            return self.fade_in(upscaled_img, final_img, alpha)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Intializing the layer converting RGB to channels and disc blocks
        self.initial_rgb = DSConv2d(3, in_channels, 1, 1, 0)
        self.fromRGB = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(factors) - 1, 0, -1):
            self.blocks.append(ConvBlockDisc(int(in_channels * factors[i]), int(in_channels * factors[i - 1])))
            self.fromRGB.append(DSConv2d(3, int(in_channels * factors[i]), kernel_size=1, stride=1, padding=0))

        self.fromRGB.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        self.final = nn.Sequential(
            DSConv2d(in_channels + 1, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            DSConv2d(in_channels, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            DSConv2d(in_channels, 1, 1, 1, 0)
        )
        self.leaky = nn.LeakyReLU(0.2)

    def minibatch_std(self, batch):
        stats = torch.std(batch, dim=0).mean().repeat(batch.shape[0], 1, batch.shape[2], batch.shape[3])
        return torch.cat((batch, stats), dim=1)

    def fade_in(self, alpha, downscaled, out):
        return out * alpha + downscaled * (1 - alpha)

    def forward(self, batch, alpha, steps):
        if steps == 0:
            out_batch = self.minibatch_std(self.initial_rgb(batch))
            return self.final(out_batch).view(out_batch.shape[0], -1)

        step = len(self.blocks) - steps
        out = self.leaky(self.fromRGB[step](batch))

        downscaled = self.leaky(self.fromRGB[step+1](self.avg_pool(batch)))
        out = self.avg_pool(self.blocks[step](out))

        out = self.fade_in(alpha, downscaled, out)

        for step in range(step + 1, len(self.blocks)):
            out = self.blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final(out).view(out.shape[0], -1)
