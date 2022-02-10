import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    '''UNet encoder block'''
    def __init__(self,  in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(EncoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    '''UNet dncoder block'''
    def __init__(self,  in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()
        
        '''Bilinear Interpolate'''
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )
        '''TransposeConv'''
        # self.block = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        #     nn.InstanceNorm2d(out_channels),
        #     nn.ReLU()
        # )
        '''PixShuffle'''
        # self.block = nn.Sequential(
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        #     nn.InstanceNorm2d(out_channels),
        #     nn.ReLU()
        # )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    '''ResNet Generator Residual Block'''
    def __init__(self, in_channels, kernel_size=3):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels= in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=0),
            # nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels= in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=0),
            # nn.InstanceNorm2d(in_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return x + self.block(x)


class ConvBlock(nn.Module):
    '''PatchGAN Discriminator Convolutional Block'''
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    '''UNet Generator with correction in size'''
    def __init__(self, in_channels, out_channels, kernel_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.kernel_channels = kernel_channels
        self.out_channels = out_channels

        # head
        self.head = nn.Conv2d(in_channels=self.in_channels, out_channels=self.kernel_channels, kernel_size=3, stride=1, padding=1)

        # encoder
        self.down1 = EncoderBlock(1 * self.kernel_channels, 2 * self.kernel_channels) # 64 -> 128
        self.down2 = EncoderBlock(2 * self.kernel_channels, 4 * self.kernel_channels) # 128 -> 256
        self.down3 = EncoderBlock(4 * self.kernel_channels, 8 * self.kernel_channels) # 256 -> 512
        self.down4 = EncoderBlock(8 * self.kernel_channels, 8 * self.kernel_channels) # 512 -> 512

        # resblock
        self.resblock = [ResBlock(8 * self.kernel_channels) for _ in range(9)]
        self.resblock = nn.Sequential(*self.resblock) # 512 -> 512

        # decoder
        self.up1 = DecoderBlock(8 * self.kernel_channels, 8 * self.kernel_channels) # 512 -> 512
        self.up2 = DecoderBlock(16 * self.kernel_channels, 4 * self.kernel_channels) # 1024 -> 256
        self.up3 = DecoderBlock(8 * self.kernel_channels, 2 * self.kernel_channels) # 512 -> 128
        self.up4 = DecoderBlock(4 * self.kernel_channels, 1 * self.kernel_channels) # 256 -> 64

        # tail (1 x 1 conv)
        self.final = nn.Conv2d(in_channels=2 * self.kernel_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0) # 128 -> 3
        
        # dropout
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x_head = self.head(x)
        x1_down = self.down1(x_head)
        x2_down = self.down2(x1_down)
        x3_down = self.down3(x2_down)
        x4_down = self.down4(x3_down)

        x_transform = self.resblock(x4_down)

        x1_up = self.dropout1(self.up1(x_transform))
        x2_up = self.dropout2(self.up2(torch.cat([x3_down, x1_up], dim=1)))
        x3_up = self.up3(torch.cat([x2_down, x2_up], dim=1))
        x4_up = self.up4(torch.cat([x1_down, x3_up], dim=1))

        x_final = self.final(torch.cat([x_head, x4_up], dim=1))
        
        output = torch.tanh(x_final)
        output = output + x
        output = torch.clamp(output, min=-1, max=1)
        
        return output


class UNet_Pruning(nn.Module):
    '''UNet with pruning version'''
    def __init__(self, in_channels, out_channels, kernel_channels):
        super(UNet_Pruning, self).__init__()
        self.in_channels = in_channels
        self.kernel_channels = kernel_channels
        self.out_channels = out_channels

        # head
        self.head = nn.Conv2d(in_channels=self.in_channels, out_channels=self.kernel_channels, kernel_size=3, stride=1, padding=1)

        # encoder
        self.down1 = EncoderBlock(1 * self.kernel_channels, 1 * self.kernel_channels) # 32 -> 32
        self.down2 = EncoderBlock(1 * self.kernel_channels, 2 * self.kernel_channels) # 32 -> 64
        self.down3 = EncoderBlock(2 * self.kernel_channels, 4 * self.kernel_channels) # 64 -> 128
        self.down4 = EncoderBlock(4 * self.kernel_channels, 8 * self.kernel_channels) # 128 -> 256

        # resblock
        self.resblock = [ResBlock(8 * self.kernel_channels) for _ in range(9)]
        self.resblock = nn.Sequential(*self.resblock) # 256 -> 256

        # decoder
        self.up1 = DecoderBlock(8 * self.kernel_channels, 4 * self.kernel_channels) # 256 -> 128
        self.up2 = DecoderBlock(8 * self.kernel_channels, 2 * self.kernel_channels) # 256 -> 64
        self.up3 = DecoderBlock(4 * self.kernel_channels, 1 * self.kernel_channels) # 128 -> 32
        self.up4 = DecoderBlock(2 * self.kernel_channels, 1 * self.kernel_channels) # 64 -> 32

        # tail (1 x 1 conv)
        self.final = nn.Conv2d(in_channels=2 * self.kernel_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0) # 64 -> 3
        
        # dropout
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x_head = self.head(x)
        x1_down = self.down1(x_head)
        x2_down = self.down2(x1_down)
        x3_down = self.down3(x2_down)
        x4_down = self.down4(x3_down)

        x_transform = self.resblock(x4_down)

        x1_up = self.dropout1(self.up1(x_transform))
        x2_up = self.dropout2(self.up2(torch.cat([x3_down, x1_up], dim=1)))
        x3_up = self.up3(torch.cat([x2_down, x2_up], dim=1))
        x4_up = self.up4(torch.cat([x1_down, x3_up], dim=1))

        x_final = self.final(torch.cat([x_head, x4_up], dim=1))
        
        output = torch.tanh(x_final)
        output = output + x
        output = torch.clamp(output, min=-1, max=1)
        
        return output


class PatchGAN(nn.Module):
    '''PatchGAN Discrimator with 70 Receptive Field'''
    def __init__(self, in_channels, kernel_channels):
        super(PatchGAN, self).__init__()

        self.in_channels = in_channels
        self.kernel_channels = kernel_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.kernel_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = ConvBlock(self.kernel_channels, 2 * self.kernel_channels)
        self.conv3 = ConvBlock(2 * self.kernel_channels, 4 * self.kernel_channels)
        self.conv4 = ConvBlock(4 * self.kernel_channels, 8 * self.kernel_channels, kernel_size=4, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=8 * self.kernel_channels, out_channels=1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        return torch.sigmoid(x5)
