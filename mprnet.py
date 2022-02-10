import torch
import torch.nn as nn


## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, n_feat, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(in_channels=n_feat, out_channels=n_feat // reduction, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.ReLU(),
                nn.Conv2d(in_channels=n_feat // reduction, out_channels=n_feat, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, reduction, bias):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        modules_body.append(nn.PReLU())
        modules_body.append(nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride=1, padding=1, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=64, reduction=4, bias=False):
        super(UNet, self).__init__()

        # shallow features 
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_feat, kernel_size=3, stride=1, padding=1, bias=bias),
            CAB(n_feat, reduction, bias)
        )
        
        # Encoder 
        self.encoder1 = [CAB(1 * n_feat, reduction, bias) for _ in range(2)]
        self.encoder2 = [CAB(2 * n_feat, reduction, bias) for _ in range(2)]
        self.encoder3 = [CAB(3 * n_feat, reduction, bias) for _ in range(2)]
        self.encoder4 = [CAB(4 * n_feat, reduction, bias) for _ in range(2)]

        self.encoder1 = nn.Sequential(*self.encoder1)
        self.encoder2 = nn.Sequential(*self.encoder2)
        self.encoder3 = nn.Sequential(*self.encoder3)
        self.encoder4 = nn.Sequential(*self.encoder4)

        self.down12 = DownSample(1 * n_feat, 2 * n_feat)
        self.down23 = DownSample(2 * n_feat, 3 * n_feat)
        self.down34 = DownSample(3 * n_feat, 4 * n_feat)

        # Decoder
        self.decoder1 = [CAB(1 * n_feat, reduction, bias) for _ in range(2)]
        self.decoder2 = [CAB(2 * n_feat, reduction, bias) for _ in range(2)]
        self.decoder3 = [CAB(3 * n_feat, reduction, bias) for _ in range(2)]
        self.decoder4 = [CAB(4 * n_feat, reduction, bias) for _ in range(2)]

        self.decoder1 = nn.Sequential(*self.decoder1)
        self.decoder2 = nn.Sequential(*self.decoder2)
        self.decoder3 = nn.Sequential(*self.decoder3)
        self.decoder4 = nn.Sequential(*self.decoder4)

        self.skip_attn1 = CAB(1 * n_feat, reduction, bias)
        self.skip_attn2 = CAB(2 * n_feat, reduction, bias)
        self.skip_attn3 = CAB(3 * n_feat, reduction, bias)

        self.up21  = SkipUpSample(2 * n_feat, 1 * n_feat)
        self.up32  = SkipUpSample(3 * n_feat, 2 * n_feat)
        self.up43  = SkipUpSample(4 * n_feat, 3 * n_feat)

        # tail
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    
    def forward(self, x):
        # Head
        head = self.shallow_feat(x)

        # Encoder
        enc1 = self.encoder1(head)
        enc2 = self.encoder2(self.down12(enc1))
        enc3 = self.encoder3(self.down23(enc2))
        enc4 = self.encoder4(self.down34(enc3))

        # Decoder
        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(self.up43(dec4, self.skip_attn3(enc3)))
        dec2 = self.decoder2(self.up32(dec3, self.skip_attn2(enc2)))
        dec1 = self.decoder1(self.up21(dec2, self.skip_attn1(enc1)))

        # Tail
        tail = self.tail(dec1)

        # Output
        output = torch.tanh(tail)
        output += x
        output = torch.clamp(output, min=-1, max=1)
        return output
