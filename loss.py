import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple

class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.register_buffer("kernel", self._cal_gaussian_kernel(11, 1.5))
        self.L = 2.0
        self.k1 = 0.01
        self.k2 = 0.03

    @staticmethod
    def _cal_gaussian_kernel(size, sigma):
        g = torch.Tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        g = g / g.sum()
        window = g.reshape([-1, 1]).matmul(g.reshape([1, -1]))
        kernel = torch.reshape(window, [1, 1, size, size]).repeat(3, 1, 1, 1)
        return kernel

    def forward(self, img0, img1):
        """
        :param img0: range in (-1, 1)
        :param img1: range in (-1, 1)
        :return: SSIM loss i.e. 1 - ssim
        """
        mu0 = torch.nn.functional.conv2d(img0, self.kernel, padding=0, groups=3)
        mu1 = torch.nn.functional.conv2d(img1, self.kernel, padding=0, groups=3)
        mu0_sq = torch.pow(mu0, 2)
        mu1_sq = torch.pow(mu1, 2)
        var0 = torch.nn.functional.conv2d(img0 * img0, self.kernel, padding=0, groups=3) - mu0_sq
        var1 = torch.nn.functional.conv2d(img1 * img1, self.kernel, padding=0, groups=3) - mu1_sq
        covar = torch.nn.functional.conv2d(img0 * img1, self.kernel, padding=0, groups=3) - mu0 * mu1
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        ssim_numerator = (2 * mu0 * mu1 + c1) * (2 * covar + c2)
        ssim_denominator = (mu0_sq + mu1_sq + c1) * (var0 + var1 + c2)
        ssim = ssim_numerator / ssim_denominator
        ssim_loss = 1.0 - ssim
        return ssim_loss


class MixedPix2PixLoss(torch.nn.Module):
    def __init__(self):
        super(MixedPix2PixLoss, self).__init__()
        self.alpha = 0.84
        self.ssim_loss = SSIMLoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred, target):
        """
        :param pred: (bs, c, h, w) image ranging in (-1, 1)
        :param target: (bs, c, h, w) image ranging in (-1, 1)
        :param reduce: (str) reduction method, "mean" or "none" or "sum"
        :return:
        """
        ssim_loss = torch.mean(self.ssim_loss(pred, target))
        l1_loss = self.l1_loss(pred, target)
        weighted_mixed_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        return weighted_mixed_loss


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple('VGGOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()

        conv_3_3_layer = 14
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.model = nn.Sequential()

        for i, layer in enumerate(list(vgg_pretrained_features)):
            self.model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        output = self.model.forward(x)
        return output


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss