import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki._private_kernels.conv import conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh


@nki.jit
def conv_wrap(img_ref, filter_ref, out_shape):
    out_arr = nl.ndarray(shape=out_shape, dtype=img_ref.dtype, buffer=nl.hbm)
    conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(img_ref, filter_ref, out_arr, **{
        'input': img_ref.shape,
        'filter': filter_ref.shape,
        'output': out_shape,
        'in_perm': [0, 1, 2, 3],
        'kern_perm': [0, 1, 2, 3],
        'out_perm': [0, 1, 2, 3],
        'stride': (1, 1), 
        'padding': ((1, 1), (1, 1))})
    return out_arr


def conv_output_shape(input_shape, weight_shape, stride=1, padding=1):
    assert len(input_shape) == 4
    batch, _, in_h, in_w = input_shape

    assert len(weight_shape) == 4
    out_channels, _, kernel_h, kernel_w = weight_shape

    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1

    return (batch, out_channels, out_h, out_w)


class Conv2dBackward(Function):
    @staticmethod
    def forward(ctx, X, K):
        ctx.save_for_backward(X, K)
        return F.conv2d(X, K, padding=1)

    @staticmethod
    def backward(ctx, dL_dO):
        X, K = ctx.saved_tensors

        dL_dK = conv_wrap(
            X.transpose(0, 1), 
            dL_dO.transpose(0, 1), 
            conv_output_shape(
                (X.shape[1], X.shape[0], X.shape[2], X.shape[3]), 
                (dL_dO.shape[1], dL_dO.shape[0], dL_dO.shape[2], dL_dO.shape[3])
            )
        ).transpose(0, 1)
        dL_dX = F.conv_transpose2d(dL_dO, K, stride=1, padding=1)

        return dL_dX, dL_dK


class BwdConv2dWithKernel(nn.Module):
    """
    Custom Conv2d module using using NKI convolution kernel for backward pass
    Fixed: padding=1, bias=False
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
        super().__init__()

        assert padding == 1
        assert bias == False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        nn.init.kaiming_uniform_(self.weight, a=0.0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return Conv2dBackward.apply(x, self.weight)

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'kernel_size={self.kernel_size}, padding=1, bias=False'


# Based on:
# milesial, U-Net: Semantic segmentation with PyTorch, GitHub repository
# https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
