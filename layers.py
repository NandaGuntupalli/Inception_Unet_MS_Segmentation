import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import operator
import sigpy as sp
import cupy as cp
from itertools import filterfalse
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, 3, 1))
            self.register_buffer('running_var', torch.zeros(1, 3, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

def _spectral_crop(array, array_shape, bounding_shape):
    start = tuple(map(lambda a, da: (a-da)//2, array_shape, bounding_shape))
    end = tuple(map(operator.add, start, bounding_shape))
    slices = tuple(map(slice, start, end))
    return array[slices]

def _spectral_pad(array, array_shape, bounding_shape):
    out = cp.zeros(bounding_shape)
    start = tuple(map(lambda a, da: (a-da)//2, bounding_shape, array_shape))
    end = tuple(map(operator.add, start, array_shape))
    slices = tuple(map(slice, start, end))
    out[slices] = array
    return out

def DiscreteHartleyTransform(input):
    N = input.ndim
    axes_n = np.arange(2,N)
    fft = sp.fft(input, axes=axes_n)
    H = fft.real - fft.imag
    return H

def CropForward(input, return_shape):

    output_shape = np.zeros(input.ndim).astype(int)
    output_shape[0] = input.shape[0]
    output_shape[1] = input.shape[1]
    output_shape[2:] = np.asarray(return_shape).astype(int)

    dht = DiscreteHartleyTransform(input)
    dht = _spectral_crop(dht, dht.shape, output_shape)
    dht = DiscreteHartleyTransform(dht)

    return dht

def PadBackward(grad_output, input_shape):
    dht = DiscreteHartleyTransform(grad_output)
    dht = _spectral_pad(dht, dht.shape, input_shape)
    dht = DiscreteHartleyTransform(dht)

    return dht


class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, return_shape):
         input = sp.from_pytorch(input)
         ctx.input_shape = input.shape
         output = CropForward(input, return_shape)
         output = sp.to_pytorch(output)
         output = output.float()
         return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = sp.from_pytorch(grad_output)
        grad_input = PadBackward(grad_output, ctx.input_shape)
        grad_input = sp.to_pytorch(grad_input)
        grad_input = grad_input.float()
        return grad_input, None, None


class SpectralPoolNd(nn.Module):
    def __init__(self, return_shape):
        super(SpectralPoolNd, self).__init__()
        self.return_shape = return_shape

    def forward(self, input):
        return SpectralPoolingFunction.apply(input, self.return_shape)

class ConvBlock(nn.Module):
  def __init__(self, input, output, kernel_size, padding):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(input, output, kernel_size, stride=1, padding=padding).to(device)
    self.Relu = nn.ReLU().to(device)

  def forward(self, x):
    x = x.to(device)
    return self.Relu(self.conv(x))

class ConvBlockIso(nn.Module):
  def __init__(self, input, output, kernel_size, padding):
    super(ConvBlockIso, self).__init__()
    self.padding = padding
    if padding == "same":
      self.padding = int((1*(kernel_size-1)) - 1 + kernel_size)//2
    if padding == "valid":
      self.padding = 0
    else:
      pass
    self.conv = nn.Conv2d(input, output, kernel_size, stride=1, padding=padding).to(device)
    self.Relu = nn.ReLU().to(device)

  def forward(self, x):
    return self.Relu(self.conv(x))

class HybridPool2D(nn.Module):
  def __init__(self, return_shape, kernel_size, stride, padding):
    super(HybridPool2D, self).__init__()
    self.padding = padding
    if padding == "same":
      self.padding = int((1*(kernel_size-1)) - stride + kernel_size)//2
    if padding == "valid":
      self.padding = 0
    else:
      pass
    self.spectral_pool = SpectralPoolNd(return_shape)
    self.maxpool = nn.MaxPool2d(kernel_size, stride, self.padding)

  def forward(self, x):
    max_pooled = self.maxpool(x)
    spectral_pooled = self.spectral_pool(x)
    spectral_pooled = torch.nn.functional.interpolate(spectral_pooled, size=max_pooled.shape[-2:], mode='bilinear', align_corners=False)
    spectral_pooled = spectral_pooled.to(device)
    output = torch.concat([max_pooled, spectral_pooled], dim=1)
    output = ConvBlock(output.shape[1], max_pooled.shape[1], kernel_size=1, padding=0)(output)
    return output

class HybridPool2DInception(nn.Module):
  def __init__(self, return_shape, padding, output_shape, kernel_size=2, stride=2):
    super(HybridPool2DInception, self).__init__()
    self.padding = padding
    self.output_shape = output_shape
    if padding == "same":
      self.padding = int((1*(kernel_size-1)) - stride + kernel_size)//2
    if padding == "valid":
      self.padding = 0
    else:
      pass
    self.spectral_pool = SpectralPoolNd(return_shape)
    self.maxpool = nn.MaxPool2d(kernel_size, stride, self.padding).to(device)

  def forward(self, x):
    max_pooled = self.maxpool(x)
    spectral_pooled = self.spectral_pool(x)
    spectral_pooled = torch.nn.functional.interpolate(spectral_pooled, size=max_pooled.shape[-2:], mode='bilinear', align_corners=False)
    spectral_pooled = spectral_pooled.to(device)
    output = torch.concat([max_pooled, spectral_pooled], dim=1)
    output = ConvBlock(output.shape[1], max_pooled.shape[1], kernel_size=1, padding=0)(output)
    output = torch.nn.functional.interpolate(output, size=self.output_shape[1:], mode='bilinear', align_corners=False)
    return output

class InceptionBlock_Real(nn.Module):
  def __init__(self, input_channels, output, hybpl):
    super(InceptionBlock_Real, self).__init__()
    self.output = output

    self.Conv1 = ConvBlock(input_channels, output, kernel_size=1, padding=0)
    self.Conv2 = ConvBlock(input_channels, output, kernel_size=3, padding=1)
    self.Conv3 = ConvBlock(input_channels, output, kernel_size=5, padding=2)
    self.HybridPool = HybridPool2DInception(return_shape=(hybpl, hybpl), padding="valid", output_shape=(output, hybpl, hybpl))

  def forward(self, x):
    x = x.to(device)
    x = torch.cat([self.Conv1(x), self.Conv2(x), self.Conv3(x),
                   self.HybridPool(x)], dim=1)
    return x