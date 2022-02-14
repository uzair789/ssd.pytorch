""" Buildign blocks for binarizing a layer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        #print('retinanet/binary_units.py/BinaryActivation forward')
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        # self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(HardBinaryConv, self).__init__()
        #print('retinanet/binary_units.py/HardBinaryCOnv init')
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_channels = out_chn
        # print('----------------in:{}, out:{}, k:{}-----------------'.format(in_chn, out_chn, kernel_size))
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        #print('retinanet/binary_units.py/HardBinaryCOnv forward')
        # print('Binary running ------------------------------------')
        #real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, dilation=self.dilation)

        return y

class BinaryLinear(nn.Module):
    def __init__(self, in_chn, out_chn, bias=False):
        super(BinaryLinear, self).__init__()
        # self.number_of_weights = in_chn * out_chn
        self.shape = (out_chn, in_chn)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        self.bias = None
        if bias:
            self.bias = LearnableBias(out_chn)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.linear(x, binary_weights)
        if self.bias:
            y = self.bias(y)
        return y
