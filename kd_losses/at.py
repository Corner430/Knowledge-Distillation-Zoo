from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
AT with sum of absolute values with power p
"""


class AT(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    """

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        # 绝对值的 p 次方
        am = torch.pow(torch.abs(fm), self.p)
        # 对张量 am 沿着第二个维度（通道）进行求和
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        # 将张量 am 的每个元素除以张量 norm 的对应元素加上一个小的常数 eps
        am = torch.div(am, norm + eps)

        return am
