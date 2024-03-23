from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss

class SoftTarget1(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(SoftTarget1, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        out_t = F.softmax(out_t / self.T, dim=1)
        # 将小于0.05的概率值置为0
        out_t = torch.where(out_t < 0.05, torch.zeros_like(out_t), out_t)
        # out_t[out_t < 0.05] = 0

        # 归一化
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        out_t,
        				reduction='batchmean') * self.T * self.T
        return loss


class SoftTarget2(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(SoftTarget2, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        out_t = F.softmax(out_t, dim=1)
        # 将小于0.05的概率值置为0
        # out_t = torch.where(out_t < 0.05, torch.zeros_like(out_t), out_t)
        out_t[out_t < 0.05] = 0

        # 归一化
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
        				F.softmax(out_t/self.T, dim=1),
        				reduction='batchmean') * self.T * self.T
        return loss
    

class SoftTarget3(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(SoftTarget3, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        out_t = F.softmax(out_t / self.T, dim=1)
        # 将小于0.05的概率值置为0
        # out_t = torch.where(out_t < 0.05, torch.zeros_like(out_t), out_t)
        out_t[out_t < 0.05] = 0

        # 归一化
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
        				F.softmax(out_t/self.T, dim=1),
        				reduction='batchmean') * self.T * self.T
        return loss