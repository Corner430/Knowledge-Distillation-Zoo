from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.autograd.gradcheck import zero_gradients

"""
Modified by https://github.com/bhheo/BSS_distillation
"""


def reduce_sum(x, keepdim=True):
    for d in reversed(range(1, x.dim())):
        x = x.sum(d, keepdim=keepdim)
    return x


def l2_norm(x, keepdim=True):
    norm = reduce_sum(x * x, keepdim=keepdim)
    return norm.sqrt()


class BSS(nn.Module):
    """
    Knowledge Distillation with Adversarial Samples Supporting Decision Boundary
    https://arxiv.org/pdf/1805.05532.pdf
    """

    def __init__(self, T):
        super(BSS, self).__init__()
        self.T = T

    def forward(self, attacked_out_s, attacked_out_t):
        loss = F.kl_div(
            F.log_softmax(attacked_out_s / self.T, dim=1),
            F.softmax(attacked_out_t / self.T, dim=1),
            reduction="batchmean",
        )  # * self.T * self.T

        return loss


class BSSAttacker:
    def __init__(self, step_alpha, num_steps, eps=1e-4):
        # 初始化攻击者参数
        self.step_alpha = step_alpha  # 步长
        self.num_steps = num_steps  # 攻击步数
        self.eps = eps  # 防止除以0的小值

    def attack(self, model, img, target, attack_class):
        # 将图像设置为需要梯度
        img = img.detach().requires_grad_(True)

        step = 0
        # 进行指定步数的攻击
        while step < self.num_steps:
            # 清除图像的梯度
            img.grad.zero_()
            # 通过模型获取输出
            _, _, _, _, _, output = model(img)

            # 计算softmax得分
            score = F.softmax(output, dim=1)
            # 获取目标类别的得分
            score_target = score.gather(1, target.unsqueeze(1))
            # 获取攻击类别的得分
            score_attack_class = score.gather(1, attack_class.unsqueeze(1))

            # 计算损失，即攻击类别得分与目标类别得分的差值
            loss = (score_attack_class - score_target).sum()
            # 反向传播计算梯度
            loss.backward()

            # 计算步长，如果目标类别与输出最大得分类别相同，则步长为self.step_alpha，否则为0
            step_alpha = self.step_alpha * (target == output.max(1)[1]).float()
            # 增加维度以匹配图像的维度
            step_alpha = step_alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # 如果所有步长都为0，即所有目标类别都不是输出最大得分类别，停止攻击
            if step_alpha.sum() == 0:
                break

            # 计算扰动，即目标类别得分与攻击类别得分的差值
            pert = (score_target - score_attack_class).unsqueeze(1).unsqueeze(1)
            # 计算规范化的扰动，即扰动与图像梯度的乘积，乘以步长和防止除以0的小值
            norm_pert = step_alpha * (pert + self.eps) * img.grad / l2_norm(img.grad)

            # 计算扰动后的图像
            step_adv = img + norm_pert
            # 将扰动后的图像的值限制在[-2.5, 2.5]之间
            step_adv = torch.clamp(step_adv, -2.5, 2.5)
            # 更新图像
            img.data = step_adv.data

            # 进行下一步攻击
            step += 1

        # 返回攻击后的图像
        return img
