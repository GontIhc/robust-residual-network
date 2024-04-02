import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import models
from core.losses.LabelSmoothing import LabelSmoothing

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

""" https://arxiv.org/abs/1901.08573 """


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


class ATLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0,
                 distance='l_inf', ce=False, cutmix=False, adjust_freeze=True, labelSmoothing=False):
        super(ATLoss, self).__init__()
        # 每次迭代时用于生成对抗样本的步长
        self.step_size = step_size
        # 对抗样本生成时允许的最大扰动范围
        self.epsilon = epsilon
        # 生成对抗样本时的迭代次数
        self.perturb_steps = perturb_steps
        # 用于平衡自然损失和鲁棒损失的权重
        self.beta = beta
        # 指定对抗样本的类型，可以是'l_inf'或'l_2'
        self.distance = distance
        self.ce = ce
        self.criterion_kl = nn.KLDivLoss(reduction='sum')  # KL散度损失
        self.cross_entropy = models.CutMixCrossEntropyLoss() if cutmix else torch.nn.CrossEntropyLoss()

        self.labelSmoothing_en = labelSmoothing
        self.labelSmoothing = LabelSmoothing(0.3)  # 平滑系数

        # 是否调整模型参数的梯度要求
        self.adjust_freeze = adjust_freeze

    def forward(self, model, x_natural, y, optimizer, kd_ratio=0., teacher_model=None):
        # define KL-loss
        # criterion_kl = self.criterion_kl
        model.eval()
        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = False

        # generate adversarial example

        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

        # 如果distance是'l_inf'，则使用PGD生成对抗样本
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        # 如果是'l_2'，则使用优化器对扰动delta进行迭代，以生成对抗样本。
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = True

        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits = model(x_natural)
        loss_natural = self.cross_entropy(logits, y)
        adv_logits = model(x_adv)

        if self.labelSmoothing_en:  # 使用标签平滑
            loss_robust = self.labelSmoothing(adv_logits, y)  # 这个地方可以修改
            loss = loss_robust
        else:
            loss_robust = self.cross_entropy(adv_logits, y)  # 这个地方可以修改
            loss = loss_natural + self.beta * loss_robust

        # soft target
        if kd_ratio > 0:
            teacher_model.train()
            with torch.no_grad():
                soft_logits = teacher_model(x_adv).detach()
                soft_label = F.softmax(soft_logits, dim=1)
            kd_loss = cross_entropy_loss_with_soft_target(adv_logits, soft_label) * kd_ratio
            loss = loss + kd_loss
            return adv_logits, loss, kd_loss
        else:
            return adv_logits, loss, 0
