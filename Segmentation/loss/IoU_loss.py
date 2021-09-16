import torch
import torch.nn as nn

def IoULoss_forward(input, target, coeff=1.):
    eps = 0.0001
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    t = (inter.float() + eps) / union.float()
    return coeff * torch.abs(1-t)

def IoULoss(input, target, coeff=1.):
    s = torch.FloatTensor(1).to(input.device).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + IoULoss_forward(c[0], c[1], coeff)
    return s / (i + 1)