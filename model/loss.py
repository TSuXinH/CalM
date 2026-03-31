import torch.nn as nn
import torch.nn.functional as F

class PearsonCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-6): 
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        if x.dim()==3: x = x[:,0,:]
        if y.dim()==3: y = y[:,0,:]
        vx = x - x.mean(dim=1, keepdim=True)
        vy = y - y.mean(dim=1, keepdim=True)
        num = (vx * vy).sum(dim=1)
        den = (vx.square().sum(dim=1).sqrt() * vy.square().sum(dim=1).sqrt()) + self.eps
        return (1.0 - num/den).mean()
