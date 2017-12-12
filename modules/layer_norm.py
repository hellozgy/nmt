import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
        else:
            self.gamma = None
            self.beta = None
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = 1 / (((x - mean).pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
        if self.affine:
            return (x - mean) * std * self.gamma + self.beta
        else:
            return (x - mean)