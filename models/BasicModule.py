import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self, opt):
        super(BasicModule, self).__init__()
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.Ls = opt.Ls
        self.Lt = opt.Lt
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.global_attn = opt.global_attn
        self.local_attn = opt.local_attn

    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-0.02, 0.02)
            else:
                p.data.zero_()

    def get_optimizer(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

