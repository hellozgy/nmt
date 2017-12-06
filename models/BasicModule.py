import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb


class BasicModule(nn.Module):
    def __init__(self, opt):
        super(BasicModule, self).__init__()
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.Ls = opt.Ls
        self.Lt = opt.Lt
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.attn_general = opt.attn_general
        self.attn_concat = opt.attn_concat
        self.dropout = opt.dropout
        self.label_smooth = opt.label_smooth

        # self.attn_Wa = None
        # self.attn_Va = None
        self.attn_fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-0.02, 0.02)
            else:
                p.data.zero_()

    def get_optimizer(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def attention(self, ctx, key, mask):
        '''
        :param ctx: (seq_len, batch_size, hidden_size*n)
        :param key: (batch_size, hidden_size)
        :param mask: (batch_size, seq_len)
        :return:
        :attn:(batch_size, hidden_size)
        :At:(batch_size, seq_len)
        '''
        residual = key
        seq_len = ctx.size(0)
        ctx = ctx.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size*n)
        if self.attn_general: #general
            At = ctx.matmul(key.matmul(self.attn_Wa).unsqueeze(2)).squeeze(-1)
        elif self.attn_concat: # concat
            key = torch.stack([key]*seq_len, 1) #(batch_size, seq_len, hidden_size)
            At = F.tanh(torch.cat([ctx, key], -1).matmul(self.attn_Wa)).matmul(self.attn_Va).squeeze(-1)
        else: # dot
            key = torch.unsqueeze(key, 2) # (batch_size, hidden_size, 1)
            At = torch.matmul(ctx, key).squeeze(-1)  # (batch_size, seqlen)
        At = At + Variable(mask)
        At = F.softmax(At, dim=1).unsqueeze(1)  # (batch_size,1, seqlen)
        attn = torch.bmm(At, ctx)[:,0,:]  # (batch_size,1, hidden_size)
        attn = self.attn_fc(torch.cat([attn, residual], 1))
        return attn, At[:, 0, :]

