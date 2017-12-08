import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from dataset import Constants
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

        self.attn_fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
        )

    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-0.02, 0.02)
            else:
                p.data.zero_()

    def forward(self, inputs, targets=None, target_len=30):
        ctx, hiddens = self.encode(inputs)
        predicts, batch_loss, aligns= self.decode(ctx, hiddens,
                                                   inputs, targets, target_len)
        return predicts, batch_loss

    def encode(self, inputs, beam_size=0):
        ctx, hiddens = self._encode(inputs)
        if beam_size == 0:
            return ctx, hiddens
        else:
            self.repeat_state(ctx, hiddens, beam_size)

    def decode(self, ctx, hiddens, inputs, targets, target_max_len):
        batch_size = inputs.size(0)
        prev_y = Variable(inputs.data.new(batch_size, 1).zero_().long())
        aligns = []
        predicts = []
        loss = 0
        mask = torch.eq(inputs, Constants.PAD_INDEX).data
        mask = mask.float().masked_fill_(mask, -float('inf'))
        for i in range(target_max_len):
            _, At, output, hiddens = self.decode_step(prev_y, mask, hiddens, ctx, beam_search=False)
            aligns.append(At)
            target = targets[:, i].contiguous().view(-1)
            o = self.adaptiveSoftmax(output, target)
            loss += self.loss_function(o, target)
            if self.training:
                if random.random() >= self.label_smooth:
                    prev_y = targets[:, i].unsqueeze(1)
                else:
                    logprob = self.adaptiveSoftmax.log_prob(output)
                    prev_y = logprob.topk(1, dim=1)[1]
            else:
                logprob = self.adaptiveSoftmax.log_prob(output)
                prev_y = logprob.topk(1, dim=1)[1]
                predicts.append(prev_y)
        return predicts, loss/target_max_len, aligns

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

    def get_optimizer(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        return optimizer

