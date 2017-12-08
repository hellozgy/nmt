import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from dataset import Constants
from modules import *
import random
import ipdb
import  torch.nn.functional as F
from .BasicModule import BasicModule

class Translate_lstm_wmt17(BasicModule):
    '''
    实现wmt17爱丁堡大学的模型
    '''
    def __init__(self, opt):
        super(Translate_lstm_wmt17, self).__init__(opt)
        vocab_size = max(self.input_size, self.output_size)
        self.encoder = nn.ModuleList([LNGRUCell(self.embeds_size, self.hidden_size) for _ in range(self.Ls)])
        self.encoder_reverse = nn.ModuleList([LNGRUCell(self.embeds_size, self.hidden_size) for _ in range(self.Ls)])
        self.enc_emb = nn.Embedding(vocab_size, self.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.decoder = nn.ModuleList([LNGRUCell(self.embeds_size, self.hidden_size)] + [LNGRUCell(2 * self.hidden_size, self.hidden_size)] +
                                     [LNGRUCell(1, self.hidden_size) for _ in range(self.Lt - 2)])
        self.dec_emb = nn.Embedding(vocab_size, self.embeds_size, padding_idx=Constants.PAD_INDEX)
        if self.attn_general:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        elif self.attn_concat:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(3*self.hidden_size, self.hidden_size))
            self.attn_Va = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        self.fw = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh()
        )
        cutoff = [3000, 20000, vocab_size]
        self.adaptiveSoftmax = AdaptiveSoftmax(self.hidden_size, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)

        self.encoder = Encoder(opt, vocab_size)
        self.decoder = Decoder(opt, vocab_size, self.attention)
        self.encoder.enc_emb.weight = self.decoder.dec_emb.weight


    def _encode(self, inputs):
        return self.encoder(inputs)

    def decode_step(self, prev_y, mask, hiddens=None, ctx=None, beam_search=False):
        '''
        :param prev_y:(batch_size*beam_size, 1)
        :param mask:(batch_size*beam_size, seq_len)
        :return:
        logprob:(batch_size*beam_size, output_size)
        At:(batch_size*beam*size, seq_len)
        '''
        hiddens = self.hiddens if beam_search else hiddens
        ctx = self.ctx if beam_search else ctx
        prev_y = self.embedding_zh(prev_y).permute(1, 0, 2)
        key = hiddens[0][-1, :, :]
        Ct, At = self.attention(ctx, key, mask)
        input = torch.cat((Ct.unsqueeze(0), prev_y), 2)
        input = F.dropout(input, p=self.dropout, training=self.training)
        output, hiddens = self.decoder(input, hiddens)
        output = self.fc(output[0])
        logprob = self.adaptiveSoftmax.log_prob(output) if beam_search else None
        if beam_search: self.hiddens = hiddens
        return logprob, At, output, hiddens

    def update_state(self, re_idx):
        '''update hidden and ctx'''
        self.ctx = self.ctx.index_select(1, re_idx)
        h_x = self.hiddens[0].index_select(1, re_idx)
        c_x = self.hiddens[1].index_select(1, re_idx)
        self.hiddens = [h_x, c_x]

    def repeat_state(self, ctx, hiddens, beam_size):
        seq_len = ctx.size(0)
        batch_size = ctx.size(1)
        self.ctx = ctx.repeat(1, 1, beam_size).view(seq_len, batch_size * beam_size, -1)
        h_n = hiddens[0].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hiddens[0].size(-1))
        c_n = hiddens[1].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hiddens[1].size(-1))
        self.hiddens = [h_n, c_n]


class Encoder(nn.Module):
    def __init__(self, opt, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.enc_emb = nn.Embedding(vocab_size, opt.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.encoder = nn.ModuleList([LNGRUCell(opt.embeds_size, opt.hidden_size)]+
                                     [LNGRUCell(1, opt.hidden_size)]*(opt.Ls - 1))
        self.encoder_reverse = nn.ModuleList([LNGRUCell(opt.embeds_size, opt.hidden_size)]+
                                             [LNGRUCell(1, opt.hidden_size)]*(opt.Ls - 1))
        self.ln = nn.ModuleList([LayerNorm(self.hidden_size)]*(opt.Ls - 1))
        self.ln_reverse = nn.ModuleList([LayerNorm(self.hidden_size)]*(opt.Ls - 1))


    def forward_step(self, inputs, hiddens):
        '''
        :param inputs: 输入的词向量,[(batch, embeds_size)]*2
        :param hiddens: 前一个时间片隐藏状态[(batch, hidden_size)]*2
        :return:
        '''
        _input = Variable(inputs[0].data.new(inputs[0].size(0), 1).zero_().float())
        hidden, hidden_reverse = hiddens
        input, input_reverse = inputs
        for layer, (rnn, rnn_reverse) in enumerate(zip(self.encoder, self.encoder_reverse)):
            if layer > 0 :
                hidden = self.ln[layer-1](hidden)
                hidden_reverse = self.ln_reverse[layer - 1](hidden_reverse)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden_reverse = F.dropout(hidden_reverse, p=self.dropout, training=self.training)
            hidden = rnn(input, hidden)
            hidden_reverse = rnn_reverse(input_reverse, hidden_reverse)
            input = _input; input_reverse = _input
        return hidden, hidden_reverse

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        inputs_emb = self.enc_emb(inputs).permute(1, 0, 2)
        inputs_emb = F.dropout(inputs_emb, p=self.dropout, training=self.training)
        hidden = Variable(inputs.data.new(batch_size, self.hidden_size).zero_().float())
        hidden_reverse = hidden
        outputs = [];outputs_reverse = []
        for i in range(seq_len):
            input,input_reverse = inputs_emb[i],inputs_emb[seq_len - i - 1]
            hidden, hidden_reverse = self.forward_step((input, input_reverse), (hidden, hidden_reverse))
            outputs.append(hidden);outputs_reverse.append(hidden_reverse)
        outputs = torch.stack(outputs)
        outputs_reverse.reverse()
        outputs_reverse = torch.stack(outputs_reverse)
        ctx = torch.cat([outputs, outputs_reverse],-1)
        return ctx, hidden

class Decoder(nn.Module):
    def __init__(self, opt, vocab_size, attn_func):
        super(Decoder, self).__init__()
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.label_smooth = opt.label_smooth
        self.attn_func = attn_func
        self.decoder = nn.ModuleList([LNGRUCell(opt.embeds_size, opt.hidden_size)]+[LNGRUCell(2*opt.hidden_size, opt.hidden_size)]+
                                     [LNGRUCell(1, opt.hidden_size) for _ in range(opt.Lt - 2)])
        self.dec_emb = nn.Embedding(vocab_size, opt.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.fw = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.BatchNorm1d(opt.hidden_size),
            nn.Tanh()
        )
        cutoff = [3000, 20000, vocab_size]
        self.adaptiveSoftmax = AdaptiveSoftmax(opt.hidden_size, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)


    def forward_step(self, mask, ctx, prev_y, hidden):
        '''
        :param ctx:编码阶段的上下文
        :param prev_y: 上个时间片的目标词向量
        :param hidden:上个时间片的隐层状态
        :return:
        '''
        batch_size = mask.size(0)
        _input = Variable(ctx.data.new(batch_size, 1).zero_().float())
        hidden = self.decoder[0](prev_y, hidden)
        context,_ = self.attention(ctx, hidden, mask)
        hidden = self.decoder[1](context, hidden)
        for index, rnn in enumerate(self.decoder):
            if index<2:continue
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = rnn(_input, hidden)
        return hidden

    def forward(self, ctx, hidden, inputs, targets, target_len):
        batch_size = ctx.size(1)
        prev_y = Variable(ctx.data.new(batch_size, self.embeds_size).zero_().float())
        predicts = []
        loss = 0

        mask = torch.eq(inputs, Constants.PAD_INDEX)
        mask = mask.float().masked_fill_(mask, float('-inf'))
        for i in range(target_len):
            hidden = self.forward_step(mask, ctx, prev_y, hidden, self.dropout)
            y = self.fw(hidden)
            target = targets[:, i].contiguous().view(-1)
            o = self.adaptiveSoftmax(y, target)
            loss += self.loss_function(o, target)
            if self.training:
                if self.label_smooth > 0 and random.random() < self.label_smooth:
                    predict = self.adaptiveSoftmax.log_prob(y)
                    y = predict.topk(1, dim=1)[1]
                else:
                    y = targets[:, i].unsqueeze(1)
            else:
                predict = self.adaptiveSoftmax.log_prob(y)
                predict = predict.topk(1, dim=1)[1]
                predicts.append(predict)
                y = predict
            prev_y = self.dec_emb(y)[:, 0, :]
        return predicts, loss / target_len