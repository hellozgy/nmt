# coding=utf-8
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from dataset import Constants
from modules import *
from .BasicModule import BasicModule
import ipdb

class Sogou(BasicModule):
    '''wmt17搜狗公司的模型'''
    def __init__(self, opt):
        super(Sogou, self).__init__(opt)
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt, self.attention)

        self.fw = nn.Sequential(
            nn.Linear(self.hidden_size, self.embeds_size),
            nn.BatchNorm1d(self.embeds_size),
            nn.Tanh()
        )

        cutoff = [3000, 20000, self.output_size]
        self.adaptiveSoftmax = AdaptiveSoftmax(self.embeds_size, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)

        if self.attn_general:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
            nn.init.xavier_normal(self.attn_Wa)
        elif self.attn_concat:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(2*self.hidden_size, self.hidden_size))
            self.attn_Va = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
            nn.init.xavier_normal(self.attn_Wa)

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
        output, hiddens, At = self.decoder.forward_step(mask, ctx, prev_y, hiddens)
        output = self.fw(output)
        logprobs = self.adaptiveSoftmax.log_prob(output) if beam_search else None
        if beam_search:self.hiddens = hiddens
        return logprobs, At, output, hiddens

    def update_state(self, re_idx):
        '''update hidden and ctx'''
        self.ctx = self.ctx.index_select(1, re_idx)
        for layer in range(len(self.hiddens)):
            hidden = self.hiddens[layer]
            hx = hidden[0].index_select(0, re_idx)
            cx = hidden[1].index_select(0, re_idx)
            self.hiddens[layer] = (hx, cx)

    def repeat_state(self, ctx, hiddens, beam_size):
        seq_len = ctx.size(0)
        batch_size = ctx.size(1)
        self.ctx = ctx.repeat(1, 1, beam_size).view(seq_len, batch_size * beam_size, -1)

        for layer in range(len(hiddens)):
            hidden = hiddens[layer]
            hx = hidden[0].repeat(1, beam_size).view(batch_size * beam_size, hidden[0].size(1))
            cx = hidden[1].repeat(1, beam_size).view(batch_size * beam_size, hidden[1].size(1))
            hiddens[layer] = (hx, cx)
        self.hiddens = hiddens

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.dropout = opt.dropout
        self.enc_emb = nn.Embedding(opt.input_size, opt.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.encoder = nn.ModuleList([nn.LSTM(opt.embeds_size, opt.hidden_size, bidirectional=True)]+
                                     [nn.LSTM(2*opt.hidden_size, opt.hidden_size)]+
                                     [nn.LSTM(opt.hidden_size, opt.hidden_size)])
        self.ln = nn.ModuleList([LayerNorm(2*opt.hidden_size)]+
                                [LayerNorm(opt.hidden_size)])

    def forward(self, inputs_word):
        inputs_emb = self.enc_emb(inputs_word).permute(1,0,2)
        inputs = F.dropout(inputs_emb, p=self.dropout, training=self.training)
        for layer, lstm in enumerate(self.encoder):
            outputs, hidden = lstm(inputs)
            if layer < 2:
                inputs = F.dropout(self.ln[layer](outputs), p=self.dropout, training=self.training)
        ctx = outputs
        return ctx, [(hidden[0][0], hidden[1][0])]*4

class Decoder(nn.Module):
    def __init__(self, opt, attn_func):
        super(Decoder, self).__init__()
        self.dropout = opt.dropout
        self.attn_func = attn_func
        self.dec_emb = nn.Embedding(opt.output_size, opt.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.decoder = nn.ModuleList([nn.LSTMCell(opt.embeds_size, opt.hidden_size)]+
                                  [nn.LSTMCell(opt.hidden_size, opt.hidden_size) for _ in range(2)]+
                                  [cLSTMCell(opt.embeds_size, opt.hidden_size, opt.hidden_size)])
        self.ln = nn.ModuleList([LayerNorm(opt.hidden_size) for _ in range(3)])

    def forward_step(self, mask, ctx, prev_y, hiddens):
        prev_embeds = self.dec_emb(prev_y)[:,0,:]
        inputs = prev_embeds
        _hiddens = []
        for layer in range(3):
            hx, cx = self.decoder[layer](inputs, hiddens[layer])
            _hiddens.append((hx, cx))
            inputs = F.dropout(self.ln[layer](hx), p=self.dropout, training=self.training)
        attn, At = self.attn_func(ctx, _hiddens[0][0], mask)
        hx, cx = self.decoder[3](prev_embeds, hiddens[3], attn, inputs)
        _hiddens.append((hx, cx))
        return hx, _hiddens, At