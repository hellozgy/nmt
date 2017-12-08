from .BasicModule import BasicModule
from dataset import Constants
import torch.nn as nn
import torch
from torch.autograd import Variable
import random
import numpy as np
from modules import *
import ipdb
import torch.nn.functional as F
import math

class Translate_lstm_resconn(BasicModule):
    def __init__(self, opt):
        super(Translate_lstm_resconn, self).__init__(opt)
        self.embedding_en = nn.Embedding(self.input_size, self.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.embedding_zh = nn.Embedding(self.output_size, self.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.encoder = nn.ModuleList([nn.LSTM(self.embeds_size, self.hidden_size, bidirectional=True)]+
                                     [nn.LSTM(2*self.hidden_size, self.hidden_size)]+
                                     [nn.LSTM(self.hidden_size, self.hidden_size)]*(self.Ls-1))
        self.decoder = nn.ModuleList([nn.LSTM(self.embeds_size + self.hidden_size, self.hidden_size)]+
                                     [nn.LSTM(self.hidden_size, self.hidden_size)]*(self.Lt-1))

        if self.attn_general:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        elif self.attn_concat:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(2*self.hidden_size, self.hidden_size))
            self.attn_Va = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.embeds_size),
            nn.BatchNorm1d(self.embeds_size),
            nn.Tanh(),
        )
        cutoff = [3000, 20000, self.output_size]
        self.adaptiveSoftmax = AdaptiveSoftmax(self.embeds_size, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)

    def _encode(self, inputs):
        embeds = self.embedding_en(inputs).permute(1, 0, 2)
        input, _ = self.encoder[0](embeds)
        hiddens = []
        for layer, lstm in enumerate(self.encoder):
            if layer==0:continue
            output, hidden = lstm(input)
            hiddens.append(hidden)
            if layer > 2:
                input = input + output
            else:
                input = output
        return output, hiddens

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
        key = hiddens[-1][0][0]
        Ct, At = self.attention(ctx, key, mask)
        input = torch.cat((Ct.unsqueeze(0), prev_y), 2)
        input = F.dropout(input, p=self.dropout, training=self.training)
        for layer, lstm in enumerate(self.decoder):
            output, hidden = lstm(input, hiddens[layer])
            hiddens[layer] = hidden
            if layer == 0:
                input = Ct.unsqueeze(0) + output
            else:
                input = input + output
        output = self.fc(output[0])
        logprob = self.adaptiveSoftmax.log_prob(output) if beam_search else None
        if beam_search:self.hiddens = hiddens
        return logprob, At, output, hiddens

    def update_state(self, re_idx):
        '''update hidden and ctx'''
        self.ctx = self.ctx.index_select(1, re_idx)
        for i in range(len(self.hiddens)):
            h_x = self.hiddens[i][0].index_select(1, re_idx)
            c_x = self.hiddens[i][1].index_select(1, re_idx)
            self.hiddens[i] = [h_x, c_x]

    def repeat_state(self, ctx, hiddens, beam_size):
        seq_len = ctx.size(0)
        batch_size = ctx.size(1)
        self.ctx = ctx.repeat(1, 1, beam_size).view(seq_len, batch_size * beam_size, -1)
        for i in range(len(hiddens)):
            hidden = hiddens[i]
            h_n = hidden[0].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hidden[0].size(-1))
            c_n = hidden[1].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hidden[1].size(-1))
            hiddens[i] = [h_n, c_n]
        self.hiddens = hiddens

