import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import Constants
from modules import *
from .BasicModule import BasicModule
import random
import ipdb

class Translate_lstm(BasicModule):
    def __init__(self, opt):
        super(Translate_lstm, self).__init__(opt)
        self.embedding_en = nn.Embedding(self.input_size, self.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.embedding_zh = nn.Embedding(self.output_size, self.embeds_size, padding_idx=Constants.PAD_INDEX)
        self.encoder = nn.LSTM(self.embeds_size, self.hidden_size,
                               num_layers=self.Ls, bidirectional=False, dropout=self.dropout)
        self.decoder = nn.LSTM(self.embeds_size + self.hidden_size, self.hidden_size,
                               num_layers=self.Lt, bidirectional=False, dropout=self.dropout)

        if self.attn_general:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(self.hidden_size, 2*self.hidden_size))
        elif self.attn_concat:
            self.attn_Wa = nn.Parameter(torch.FloatTensor(2*self.hidden_size, self.hidden_size))
            self.attn_Va = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.embeds_size),
            nn.BatchNorm1d(self.embeds_size),
            nn.Tanh()
        )
        cutoff = [3000, 20000, self.output_size]
        self.adaptiveSoftmax = AdaptiveSoftmax(self.embeds_size, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)

    def _encode(self, inputs):
        '''
        :param inputs: (batch_size, seq_len)
        :param beam_size:
        :return:
        '''
        inputs = self.embedding_en(inputs).permute(1, 0, 2)
        inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        ctx, hiddens = self.encoder(inputs)
        return ctx, hiddens

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
        if beam_search:self.hiddens = hiddens
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