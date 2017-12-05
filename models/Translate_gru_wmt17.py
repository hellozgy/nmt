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
        self.encoder = Encoder(self.Ls, self.embeds_size, self.hidden_size, vocab_size)
        self.decoder = Decoder(self.Lt, self.embeds_size, self.hidden_size, vocab_size)
        self.encoder.enc_emb.weight = self.decoder.dec_emb.weight

        self.init_weights()

    def forward(self, inputs, targets, target_len=30, label_smooth=0, dropout=0.):
        ctx, hidden = self.encoder(inputs, dropout)
        outputs, loss = self.decoder(ctx, hidden, inputs, targets, target_len, label_smooth, dropout)
        return outputs, loss

class Encoder(nn.Module):
    def __init__(self, Ls, embeds_size, hidden_size, input_vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.ModuleList([LNGRUCell(embeds_size, hidden_size) for _ in range(Ls)])
        self.encoder_reverse = nn.ModuleList([LNGRUCell(embeds_size, hidden_size) for _ in range(Ls)])
        self.enc_emb = nn.Embedding(input_vocab_size, embeds_size, padding_idx=Constants.PAD_INDEX)
        self.linear = nn.Linear(2*hidden_size, hidden_size)


    def forward_step(self, inputs, hiddens, dropout):
        '''
        :param inputs: 输入的词向量,[(batch, embeds_size)]*2
        :param hiddens: 前一个时间片隐藏状态[(batch, hidden_size)]*2
        :return:
        '''
        _input = Variable(inputs[0].data.new(1).zero_().float()).expand_as(inputs[0])
        hidden, hidden_reverse = hiddens
        input, input_reverse = inputs
        for rnn, rnn_reverse in zip(self.encoder, self.encoder_reverse):
            hidden = F.dropout(hidden, p=dropout, training=self.training)
            hidden_reverse = F.dropout(hidden_reverse, p=dropout, training=self.training)
            hidden = rnn(input, hidden)
            hidden_reverse = rnn_reverse(input_reverse, hidden_reverse)
            input = _input; input_reverse = _input
        return hidden, hidden_reverse

    def forward(self, inputs, dropout):
        inputs_emb = self.enc_emb(inputs).permute(1, 0, 2)
        inputs_emb = F.dropout(inputs_emb, p=dropout, training=self.training)
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        hidden = Variable(inputs.data.new(batch_size, self.hidden_size).zero_().float())
        hidden_reverse = hidden
        outputs = [];outputs_reverse = []
        for i in range(seq_len):
            input,input_reverse = inputs_emb[i],inputs_emb[seq_len - i - 1]
            hidden, hidden_reverse = self.forward_step((input, input_reverse), (hidden, hidden_reverse), dropout)
            outputs.append(hidden);outputs_reverse.append(hidden_reverse)
        outputs_reverse.reverse()
        outputs = torch.stack(outputs)
        outputs_reverse = torch.stack(outputs_reverse)
        ctx = torch.cat([outputs, outputs_reverse],-1)
        hidden = F.tanh(self.linear(torch.cat([hidden, hidden_reverse],1)))
        return ctx, hidden

class Decoder(nn.Module):
    def __init__(self, Lt, embeds_size, hidden_size, output_vocab_size):
        super(Decoder, self).__init__()
        self.embeds_size = embeds_size
        self.hidden_size = hidden_size
        self.decoder = nn.ModuleList([LNGRUCell(embeds_size, hidden_size)]+[LNGRUCell(2*hidden_size, hidden_size)]+
                                     [LNGRUCell(1, hidden_size) for _ in range(Lt - 2)])
        self.dec_emb = nn.Embedding(output_vocab_size, embeds_size, padding_idx=Constants.PAD_INDEX)
        self.pt = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.Wa = nn.Linear(self.hidden_size, 2*self.hidden_size)
        self.fw = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh()
        )
        cutoff = [3000, 20000, output_vocab_size]
        self.adaptiveSoftmax = AdaptiveSoftmax(hidden_size, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)


    def attention(self, hs, ht, sentence_en_mask):
        '''
               hs:(seqlen, batch_size, hidden_size)
               ht:(batch_size, hidden_size)
               return :(1,batch_size, hidden_size)
           '''
        seq_len = len(hs)
        hs = hs.permute(1, 0, 2)  # (batch_size, seqlen, hidden_size)
        pt = seq_len * self.pt(ht)
        pt = torch.cat([torch.exp(-((s-pt)**2)/18) for s in range(seq_len)] ,1)
        At = torch.matmul(hs, self.Wa(ht).unsqueeze(2))
        At = At[:,:,0] + sentence_en_mask
        At = F.softmax(At)  # (batch_size,1, seqlen)
        At = At * pt
        At = At.unsqueeze(1)
        attn = torch.bmm(At, hs)  # (batch_size,1, hidden_size)
        return attn[:,0,:], At[:,0,:] # (1, batch_size, hidden_size)

    def forward_step(self, mask, ctx, prev_y, hidden, dropout):
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
            hidden = F.dropout(hidden, p=dropout, training=self.training)
            hidden = rnn(_input, hidden)
        return hidden

    def forward(self, ctx, hidden, inputs, targets, target_len, label_smooth, dropout):
        batch_size = ctx.size(1)
        prev_y = Variable(ctx.data.new(batch_size, self.embeds_size).zero_().float())
        predicts = []
        loss = 0

        mask = torch.eq(inputs, Constants.PAD_INDEX)
        mask = mask.float().masked_fill_(mask, float('-inf'))
        for i in range(target_len):
            hidden = self.forward_step(mask, ctx, prev_y, hidden, dropout)
            y = self.fw(hidden)
            target = targets[:, i].contiguous().view(-1)
            o = self.adaptiveSoftmax(y, target)
            loss += self.loss_function(o, target)
            if self.training:
                if label_smooth > 0 and random.random() < label_smooth:
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