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

        self.pt = nn.Sequential(
            nn.Linear(2*self.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.Wa = nn.Linear(self.hidden_size, self.hidden_size)
        if self.global_attn and not self.local_attn:
            self.pt = None
            self.Wa = None
        self.attn_fc = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.embeds_size),
            nn.BatchNorm1d(self.embeds_size),
            nn.Tanh(),
        )
        cutoff = [3000, 20000, self.output_size]
        self.adaptiveSoftmax = AdaptiveSoftmax(self.embeds_size, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)

        self.init_weights()

    def encode(self, sentence_en):
        embeds = self.embedding_en(sentence_en).permute(1, 0, 2)
        input, _ = self.encoder[0](embeds)
        hiddens = []
        for layer, lstm in enumerate(self.encoder):
            if layer==0:continue
            output, hidden = lstm(input)
            hiddens.append(hidden)
            if layer == 2:
                input = (output + embeds) * math.sqrt(0.5)
            elif layer > 2:
                input = (input + output) * math.sqrt(0.5)
            else:
                input = output
        return (output + embeds) * math.sqrt(0.5), hiddens

    def attention(self, hs, ht, sentence_en_mask, ctx):
        '''
        :param hs: (seq_len, batch_size, hidden_size)
        :param ht: (batch_size, hidden_size)
        :param sentence_en_mask: (batch_size, seq_len)
        :return: (batch_size, hidden_size)
        '''
        _ht = ht
        seq_len = len(hs)
        ht = torch.unsqueeze(ht, 2)
        hs = hs.permute(1, 0, 2)  # hs: (batch_size, seqlen, hidden_size)
        if self.global_attn:
            At = torch.matmul(hs, ht).view(-1, seq_len)  # (batch_size, seqlen)
            At = At + sentence_en_mask
            At = F.softmax(At).unsqueeze(1)  # (batch_size,1, seqlen)
        else:
            pt = seq_len * self.pt(torch.cat([ht[:, :, 0], ctx], -1))
            pt = torch.cat([torch.exp(-((s - pt) ** 2) / 18) for s in range(seq_len)], 1)
            At = torch.matmul(hs, self.Wa(ht[:, :, 0]).unsqueeze(2)).view(-1, seq_len)
            At = At + sentence_en_mask
            At = F.softmax(At)  # (batch_size,1, seqlen)
            At = At * pt
            At = At.unsqueeze(1)
        attn = torch.bmm(At, hs)[:,0,:]  # (batch_size,1, hidden_size)
        attn = self.attn_fc(torch.cat([attn, _ht], 1))
        return attn, At[:, 0, :]  # (1, batch_size, hidden_size)

    def decode(self, outputs_encoder, hidden_encoder, sentence_en, sentence_zh=None, target_max_len=30, label_smooth=0):
        batch_size = sentence_en.size(0)
        y_embs = Variable(sentence_en.data.new(batch_size, self.embeds_size).zero_().float())
        hiddens = hidden_encoder
        aligns = []
        predicts = []
        loss = 0
        sentence_en_mask = torch.eq(sentence_en, Constants.PAD_INDEX)
        sentence_en_mask = sentence_en_mask.float().masked_fill_(sentence_en_mask, float('-inf'))

        target_max_len = sentence_zh.size(1) if self.training else target_max_len
        ctx = hiddens[-1][0][0]
        for i in range(target_max_len):
            ht = hiddens[-1][0][0]
            Ct, At = self.attention(outputs_encoder, ht, sentence_en_mask, ctx)
            input = torch.cat([Ct, y_embs], -1).unsqueeze(0)
            for layer, lstm in enumerate(self.decoder):
                output, hidden = lstm(input, hiddens[layer])
                hiddens[layer] = hidden
                if layer == 0:
                    input = ((Ct + y_embs).unsqueeze(0) + output) * math.sqrt(1 / 3)
                else:
                    input = (input + output) * math.sqrt(0.5)
            y_embs = self.fc(output[0])
            target = sentence_zh[:, i].contiguous().view(-1)
            o = self.adaptiveSoftmax(y_embs, target)
            loss += self.loss_function(o, target)
            if self.training and random.random()>=label_smooth:
                y_embs = sentence_zh[:, i].unsqueeze(1)
            else:
                predict = self.adaptiveSoftmax.log_prob(y_embs)
                predict = predict.topk(1, dim=1)[1]
                predicts.append(predict)
                y_embs = predict
            y_embs = self.embedding_zh(y_embs)[:,0,:]
        return predicts, loss/target_max_len

    def forward(self, inputs, target_sentence, target_len=30, label_smooth=0):
        outputs_encoder, hidden_encoder = self.encode(inputs)
        predicts, batch_loss= self.decode(outputs_encoder, hidden_encoder,
                                                   inputs, target_sentence, target_len)
        return predicts, batch_loss

    def translate_batch(self, sentence_en, pos, beam_size=5, max_len_zh=70, cuda=True):
        outputs_encoder, hidden = self.encode(sentence_en)

        batch_size = sentence_en.data.shape[0]
        beam = [Beam(beam_size, cuda, self.ngpu) for _ in range(batch_size)]
        n_remaining_sents = batch_size
        # ipdb.set_trace()
        sentence_en = sentence_en.unsqueeze(0).repeat(beam_size, 1, 1).permute(1, 0, 2).contiguous().view(batch_size * beam_size, -1)
        outputs_encoder = outputs_encoder.unsqueeze(0).repeat(beam_size, 1, 1, 1).permute(2, 1, 0, 3).permute(1,0,2,3)\
            .contiguous().view(outputs_encoder.data.shape[0], outputs_encoder.data.shape[1]*beam_size, -1)
        h_n = hidden[0].unsqueeze(0).repeat(beam_size, 1, 1, 1).permute(2, 1, 0, 3).permute(1,0,2,3)\
            .contiguous().view(hidden[0].data.shape[0], hidden[0].data.shape[1]* beam_size, -1)
        c_n = hidden[1].unsqueeze(0).repeat(beam_size, 1, 1, 1).permute(2, 1, 0, 3).permute(1,0,2,3) \
            .contiguous().view(hidden[1].data.shape[0], hidden[1].data.shape[1] * beam_size, -1)
        hidden = (h_n, c_n)
        sentence_en_mask = torch.eq(sentence_en, Constants.PAD_INDEX)
        sentence_en_mask = sentence_en_mask.float().masked_fill_(sentence_en_mask, float('-inf'))
        if cuda:sentence_en_mask = sentence_en_mask.cuda(self.ngpu)

        for ii in range(self.target_max_len):
            pre_word = torch.stack([
                b.get_current_state() for b in beam if not b.done])
            pre_word = pre_word.view(-1, 1)
            pre_word = self.embedding_zh(Variable(pre_word)).permute(1, 0, 2)
            ht = hidden[0][-1, :, :]
            Ct, At = self.attention(outputs_encoder, ht, sentence_en_mask)
            input = torch.cat((Ct, pre_word), 2)
            output, hidden = self.decoder(input, hidden)
            output = self.fc(output[0])
            output = self.adaptiveSoftmax.log_prob(output)
            output = output.view(n_remaining_sents, beam_size, -1).contiguous()
            active = []
            active_id = -1
            for b in range(batch_size):
                if beam[b].done:continue
                active_id += 1
                if not beam[b].advance(output.data[active_id]):
                    active += [active_id]

            if len(active)==0:break
            active_idx = torch.LongTensor([k for k in active])
            if cuda: active_idx = active_idx.cuda(self.ngpu)
            # batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active_seq3d(seq, active_idx):
                new_size = list(seq.size())
                new_size[1] = len(active_idx) * beam_size
                return seq.view(new_size[0], n_remaining_sents, -1).index_select(1, active_idx).view(*new_size)


            def update_active_seq2d(seq, active_idx):
                new_size = list(seq.size())
                new_size[0] = len(active_idx) * beam_size
                # ipdb.set_trace()
                return seq.view(n_remaining_sents, -1).index_select(0, active_idx).view(*new_size)

            sentence_en_mask = update_active_seq2d(sentence_en_mask, active_idx)
            outputs_encoder = update_active_seq3d(outputs_encoder, active_idx)
            h_n = update_active_seq3d(hidden[0], active_idx)
            c_n = update_active_seq3d(hidden[1], active_idx)
            hidden = (h_n, c_n)

            n_remaining_sents = len(active)

        all_hyp = []
        for b in range(batch_size):
            all_hyp += [beam[b].get_best_hypothesis()]

        return all_hyp