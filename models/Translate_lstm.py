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
            self.attn_Wa = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
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

    def forward(self, inputs, targets=None, target_len=30):
        ctx, hiddens = self.encode(inputs)
        predicts, batch_loss, aligns = self.decode(ctx, hiddens,inputs, targets, target_len)
        return predicts, batch_loss

    def encode(self, inputs, beam_size=0):
        '''
        :param inputs: (batch_size, seq_len)
        :param beam_size:
        :return:
        '''
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        inputs = self.embedding_en(inputs).permute(1, 0, 2)
        inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        ctx, hiddens = self.encoder(inputs)
        if beam_size == 0:
            return ctx, hiddens
        else:
            self.ctx = ctx.repeat(1, 1, beam_size).view(seq_len, batch_size * beam_size, -1)
            h_n = hiddens[0].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hiddens[0].size(-1))
            c_n = hiddens[1].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hiddens[1].size(-1))
            self.hiddens = [h_n, c_n]

    def decode(self, ctx, hiddens, inputs, targets, target_max_len):
        batch_size = inputs.size(0)
        prev_y = Variable(inputs.data.new(batch_size, 1).zero_().long())
        hiddens = hiddens
        aligns = []
        predicts = []
        loss = 0
        mask = torch.eq(inputs, Constants.PAD_INDEX).data
        mask = mask.float().masked_fill_(mask, -float('inf'))
        for i in range(target_max_len):
            _, At, output, hiddens = self.decode_step(prev_y, mask, hiddens, ctx,  beam_search=False)
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


    def translate_batch(self, sentence_en, beam_size=5, target_max_len=60, pr=1.1):
        #0.2177
        #0.2217
        ngpu = sentence_en.get_device()
        batch_size = sentence_en.size(0)
        seq_len = sentence_en.size(1)
        outputs_encoder, hidden = self.encode(sentence_en)
        sentence_en = sentence_en.repeat(1, beam_size).view(batch_size*beam_size, seq_len)
        sentence_en_mask = torch.eq(sentence_en, Constants.PAD_INDEX)
        sentence_en_mask = sentence_en_mask.float().masked_fill_(sentence_en_mask, float('-inf'))
        outputs_encoder = outputs_encoder.repeat(1, 1, beam_size).view(seq_len, batch_size*beam_size, -1)
        h_n = hidden[0].repeat(1, 1, beam_size).view(-1, batch_size*beam_size, hidden[0].size(-1))
        c_n = hidden[1].repeat(1, 1, beam_size).view(-1, batch_size*beam_size, hidden[1].size(-1))
        hidden = (h_n, c_n)

        n_remaining_sents = batch_size
        beam = [Beam(beam_size, ngpu,
                     hidden=[h_n[:,i*beam_size:(i+1)*beam_size,:],c_n[:,i*beam_size:(i+1)*beam_size,:]],
                     global_scorer=None, pr=pr) for i in range(batch_size)]

        predict_len = 0
        for ii in range(target_max_len):
            if all((b.done() for b in beam)):
                break
            pre_word = torch.stack([
                b.getCurrentState() for b in beam if not b.done()])
            hidden = [torch.cat([b.getCurrentHidden()[0] for b in beam if not b.done()], 1),
                      torch.cat([b.getCurrentHidden()[1] for b in beam if not b.done()], 1)]
            pre_word = pre_word.view(-1, 1)
            pre_word = self.embedding_zh(Variable(pre_word)).permute(1, 0, 2)
            ht = hidden[0][-1, :, :]
            # ipdb.set_trace()
            Ct, At = self.attention(outputs_encoder, ht, sentence_en_mask)
            # if ii==5:ipdb.set_trace()
            input = torch.cat((Ct, pre_word), 2)
            output, hidden = self.decoder(input, hidden)
            output = self.fc(output[0])
            output = self.adaptiveSoftmax.log_prob(output)
            output = output.view(n_remaining_sents, beam_size, -1).contiguous()
            active = []
            active_id = -1
            for b in range(batch_size):
                if beam[b].done(): continue
                active_id += 1
                if not beam[b].advance(output.data[active_id], At, [hidden[0][:,active_id*beam_size:(active_id+1)*beam_size,:],
                                                                    hidden[1][:, active_id * beam_size:(active_id+1) * beam_size, :]]):
                    active += [active_id]

            if len(active) == 0: break
            active_idx = torch.LongTensor([k for k in active])
            active_idx = active_idx.cuda(ngpu)

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
            # h_n = update_active_seq3d(hidden[0], active_idx)
            # c_n = update_active_seq3d(hidden[1], active_idx)
            # hidden = (h_n, c_n)

            n_remaining_sents = len(active)

            predict_len = ii
        print('predict:{}'.format(predict_len + 1))

        allHyps, allScores, allAttn = [], [], []
        for b in beam:
            scores, ks = b.sortFinished(beam_size)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:beam_size]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            allHyps.append(hyps)
            allScores.append(scores)
            allAttn.append(attn)
        # return allHyps, allScores, allAttn
        allHyps = [h[0] for h in allHyps]
        return allHyps