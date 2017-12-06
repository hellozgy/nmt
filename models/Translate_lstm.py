import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import Constants
from modules import *
from .BasicModule import BasicModule


class Translate_lstm(BasicModule):
    def __init__(self, opt):
        super(Translate_lstm, self).__init__(opt)
        self.embedding_en = nn.Embedding(num_embeddings=self.vocab_size_en, embedding_dim=self.embeds_dim, padding_idx=Constants.PAD_INDEX)
        self.embedding_zh = nn.Embedding(num_embeddings=self.vocab_size_zh, embedding_dim=self.embeds_dim, padding_idx=Constants.PAD_INDEX)
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.layers, bidirectional=False)
        self.decoder = nn.LSTM(input_size=self.embeds_dim + self.hidden_size, hidden_size=self.hidden_size,
                               num_layers=self.layers, bidirectional=False)

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.pt = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.Wa = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.Softmax()
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.embeds_dim),
            nn.BatchNorm1d(self.embeds_dim),
            nn.Tanh()
        )
        cutoff = [3000, 20000, self.vocab_size_zh]
        self.adaptiveSoftmax = AdaptiveSoftmax(self.embeds_dim, cutoff=cutoff)
        self.loss_function = AdaptiveLoss(cutoff)


    def encode(self, sentence_en):
        input = self.embedding_en(sentence_en).permute(1, 0, 2)
        input = self.dropout(input)
        outputs, hidden = self.encoder(input)
        return outputs, hidden


    def attention(self, hs, ht, sentence_en_mask):
        '''
               hs:(seqlen, batch_size, hidden_size)
               ht:(batch_size, hidden_size)
               return :(1,batch_size, hidden_size)
           '''
        seq_len = len(hs)
        ht = torch.unsqueeze(ht, 0)
        hs = hs.permute(1, 0, 2)  # (batch_size, seqlen, hidden_size)
        ht = ht.permute(1, 2, 0)  # (batch_size, hidden_size, 1)
        pt = seq_len * self.pt(ht[:,:,0])
        pt = torch.cat([torch.exp(-((s-pt)**2)/18) for s in range(seq_len)] ,1)
        At = torch.matmul(hs, self.Wa(ht[:,:,0]).unsqueeze(2)).view(-1, seq_len)
        At = At + sentence_en_mask
        At = self.softmax(At)  # (batch_size,1, seqlen)
        At = At * pt
        At = At.unsqueeze(1)
        attn = torch.bmm(At, hs)  # (batch_size,1, hidden_size)
        return attn.permute(1, 0, 2), At[:,0,:] # (1, batch_size, hidden_size)

    def decode(self, outputs_encoder, hidden_encoder, sentence_en, sentence_zh=None, target_max_len=30):
        batch_size = sentence_en.size(0)
        output = Variable(sentence_en.data.new(batch_size, self.embeds_dim).zero_().float())
        output = output.view(1, -1, self.embeds_dim)
        hidden = hidden_encoder
        # batch_loss = None
        aligns = []
        predicts = []
        teacher = True
        loss = 0
        sentence_en_mask = torch.eq(sentence_en, Constants.PAD_INDEX)
        sentence_en_mask = sentence_en_mask.float().masked_fill_(sentence_en_mask, float('-inf'))

        target_max_len = sentence_zh.size(1) if self.training else target_max_len
        for i in range(target_max_len):
            ht = hidden[0][-1, :, :]
            Ct, At = self.attention(outputs_encoder, ht, sentence_en_mask)
            aligns.append(At)
            input = torch.cat((Ct, output), 2)
            output, hidden = self.decoder(input, hidden)
            output = self.fc(output[0])
            target = sentence_zh[:, i].contiguous().view(-1)
            o = self.adaptiveSoftmax(output, target)
            loss += self.loss_function(o, target)
            if self.training:
                if teacher:
                    output = sentence_zh[:, i].unsqueeze(1)
                else:
                    predict = self.adaptiveSoftmax.log_prob(output)
                    predict = predict.topk(1, dim=1)[1]
                    output = predict
            else:
                predict = self.adaptiveSoftmax.log_prob(output)
                predict = predict.topk(1, dim=1)[1]
                predicts.append(predict)
                output = predict
            output = self.embedding_zh(output).permute(1,0,2)
        return predicts, loss/target_max_len, aligns

    def forward(self, inputs, target_sentence, target_len=30, label_smooth=0):
        outputs_encoder, hidden_encoder = self.encode(inputs)
        predicts, batch_loss, aligns = self.decode(outputs_encoder, hidden_encoder,
                                                   inputs, target_sentence, target_len)
        return predicts, batch_loss

    def beam_encode(self, inputs, beam_size=0):
        '''
        :param inputs: (batch_size, seq_len)
        :param beam_size:
        :return:
        '''
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        ctx, hidden = self.encode(inputs)
        self.ctx = ctx.repeat(1, 1, beam_size).view(seq_len, batch_size * beam_size, -1)
        h_n = hidden[0].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hidden[0].size(-1))
        c_n = hidden[1].repeat(1, 1, beam_size).view(-1, batch_size * beam_size, hidden[1].size(-1))
        self.hidden = [h_n, c_n]
        return ctx, hidden

    def beam_decode(self, prev_y, mask):
        '''
        :param prev_y:(batch_size*beam_size, 1)
        :param mask:(batch_size*beam_size, seq_len)
        :return:
        logprob:(batch_size*beam_size, output_size)
        At:(batch_size*beam*size, seq_len)
        '''
        hidden = self.hidden
        prev_y = self.embedding_zh(Variable(prev_y)).permute(1, 0, 2)
        key = hidden[0][-1, :, :]
        Ct, At = self.attention(self.ctx, key, mask)
        input = torch.cat((Ct.unsqueeze(0), prev_y), 2)
        output, hiddens = self.decoder(input, hidden)
        output = self.fc(output[0])
        logprob = self.adaptiveSoftmax.log_prob(output)
        return logprob, At

    def update_state(self, re_idx):
        '''update hidden and ctx'''
        self.ctx = self.ctx.index_select(1, re_idx)
        self.hidden = self.hidden.index_select(1, re_idx)


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