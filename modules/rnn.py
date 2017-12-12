import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from .layer_norm import LayerNorm
import ipdb




class LNGRUCell(nn.Module):
    # LayerNorm + GRU
    def __init__(self, input_size, hidden_size, bias=True, affine=True):
        super(LNGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 为了提升速度，这里将4个layer_horm合并成2个
        self.W = nn.Parameter(torch.FloatTensor(input_size, 3*hidden_size))
        self.U = nn.Parameter(torch.FloatTensor(hidden_size, 3*hidden_size))
        self.b = nn.Parameter(torch.zeros(3*hidden_size))
        self.W_ln = LayerNorm(3*hidden_size, affine=affine)
        self.U_ln = LayerNorm(3*hidden_size, affine=affine)

        self.reset_parameters()

    def reset_parameters(self):
        weight_data = torch.eye(self.hidden_size)
        weight_data = weight_data.repeat(1, 3)
        self.W.data.set_(weight_data)
        self.U.data.set_(weight_data)

    def forward(self, input, hx):
        assert input.dim()==2 and hx.dim()==2
        assert input.size(0)==hx.size(0)
        assert input.size(1)==self.input_size and hx.size(1)==self.hidden_size

        xw = torch.matmul(input, self.W)
        hu = torch.matmul(hx, self.U)
        xw = self.W_ln(xw)
        hu = self.U_ln(hu)
        xw = torch.split(xw, self.hidden_size, -1)
        hu = torch.split(hu, self.hidden_size, -1)

        z = F.sigmoid(xw[0] + hu[0] + self.b[:self.hidden_size])
        r = F.sigmoid(xw[1] + hu[1] + self.b[self.hidden_size:2*self.hidden_size])
        hx_ = F.tanh(r * hu[2] + xw[2] + self.b[2*self.hidden_size:])
        hx = (1 - z) * hx_ + z * hx
        return hx

class LNGRU(nn.Module):
    '''
    使用参见标准GRU
    '''
    def __init__(self, GRU, input_size, hidden_size, num_layers=1, bidirectory=False, bias=True, affine=True, dropout=0):
        super(LNGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectory = bidirectory
        self.ff_gru = nn.ModuleList([GRU(input_size, hidden_size, bias, affine)] +
                                    [GRU(hidden_size, hidden_size, bias, affine)] * (num_layers-1))
        self.back_gru = None
        if bidirectory:
            self.back_gru = nn.ModuleList([GRU(input_size, hidden_size, bias, affine)] +
                                          [GRU(hidden_size, hidden_size, bias, affine)] * (num_layers-1))
        self.dropout = dropout

    def forward(self, input, h0=None):
        assert input.dim() == 3, input.size()
        if h0 is None:
            h0 = Variable(input.data.new(self.num_layers * (2 if self.bidirectory else 1),
                                input.size(1), self.hidden_size).zero_().float())
        assert h0.dim() == 3
        assert input.size(2) == self.input_size and h0.size(2) == self.hidden_size, 'input:{},h0:{}'.format(str(input.size()), str(h0.size()))
        assert input.size(1) == h0.size(1)
        assert h0.size(0) == self.num_layers * (2 if self.bidirectory else 1)
        hiddens = []
        input_ff = [input[i] for i in range(input.size(0))]
        input_back = [input[i] for i in range(input.size(0))]
        seq_len = input.size(0)
        for layer in range(self.num_layers):
            hidden = h0[layer*(2 if self.bidirectory else 1),:,:]
            output_ff = []
            for timestep in range(seq_len):
                x = input_ff[timestep]
                hidden = self.ff_gru[layer](x, hidden)
                output_ff.append(F.dropout(hidden, self.dropout, self.training))
            input_ff = output_ff
            hiddens.append(hidden)

            if self.bidirectory:
                hidden = h0[1 + layer * (2 if self.bidirectory else 1), :, :]
                output_back = []
                for timestep in range(seq_len-1, -1, -1):
                    x = input_back[timestep]
                    hidden = self.back_gru[layer](x, hidden)
                    output_back.append(F.dropout(hidden, self.dropout, self.training))
                output_back.reverse()
                input_back = output_back
                hiddens.append(hidden)

        hiddens = torch.stack(hiddens)
        output = torch.stack(input_ff)
        if self.bidirectory:
            output = torch.cat([output, torch.stack(input_back)], 2)
        return output, hiddens

class  cGRUCell(nn.Module):
    def __init__(self, tgt_embds_size, hidden_size, bias=True, affine=True):
        super(cGRUCell, self).__init__()
        self.tgt_embds_size = tgt_embds_size
        self.hidden_size = hidden_size

        self.rec1 = nn.GRUCell(tgt_embds_size, hidden_size)
        self.rec2 = nn.GRUCell(hidden_size, hidden_size)

        self.U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.W = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.v = nn.Parameter(torch.FloatTensor(hidden_size, 1))

    def reset_parameters(self):
        nn.init.orthogonal(self.U)
        nn.init.orthogonal(self.W)

    def attention(self, ctx, key, sentence_mask):
        '''
        :param ctx: (seqlen, batch, dim)
        :param key: (batch, dim)
        :return:
        '''
        Uk = torch.matmul(key, self.U).repeat(ctx.size(0), 1)
        Wctx = torch.matmul(ctx.view(ctx.size(0)*ctx.size(1), ctx.size(2)), self.W)
        e = torch.matmul(F.tanh(Uk + Wctx), self.v)
        e = F.softmax(torch.cat(torch.split(e, ctx.size(1), 0), 1)+sentence_mask)
        attn = torch.bmm(e.unsqueeze(1), ctx.permute(1,0,2))
        return attn[:,0,:]

    def forward(self, hx, y, ctx, sentence_mask):
        _hx = self.rec1(y, hx)
        c = self.attention(ctx, _hx, sentence_mask)
        hx = self.rec2(c, _hx)
        return hx

class  cLSTMCell(nn.Module):
    def __init__(self, embds_size, hidden_size, ctx_size):
        super(cLSTMCell, self).__init__()
        self.embds_size = embds_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size

        self.rec1 = nn.LSTMCell(embds_size, hidden_size)
        self.rec2 = nn.LSTMCell(hidden_size + ctx_size, hidden_size)

    def forward(self, y_embeds, pre_hiddens, ctx, low_inputs):
        '''
        :param y_embeds: (batch, embeds_size)
        :param pre_hiddens: ((batch, hidden_size),(batch, hidden_size))
        :param ctx:(batch, ctx_size)
        :param low_inputs:(batch, hidden_size)
        :return:((batch, hidden_size),(batch, hidden_size))
        '''
        hx, cx = self.rec1(y_embeds, pre_hiddens)
        hx, cx = self.rec2(torch.cat([ctx, low_inputs], -1), (hx, cx))
        return (hx, cx)

if __name__=='__main__':
    #hx, y, ctx, sentence_mask
    rnn = cLSTMCell(3, 4)
    hx = Variable(torch.randn(2, 4))
    y = Variable(torch.randn(2, 3))
    ctx = Variable(torch.randn(3, 2, 4))
    sentence_mask = Variable(torch.randn(2, 3))
    hx = rnn(hx, y, ctx, sentence_mask)
    print(hx)








