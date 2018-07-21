from torch.utils import data
from .data_generate import getdata
import numpy as np
from .Constants import PAD_INDEX, EOS_INDEX

class AIDataset(data.Dataset):
    def __init__(self, subset, max_len, opt):
        '''
        :param subset: train,valid,test
        '''
        self.subset = subset
        self.max_len = max_len
        self.ens, self.zhs, self.word2index_en, self.word2index_zh = getdata(subset, opt)
        self.index2word_en = dict(zip(self.word2index_en.values(), self.word2index_en.keys()))
        self.index2word_zh = dict(zip(self.word2index_zh.values(), self.word2index_zh.keys()))
        self.vocab_size_en = len(self.word2index_en)
        self.vocab_size_zh = len(self.word2index_zh)

    def __getitem__(self, index):
        en = self.ens[index][:self.max_len-1]+[EOS_INDEX]
        en = en + [PAD_INDEX] * (self.max_len - len(en))
        en.reverse()
        zh = self.zhs[index][:self.max_len-1]+[EOS_INDEX]
        zh = zh + [PAD_INDEX] * (self.max_len - len(zh))
        return np.asarray(en), np.asarray(zh)

    def __len__(self):
        return len(self.ens)