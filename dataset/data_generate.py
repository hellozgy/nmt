#coding:utf-8
import os
import numpy as np
from .Constants import PAD_INDEX, EOS_INDEX, UNK_INDEX, PAD_WORD, EOS_WORD, UNK_WORD

# data_dir = '/users2/hpzhao/gyzhu/wmt17/data_10w/'
# base_dir = '/users2/hpzhao/gyzhu/nmt2/'
def word2index(file):
    res = {PAD_WORD:PAD_INDEX, EOS_WORD:EOS_INDEX, UNK_WORD:UNK_INDEX}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            s = line.rstrip().split()
            res[s[0]] = len(res)
    return res

def generate(opt):
    # word2index_en = word2index(os.path.join(data_dir, 'train.en.tok.tc.bpe.vocab'))
    # word2index_zh = word2index(os.path.join(data_dir, 'train.zh.tok.bpe.vocab'))
    word2index_en = word2index(os.path.join(opt.data_dir, opt.src_vocab))
    word2index_zh = word2index(os.path.join(opt.data_dir, opt.dst_vocab))
    np.savez_compressed(os.path.join(opt.base_dir, 'data_train/word2index.npz'),
                        word2index_en=word2index_en, word2index_zh=word2index_zh)
    subsets = ['train', 'valid', 'test']
    # enfiles = ['train.en.tok.tc.bpe', 'valid.en.tok.tc.bpe', 'test.en.tok.tc.bpe']
    # zhfiles = ['train.zh.tok.bpe', 'valid.zh.tok.bpe', 'valid.zh.tok.bpe']
    enfiles = [opt.src_train, opt.src_dev, opt.src_test]
    zhfiles = [opt.dst_train, opt.dst_dev, opt.dst_test]
    for subset, enfile, zhfile in zip(subsets, enfiles, zhfiles):
        ens = [];zhs = []
        with open(os.path.join(opt.data_dir, enfile), 'r', encoding='utf-8') as fen, \
                open(os.path.join(opt.data_dir + zhfile), 'r', encoding='utf-8') as fzh:
            for line_en, line_zh in zip(fen, fzh):
                en = [word2index_en.get(t, UNK_INDEX) for t in line_en.strip().split()]
                zh = [word2index_zh.get(t, UNK_INDEX) for t in line_zh.strip().split()]
                ens.append(en)
                zhs.append(zh)
        print("{} total sentences:{}".format(subset, len(ens)))
        save2file = os.path.join(opt.base_dir, 'data_{}/{}.npz'.format(subset, subset))
        np.savez_compressed(save2file, ens=ens, zhs=zhs)

def getdata(subset, opt):
    file = os.path.join(opt.base_dir, 'data_{}/{}.npz'.format(subset, subset))
    if not os.path.exists(file):
        generate(opt)

    data = np.load(file)
    word2index = np.load(os.path.join(opt.base_dir, 'data_train/word2index.npz'))
    return data['ens'], data['zhs'], word2index['word2index_en'].item(), word2index['word2index_zh'].item()

if __name__ == '__main__':
    ens, zhs, word2index_en, word2index_zh = getdata('valid')
    index2word_en = dict(zip(word2index_en.values(), word2index_en.keys()))
    index2word_zh = dict(zip(word2index_zh.values(), word2index_zh.keys()))
    print('lenght:{}'.format(len(ens)))

    en,zh = ens.tolist()[1170],zhs.tolist()[1170]
    print(' '.join(index2word_en.get(e) for e in en))
    print(' '.join(index2word_zh.get(e) for e in zh))