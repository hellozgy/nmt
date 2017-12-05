import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import time

embeds_dim = 128
epoch = 30
base_dir = '/users2/hpzhao/gyzhu/wmt17/'
corpus = base_dir +  'data_10w/train.zh.tok.bpe'
word2vec_path = base_dir + 'data_10w/word2vec{}_{}.zh'.format(embeds_dim, epoch)

def build_word_freq(fname='../data_10w/train.zh.tok.bpe.vocab'):
    word_freq = {}
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            word, freq = line.strip().split(' ')
            assert word not in word_freq
            word_freq[word] = int(freq)
    return word_freq


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        with open(self.dirname, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()
def build_word2vec():
    sentences = MySentences(corpus)
    model = gensim.models.Word2Vec(sentences=sentences, size=embeds_dim, min_count=1, workers=10, iter=epoch,
                                   hs=1, negative=0, seed=int(time.time()))
    model.save(word2vec_path)
    model.wv.save_word2vec_format(word2vec_path+'.c')

def test_word2vec():
    model2 = KeyedVectors.load_word2vec_format(word2vec_path+'.c', binary=False)
    model2.wv.most_similar(positive=['女', '男'], negative=['黑'])
def re_train(epoch=20):
    model = gensim.models.Word2Vec.load(word2vec_path)
    model.train(MySentences(corpus), epochs=epoch)

if __name__ == '__main__':
    build_word2vec()



