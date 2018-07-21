
class Config():
    base_dir = '.'
    data_dir = 'data/'
    src_vocab = 'vocab.en'
    dst_vocab = 'vocab.zh'
    src_train = 'train.en'
    src_dev = 'val.en'
    src_test = 'val.en'
    dst_train = 'train.zh'
    dst_dev = 'val.zh'
    dst_test= 'val.zh'

    ngpu = -1  # 指定gpu
    model = None
    input_size = -1
    output_size = -1
    Ls = 4
    Lt = 4
    embeds_size = 512  # 词向量维度
    hidden_size = 1024  # 隐层维度
    max_len = 30  # 句长截断上限
    label_smooth = 0.01
    dropout = 0.
    attn_general = True
    attn_concat = False
    id = None
    eval_iter = 1000  # 评估模型
    batch_size = 128
    epochs = 10
    adjust = True # 是否调整学习率
    lr = 1e-3
    limit_lr = 1e-6
    factor = 0.5 # 学习率调整速度
    beam_size = 0
    alpha = 1 # beam search长度惩罚
    beta = 0 # attention覆盖惩罚
    generate_max_len = 60  # 生成时最大句长
    restore_file=None
    save=True

    def parse(self, args):
        for k, v in args.items():
            assert hasattr(self, k), 'opt has no attribute:'+str(k)
            setattr(self, k, v)
        return self

    def show(self):
        keys = [k for k in dir(self) if not k.startswith('_') and not k.startswith('parse') and not k.startswith('show')]
        for key in keys:
            print("{}:{}".format(key, getattr(self, key)))

    def parseopt(self, opt):
        keys = [k for k in dir(opt) if not k.startswith('_') and not k.startswith('parse')
                and not k.startswith('show') and k not in set(['ngpu', 'id', 'beam_size', 'alpha', 'beta', 'restore_file'])]
        for key in keys:
            setattr(self, key, getattr(opt, key))



opt = Config()
