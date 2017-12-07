#coding:utf-8
from config import opt
import models
import torch
from torch.autograd import Variable
from dataset import AIDataset
from torch.utils import data
from data_valid import mybleu
from dataset import Constants
from modules import Beam,GlobalScorer
import ipdb

def generate(**kwargs):
    opt.parse(kwargs)
    assert opt.id is not None
    dataset = AIDataset('valid', opt.max_len)
    opt.input_size = dataset.vocab_size_en
    opt.output_size = dataset.vocab_size_zh
    _models = []
    for model_name, model_path in opt.restore_file.items():
        model = getattr(models, model_name)(opt)
        model_file = './checkpoints/{}/{}'.format(model_name, model_path)
        model_file = torch.load(model_file)
        model.load_state_dict(model_file['model'], strict=False)
        model.cuda(opt.ngpu)
        model.eval()
        _models.append(model)
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=False, num_workers=1)
    fw = open('./data_valid/result_{}.txt'.format(opt.id), 'w', encoding='utf-8')
    for ii, (sentence_en,sentence_zh) in enumerate(dataloader):
        sentence_en = Variable(sentence_en, volatile=True).long().cuda(opt.ngpu)
        predicts = translate_batch(_models, sentence_en, opt.beam_size, opt.generate_max_len)
        for i in range(len(predicts)):
            sentence = ''.join([dataset.index2word_zh[index] for index in predicts[i]])
            fw.write(sentence + '\n')
        fw.flush()
    fw.close()
    score = float(mybleu(opt.id))
    print('bleu:{}'.format(score))

def translate_batch(_models, inputs, beam_size, generate_max_len):
    n_model = len(_models)
    ngpu = inputs.get_device()
    batch_size = inputs.size(0)
    seq_len = inputs.size(1)
    for model in _models:
        model.encode(inputs, beam_size)
    inputs = inputs.repeat(1, beam_size).view(batch_size * beam_size, seq_len)
    mask = torch.eq(inputs, Constants.PAD_INDEX).data
    mask = mask.float().masked_fill_(mask, float('-inf'))

    n_remaining_sents = batch_size
    beam = [Beam(beam_size, ngpu,
                 global_scorer=GlobalScorer(inputs[i*beam_size:(i+1)*beam_size,:])) for i in range(batch_size)]

    length = 0
    # ipdb.set_trace()
    for ii in range(generate_max_len):
        # if ii<=3:
        #     ipdb.set_trace()
        if all((b.done() for b in beam)):
            break
        prev_y = torch.stack([
            b.getCurrentState() for b in beam if not b.done()]).view(-1, 1)
        logprobs = 0
        Align = 0
        for n, model in enumerate(_models):
            logprob, At, _, _ = model.decode_step(Variable(prev_y), mask, beam_search=True)
            Align += At
            logprobs += logprob
        logprobs = logprobs.view(n_remaining_sents, beam_size, -1).contiguous()
        active = []
        active_id = -1
        re_idx = []
        # if ii<=3:
        #     ipdb.set_trace()
        for b in range(batch_size):
            if beam[b].done(): continue
            active_id += 1
            done, idx = beam[b].advance(logprobs.data[active_id] / n_model, Align / n_model)

            if not done:
                active += [active_id]
                re_idx.append(idx + (active_id * beam_size))
        if len(active) == 0: break
        re_idx = Variable(torch.cat(re_idx, 0))
        for model in _models:
            model.update_state(re_idx)
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

        mask = update_active_seq2d(mask, active_idx)
        n_remaining_sents = len(active)
        length = ii
    print('length:{}'.format(length))
    for model in _models:
        model.hiddens = None
        model.ctx = None

    # ipdb.set_trace()
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
    # ipdb.set_trace()
    allHyps = [h[0] for h in allHyps]
    return allHyps

'''
python generate.py generate --batch_size 128 --ngpu=4 \
--beam-size=1 --restore-file '{"Translate_lstm":"checkpoint4_score0.1442"}'
'''

if __name__=='__main__':
    import fire
    fire.Fire()