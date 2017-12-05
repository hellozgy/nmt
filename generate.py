#coding:utf-8
from config import opt
import models
import torch
from torch.autograd import Variable
from dataset import AIDataset
from torch.utils import data
from data_valid import mybleu
from dataset import Constants
from .modules import Beam,GlobalScorer
import ipdb

def generate(**kwargs):
    opt.parse(kwargs)
    opt.id = opt.model if opt.id is None else opt.id
    dataset = AIDataset('valid', opt.max_len)
    opt.input_size = dataset.vocab_size_en
    opt.output_size = dataset.vocab_size_zh
    model = getattr(models, opt.model)(opt)
    restore_file = './checkpoints/{}/{}'.format(opt.model,
                                                'checkpoint_last' if opt.restore_file is None else opt.restore_file)
    print('restore parameters from {}'.format(restore_file))
    model_file = torch.load(restore_file)
    model.load_state_dict(model_file['model'])
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=False, num_workers=1)
    fw = open('./data_valid/result_{}.txt'.format(opt.id), 'w', encoding='utf-8')
    model.eval()
    for ii, (sentence_en,sentence_zh) in enumerate(dataloader):
        sentence_en = Variable(sentence_en, volatile=True).long().cuda(opt.ngpu)
        predicts = model.translate_batch(sentence_en, opt.beam_size, opt.generate_max_len)
        for i in range(len(predicts)):
            sentence = ''.join([dataset.index2word_zh[index] for index in predicts[i]])
            fw.write(sentence + '\n')
        fw.flush()
    fw.close()
    model.train()
    score = float(mybleu(id))
    print('bleu:{}'.format(score))

def translate_batch(models, inputs, beam_size, generate_max_len):
    ngpu = models.get_device()
    n_model = len(models)
    batch_size = models.size(0)
    seq_len = models.size(1)
    for model in models:
        model.beam_encode(inputs,beam_size)
    sentence_en = models.repeat(1, beam_size).view(batch_size * beam_size, seq_len)
    sentence_en_mask = torch.eq(sentence_en, Constants.PAD_INDEX)
    sentence_en_mask = sentence_en_mask.float().masked_fill_(sentence_en_mask, float('-inf'))

    n_remaining_sents = batch_size
    beam = [Beam(beam_size, ngpu,
                 global_scorer=GlobalScorer(sentence_en[i*beam_size:(i+1)*beam_size,:])) for i in range(batch_size)]

    for ii in range(generate_max_len):
        if all((b.done() for b in beam)):
            break
        prev_y = torch.stack([
            b.getCurrentState() for b in beam if not b.done()]).view(-1, 1)
        logprobs = 0
        Align = 0
        for n, model in enumerate(models):
            logprob, At = model.beam_decoder(prev_y, sentence_en_mask)
            Align += At
            logprobs += logprob
        logprobs = logprobs.view(n_remaining_sents, beam_size, -1).contiguous()
        active = []
        active_id = -1
        re_idx = {}
        for b in range(batch_size):
            if beam[b].done(): continue
            active_id += 1
            done, idx = beam[b].advance(logprobs.data[active_id] / n_model, Align / n_model)
            if not done:
                active += [active_id]
                re_idx[active_id] = idx

        if len(active) == 0: break
        for model in models:
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

        sentence_en_mask = update_active_seq2d(sentence_en_mask, active_idx)
        n_remaining_sents = len(active)

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



if __name__=='__main__':
    import fire
    fire.Fire()