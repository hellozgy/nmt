#coding:utf-8
import os
from config import opt
import models
import torch
from torch.autograd import Variable
from dataset import AIDataset
from dataset import Constants
from torch.utils import data
import tqdm
from data_valid import mybleu
import shutil
import ipdb

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def eval(opt, model, best_score, checkpoint_id, epoch, batch):
    dataset = AIDataset('valid', opt.max_len, opt)
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=False,num_workers=1)
    total_loss = 0
    step = 0
    fw = open('./data_valid/result_{}.txt'.format(opt.id), 'w', encoding='utf-8')
    model.eval()
    print('start eval')
    for sentence_en, sentence_zh in dataloader:
        step += 1
        sentence_en = Variable(sentence_en, volatile=True).long().cuda(opt.ngpu)
        sentence_zh = Variable(sentence_zh, volatile=True).long().cuda(opt.ngpu)
        predicts, loss = model(sentence_en, sentence_zh, target_len=opt.max_len)
        total_loss += loss.data[0]
        predicts = [p.view(-1).data.tolist() for p in predicts]
        for i in range(len(predicts[0])):
            sentence = ''
            for j in range(opt.max_len):
                index = predicts[j][i]
                if index == Constants.EOS_INDEX: break
                sentence += dataset.index2word_zh[index]
            fw.write(sentence + '\n')
        fw.flush()
    fw.flush()
    fw.close()
    model.train()
    score = float(mybleu(opt.id))
    total_loss = total_loss / step

    msg = 'eval_loss:{:,.4f} bleu{:,.4f}_best{:,.4f}_epoch={},batch={}'.format(total_loss, score, best_score, epoch, batch)
    print(msg)
    with open('./log/{}.txt'.format(opt.id), 'a', encoding='utf-8') as fw:
        fw.write(msg + '\n')
    if opt.save:
        torch.save({'model': model.state_dict(), 'checkpoint_id': checkpoint_id, 'score': score, 'opt':opt},
                   './checkpoints/{}/checkpoint{}_score{}'.format(opt.id, checkpoint_id, score))
        shutil.copy('./checkpoints/{}/checkpoint{}_score{}'.format(opt.id, checkpoint_id, score),
                    './checkpoints/{}/checkpoint_last'.format(opt.id))
        if score > best_score:
            best_score = score
            shutil.copy('./checkpoints/{}/checkpoint_last'.format(opt.id),
                        './checkpoints/{}/checkpoint_best'.format(opt.id))
    return best_score, checkpoint_id + 1

def train(**kwargs):
    opt.parse(kwargs)
    opt.id = opt.model if opt.id is None else opt.id
    save2path = './checkpoints/{}/'.format(opt.id)
    if  not os.path.exists(save2path): os.system('mkdir {}'.format(save2path))
    assert opt.ngpu >= 0
    dataset_train = AIDataset('train', opt.max_len, opt)
    opt.input_size = dataset_train.vocab_size_en
    opt.output_size = dataset_train.vocab_size_zh
    print('vocab_en:{} vocab_zh:{}'.format(opt.input_size, opt.output_size))

    model = getattr(models, opt.model)(opt)
    best_score = 0.
    checkpoint_id = 1
    # restore_file = './checkpoints/{}/{}'.format(opt.model,
    #                 'checkpoint_last' if opt.restore_file is None else opt.restore_file)
    # if os.path.exists(restore_file):
    #     print('restore parameters from {}'.format(restore_file))
    #     model_file = torch.load(restore_file)
    #     if 'opt' in model_file: opt.parseopt(model_file['opt'])
    #     model = getattr(models, opt.model)(opt)
    #     model.load_state_dict(model_file['model'], strict=False)
    #     checkpoint_id = int(model_file['checkpoint_id']) + 1
    #     best_score = float(model_file['score'])
    model.cuda(opt.ngpu)

    optimizer = model.get_optimizer(opt.lr)
    dataloader_train = data.DataLoader(
        dataset=dataset_train, batch_size=opt.batch_size,
        shuffle=True, num_workers=1, drop_last=False)
    for epoch in range(opt.epochs):
        loss = 0
        batch = 0
        for _, (sentence_en, sentence_zh) in tqdm.tqdm(enumerate(dataloader_train)):
            sentence_en = Variable(sentence_en).long().cuda(opt.ngpu)
            sentence_zh = Variable(sentence_zh).long().cuda(opt.ngpu)
            batch += 1
            optimizer.zero_grad()
            _, batch_loss = model(sentence_en, sentence_zh, target_len=opt.max_len)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()
            loss += batch_loss.data[0]
            if batch % 100 == 0:
                print('{}:loss:{}'.format(batch, loss / batch))

            if batch % opt.eval_iter == 0:
                best_score, checkpoint_id = eval(opt, model, best_score, checkpoint_id, epoch + 1, batch)
        best_score, checkpoint_id = eval(opt, model, best_score, checkpoint_id, epoch + 1, batch)

if __name__=='__main__':
    import fire
    fire.Fire()

