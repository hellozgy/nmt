import torch
from torch import nn
from torch.autograd import Variable
import math
from dataset import Constants
import torch.nn.functional as F
import ipdb

class AdaptiveSoftmax(nn.Module):
    def __init__(self, input_size, cutoff):
        super(AdaptiveSoftmax, self).__init__()
        
        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1
        
        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()
        
        for i in range(len(cutoff) - 1):
            seq = nn.Sequential(
                nn.Linear(input_size, input_size // 4 ** i, False),
                nn.Linear(input_size // 4 ** i, cutoff[i + 1] - cutoff[i], False)
            )

            self.tail.append(seq)
        self.reset()
        
    def reset(self):
        init = 1.0 / math.sqrt(self.input_size)
        self.head.weight.data.uniform_(-init, init)

        for tail in self.tail:
            tail[0].weight.data.uniform_(-init, init)
            tail[1].weight.data.uniform_(-init, init)

    # 判断在频率低的词里面有目标词的batch id
    def set_target(self, target):
        id = []
        
        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1])).cpu().data.long()

            if mask.sum() > 0:
                id.append(Variable(mask.nonzero().squeeze(1)).cuda(target.get_device()))
                
            else:
                id.append(None)
        return id

    def forward(self, input, target):
        id = self.set_target(target)
        output = [self.head(input)]
        
        for i in range(len(id)):
            if id[i] is not None:
                output.append(self.tail[i](input.index_select(0, id[i])))

            else:
                output.append(None)
                
        return output
        

    def log_prob(self, input):
        head_out = self.head(input)
        batch_size = head_out.size(0)
        prob = torch.zeros(batch_size, self.cutoff[-1]).cuda(input.get_device())

        lsm_head = F.log_softmax(head_out, -1)
        prob.narrow(1, 0, self.output_size).add_(lsm_head.narrow(1, 0, self.output_size).data)
        for i in range(len(self.tail)):
            pos = self.cutoff[i]
            i_size = self.cutoff[i + 1] - pos
            buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1)
            buffer = buffer.expand(batch_size, i_size)
            lsm_tail = F.log_softmax(self.tail[i](input), -1)
            prob.narrow(1, pos, i_size).copy_(buffer.data).add_(lsm_tail.data)

        return Variable(prob)


class AdaptiveLoss(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        
        self.cutoff = cutoff
        self.criterions = nn.ModuleList()
        
        for i in self.cutoff:
            self.criterions.append(nn.KLDivLoss(size_average=False))
            # self.criterions.append(nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD_INDEX))

            
    def remap_target(self, target): 
        '''
            映射target到各个分区
        '''
        new_target = [target.clone()]
        
        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i
            
            if mask.sum() > 0:
                new_target.append(target[mask].add(-self.cutoff[i]))
                
            else:
                new_target.append(None)
                
        return new_target

    def label_smooth(self, inputs, targets):
        label_smooth = 0.1

        for i in range(len(targets)):
            if inputs[i] is None:continue
            true_dist = inputs[i].data.clone()
            true_dist.fill_(label_smooth / (len(true_dist[0])-2))
            true_dist.scatter_(1, targets[i].data.unsqueeze(1), 1-label_smooth)
            true_dist[:, Constants.PAD_INDEX] = 0
            if i == 0:
                mask = torch.nonzero(targets[i].data==Constants.PAD_INDEX)
                if len(mask)>0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.)
            targets[i]=true_dist
        return targets
    
    def forward(self, input, target):
        '''
            :input a list of size len(self.cutoff) 
            每个input[i]是logsoftmax后的概率
            :target batch_size
            return :loss
        '''
        batch_size = input[0].size(0)
        target = self.remap_target(target.data)
        target = self.label_smooth(input, target)
        output = 0.0
    
        for i in range(len(input)):
            if input[i] is not None:
                assert(target[i].min() >= 0 and target[i].max() <= input[i].size(1))
                criterion = self.criterions[i]
                output += criterion(F.log_softmax(input[i], dim=-1), Variable(target[i]))
                
        output /= batch_size
        
        return output