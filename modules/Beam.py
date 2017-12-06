import torch
from dataset import Constants
import ipdb


class Beam(object):
    # refer to OpenNMT
    def __init__(self, beam_size, ngpu, cuda=True, global_scorer=None):
        self.beam_size = beam_size
        self.ngpu = ngpu

        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(beam_size).zero_().cuda(ngpu)
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(beam_size).fill_(Constants.PAD_INDEX).cuda(ngpu)]

        # Has EOS topped the beam yet.
        self._eos = Constants.EOS_INDEX
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []

        # Information for global scoring.
        self.globalScorer = global_scorer
        self.globalState = {}


    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        if len(self.prevKs)==0:
            return [torch.LongTensor(self.beam_size).fill_(0).cuda(self.ngpu)]
        else:
            return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        '''
        :param wordLk: (beam_size, n_words)
        :param attnOut: (beam_size, seq_len)
        :param hidden: (h_n,c_n)
        :return:
        '''
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = float('-inf')
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.beam_size, 0, True, True)


        self.allScores.append(self.scores)
        self.scores = bestScores


        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        # ipdb.set_trace()
        # self.hidden=[hidden[0].index_select(1, prevK), hidden[1].index_select(1, prevK)]
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        if self.globalScorer is not None:
            self.globalScorer.updateGlobalState(self)

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]/(len(self.nextYs) - 1)
                if self.globalScorer is not None:
                    globalScores = self.globalScorer.score(self, self.scores,  i)
                    s = globalScores
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == Constants.EOS_INDEX:
            self.eosTop = True
        return self.done(), prevK

    def done(self):
        return self.eosTop and len(self.finished) >= self.beam_size

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            while len(self.finished) < minimum:
                s = self.scores[i]/(len(self.nextYs) - 1)
                if self.globalScorer is not None:
                    globalScores = self.globalScorer.score(self, self.scores, i)
                    s = globalScores
                self.finished.append((s, len(self.nextYs) - 1, i))

        try:
            self.finished.sort(key=lambda a: -a[0])
        except Exception as e:
            ipdb.set_trace()
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            if self.nextYs[j+1][k] != Constants.EOS_INDEX:
                hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GlobalScorer(object):
    # refer to GNMT
    def __init__(self, src, alpha=1, beta=0):
        self.alpha = alpha
        self.beta = beta
        self.src = src
        sentence_en_mask = torch.eq(src, Constants.PAD_INDEX)
        self.mask = sentence_en_mask.float().masked_fill_(sentence_en_mask, float('inf')).data


    def score(self, beam, logprobs, i):
        cov = beam.globalState["coverage"]
        pen = self.beta * (torch.min(cov.data+self.mask, cov.data.clone().fill_(1.0)).mean(1).log())
        l_term = (len(beam.nextYs)-1)**self.alpha
        return logprobs[i] / l_term + pen[i]

    def updateGlobalState(self, beam):
        if len(beam.prevKs) == 1:
            beam.globalState["coverage"] = beam.attn[-1]
        else:
            beam.globalState["coverage"] = beam.globalState["coverage"] \
                .index_select(0, beam.prevKs[-1]).add(beam.attn[-1])
