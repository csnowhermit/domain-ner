import torch
import torch.nn as nn

import config
import dataset

def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)

class BiLSTM_CRF(nn.Module):
    def __init__(self, entities):
        super(BiLSTM_CRF, self).__init__()

        # 加上开始结束符号
        self.entities = entities
        self.entities.append(config.START_TAG)
        self.entities.append(config.STOP_TAG)

        self.tag_size = len(self.entities)
        self.tag_map = dict(zip(self.entities, range(0, len(self.entities))))  # {标签：id}，id下标从0开始

        self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.CRF = nn.Parameter(torch.randn(self.tag_size, self.tag_size))

        self.CRF.data[:, self.tag_map[config.START_TAG]] = -1000
        self.CRF.data[self.tag_map[config.STOP_TAG], :] = -1000

        self.hidden2tag = nn.Linear(config.hidden_dim, self.tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, config.batch_size, config.hidden_dim // 2),
                torch.randn(2, config.batch_size, config.hidden_dim // 2))

    def __get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        embeddings = self.word_embeddings(sentence).view(config.batch_size, length, config.embedding_dim)  # [1, 7, 100]

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)  # lstm_out [1, 17, 128], hidden []
        lstm_out = lstm_out.view(config.batch_size, -1, config.hidden_dim)
        logits = self.hidden2tag(lstm_out)
        return logits

    def real_path_score_(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_map[config.START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i], tags[i + 1]] + feat[tags[i + 1]]
        score = score + self.transitions[tags[-1], self.tag_map[config.STOP_TAG]]
        return score

    '''
        caculate real path score
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]
        
        Score = Emission_Score + Transition_Score
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])
    '''
    def real_path_score(self, logits, label):
        score = torch.zeros(1)
        label = torch.cat([torch.tensor([self.tag_map[config.START_TAG]], dtype=torch.long), label])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag_map[config.STOP_TAG]]
        return score

    """
        caculate total score
        
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]
        
        SCORE = log(e^S1 + e^S2 + ... + e^SN)
    """
    def total_score(self, logits, label):
        obs = []
        previous = torch.full((1, self.tag_size), 0)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map[config.STOP_TAG]]
        # caculate total_scores
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def neg_log_likelihood(self, sentences, tags, length):
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        real_path_score = torch.zeros(1)
        total_score = torch.zeros(1)
        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng]
            tag = tag[:leng]
            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit, tag)
        # print("total score ", total_score)
        # print("real score ", real_path_score)
        return total_score - real_path_score

    def forward(self, sentences, lengths=None):
        """
        :params sentences sentences to predict
        :params lengths represent the ture length of sentence, the default is sentences.size(-1)
        """
        sentences = torch.tensor(sentences, dtype=torch.long)
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]
            score, path = self.__viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def __viterbi_decode(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size())
        backpointers = torch.zeros(logits.size(), dtype=torch.long)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi

    def __viterbi_decode_v1(self, logits):
        init_prob = 1.0
        trans_prob = self.transitions.t()
        prev_prob = init_prob
        path = []
        for index, logit in enumerate(logits):
            if index == 0:
                obs_prob = logit * prev_prob
                prev_prob = obs_prob
                prev_score, max_path = torch.max(prev_prob, -1)
                path.append(max_path.cpu().tolist())
                continue
            obs_prob = (prev_prob * trans_prob).t() * logit
            max_prob, _ = torch.max(obs_prob, 1)
            _, final_max_index = torch.max(max_prob, -1)
            prev_prob = obs_prob[final_max_index]
            prev_score, max_path = torch.max(prev_prob, -1)
            path.append(max_path.cpu().tolist())
        return prev_score.cpu().tolist(), path

if __name__ == '__main__':
    model = BiLSTM_CRF(dataset.ENTITIES)
    print(model)