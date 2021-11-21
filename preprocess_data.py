import os
import math
import numpy as np
from sklearn.model_selection import ShuffleSplit
from gensim.models import Word2Vec
from torch.utils.data import DataLoader

import dataset

if __name__ == '__main__':
    if __name__ == '__main__':
        # 数据预处理
        data_dir = './data/'
        ent2idx = dict(zip(dataset.ENTITIES, range(1, len(dataset.ENTITIES) + 1)))  # {标签：id}
        idx2ent = dict([(v, k) for k, v in ent2idx.items()])  # {id: 标签}

        docs = dataset.Documents(data_dir=data_dir)
        rs = ShuffleSplit(n_splits=1, test_size=20, random_state=2018)
        train_doc_ids, test_doc_ids = next(rs.split(docs))  # 切分训练集和测试集
        train_docs, test_docs = docs[train_doc_ids], docs[test_doc_ids]

        # 做滑动窗口取词
        num_cates = max(ent2idx.values()) + 1  # 15+1种类别，1为什么都不是
        sent_len = 64
        vocab_size = 3000
        emb_size = 100
        sent_pad = 10
        sent_extrator = dataset.SentenceExtractor(window_size=sent_len, pad_size=sent_pad)
        train_sents = sent_extrator(train_docs)
        test_sents = sent_extrator(test_docs)

        train_data = dataset.Dataset(train_sents, cate2idx=ent2idx)
        train_data.build_vocab_dict(vocab_size=vocab_size)

        test_data = dataset.Dataset(test_sents, word2idx=train_data.word2idx, cate2idx=ent2idx)
        vocab_size = len(train_data.word2idx)

        #
        w2v_train_sents = []
        for doc in docs:
            w2v_train_sents.append(list(doc.text))
        w2v_model = Word2Vec(w2v_train_sents)  # 对训练数据集做词向量

        w2v_embeddings = np.zeros((vocab_size, emb_size))  # [3000, 100]
        for char, char_idx in train_data.word2idx.items():
            if char in w2v_model.wv:
                w2v_embeddings[char_idx] = w2v_model.wv[char]

        seq_len = sent_len + 2 * sent_pad
        print(seq_len)

        # 拆分数据与标签
        train_X, train_y = train_data[:]
        print(type(train_data))
        print('train_X.shape', train_X.shape)
        print('train_y.shape', train_y.shape)

        test_X, test_y = test_data[:]
        print(type(test_data))
        print('test_X.shape', test_X.shape)
        print('test_y.shape', test_y.shape)

        train_dataloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
        for i, data in enumerate(train_dataloader):
            train_x, train_y = data[:]
            print(i, len(data), train_x.shape, train_y.shape)
