import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit
from gensim.models import Word2Vec

import config
import dataset
from model import BiLSTM_CRF


if __name__ == '__main__':
    # 准备训练数据集
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

    # 准备词向量模型
    w2v_train_sents = []
    for doc in docs:
        w2v_train_sents.append(list(doc.text))
    w2v_model = Word2Vec(w2v_train_sents)  # 对训练数据集做词向量

    w2v_embeddings = np.zeros((vocab_size, emb_size))  # [3000, 100]
    for char, char_idx in train_data.word2idx.items():
        if char in w2v_model.wv:
            w2v_embeddings[char_idx] = w2v_model.wv[char]

    seq_len = sent_len + 2 * sent_pad    # 每个句子的长度

    # 做成DataLoader
    train_dataloader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)

    # 测试集直接用
    test_X, test_Y = test_data[:]
    # print('test_X.shape', test_X.shape)
    # print('test_Y.shape', test_Y.shape)

    # # 训练数据集：list(tuple(sentences_list, label_list))
    # training_data = [(
    #     "the wall street journal reported today that apple corporation made money".split(),
    #     "B I I I O O O B I O O".split()
    # ), (
    #     "georgia tech is a university in georgia".split(),
    #     "B I O O O O B".split()
    # )]

    # word_to_ix = {}    # 字和下标的对应关系
    # for sentence, tags in training_data:
    #     for word in sentence:
    #         if word not in word_to_ix:
    #             word_to_ix[word] = len(word_to_ix)

    # tag_to_ix = {"B": 0, "I": 1, "O": 2, config.START_TAG: 3, config.STOP_TAG: 4}    # 标签类别与下标的对应关系

    # model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model = BiLSTM_CRF(vocab_size, ent2idx, seq_len, config.HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(config.total_epoch):
        for i, batch_data in enumerate(train_dataloader):
            x, y = batch_data[:]
            model.zero_grad()
            loss = model.neg_log_likelihood(x, y)
            # loss = model(x)

            loss.backward()
            optimizer.step()

        # 训练完一轮后，输入测试集看下效果
        test_result = model(test_X)
        print("test_result:", test_result)


    # # Make sure prepare_sequence from earlier in the LSTM section is loaded
    # for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    #     for sentence, tags in training_data:
    #         # 1.清空梯度
    #         model.zero_grad()
    #
    #         # 2.准备输入数据及标签
    #         sentence_in = utils.prepare_sequence(sentence, word_to_ix)    # 输入数据
    #         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)    # 标签
    #
    #         # Step 3. Run our forward pass.
    #         loss = model.neg_log_likelihood(sentence_in, targets)
    #
    #         # Step 4. Compute the loss, gradients, and update the parameters by
    #         # calling optimizer.step()
    #         loss.backward()
    #         optimizer.step()
    #
    # # Check predictions after training
    # with torch.no_grad():
    #     precheck_sent = utils.prepare_sequence(training_data[0][0], word_to_ix)
    #     print(model(precheck_sent))
    # # We got it!