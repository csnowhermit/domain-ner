import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit
from gensim.models import Word2Vec

import utils
import config
import dataset
from model import BiLSTM_CRF

'''
    参考链接：https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
'''

if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia reported today that apple".split(),
        "B I O O O O B O O O B".split()
    )]

    word_to_ix = {}    # 字和字典中下标的对应关系 {字: 下标}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # # Check predictions before training
    # with torch.no_grad():
    #     precheck_sent = utils.prepare_sequence(training_data[0][0], word_to_ix)
    #     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #     print(model(precheck_sent))

    # 做个dataloader。若输入为文本，batch_size>1时，需手动重写
    train_dataloader = DataLoader(dataset=training_data, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)

    for i, batch_data in enumerate(train_dataloader):
        sentence, tags = batch_data[:]
        print("sentence:", sentence)    # 文本文字：list：[('georgia',), ('tech',), ('is',), ('a',), ('university',), ('in',), ('georgia',)]
        print("tags", tags)    # 标签：list：[('B',), ('I',), ('O',), ('O',), ('O',), ('O',), ('B',)]

        model.zero_grad()

        sentence_in = utils.prepare_sequence(sentence, word_to_ix)    # 转为字典下标表示: tensor([[11, 12, 13, 14, 15, 16, 11,  4,  5,  6,  7], [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]])
        targets = utils.prepare_sequence(tags, tag_to_ix)    # 标签转下标表示：tensor([[0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 0], [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2]])

        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward(loss.clone().detach())    # loss为张量，得加loss.clone().detach()参数转为标量后进行前向传播
        optimizer.step()

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    # for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        # for sentence, tags in training_data:
        #     # Step 1. Remember that Pytorch accumulates gradients.
        #     # We need to clear them out before each instance
        #     model.zero_grad()
        #
        #     # Step 2. Get our inputs ready for the network, that is,
        #     # turn them into Tensors of word indices.
        #     sentence_in = utils.prepare_sequence(sentence, word_to_ix)
        #     targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        #
        #     # Step 3. Run our forward pass.
        #     loss = model.neg_log_likelihood(sentence_in, targets)
        #
        #     # Step 4. Compute the loss, gradients, and update the parameters by
        #     # calling optimizer.step()
        #     loss.backward()
        #     optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = utils.prepare_sequence(training_data[0][0], word_to_ix)
        print(model(precheck_sent))
    # We got it!