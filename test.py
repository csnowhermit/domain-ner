import os
import json
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import config
import dataset
from model import BiLSTM_CRF


if __name__ == '__main__':
    # 加载word2idx结构
    with open(config.word2idx_path, 'r', encoding='utf-8') as f:
        word2idx = f.read()
        word2idx = eval(word2idx)

    # print(type(word2idx))
    # print(word2idx.get('糖', 0))

    # 初始化模型
    model = BiLSTM_CRF(dataset.ENTITIES)
    # model.load_state_dict(torch.load("./checkpoint/***.pth"))
    print(model)

    input_str = input("请输入文本: ")
    input_vec = [word2idx.get(i, 0) for i in input_str]
    # convert to tensor
    sentences = torch.tensor(input_vec).view(1, -1)
    _, paths = model(sentences)    # paths为每个字的类别

    print("NERresult: ", paths)
