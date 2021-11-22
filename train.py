import numpy as np
import torch
import torch.optim as optim

import config
import dataset
from model import BiLSTM_CRF


if __name__ == '__main__':
    model = BiLSTM_CRF(dataset.ENTITIES)
    optimizer = optim.Adam(model.parameters())

    train_dataloader = dataset.getDataLoader()

    for epoch in range(config.total_epoch):
        for i, batch_data in enumerate(train_dataloader):
            model.zero_grad()
            x, y = batch_data[:]    # x [batch_size, 1, seq_len], y [batch_size, 1, seq_len, 1]
            print(type(x))





