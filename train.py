import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import config
import dataset
from model import BiLSTM_CRF


if __name__ == '__main__':
    model = BiLSTM_CRF(dataset.ENTITIES)
    optimizer = optim.Adam(model.parameters())

    train_dataloader, test_dataloader = dataset.getDataLoader()
    train_loss = 99999    # 记录当前train loss
    eval_loss = 99999    # 记录当前eval loss

    m_progress = tqdm(range(0, config.total_epoch))
    for epoch in m_progress:
        model.train()
        for i, batch_data in enumerate(train_dataloader):
            model.zero_grad()
            x, y = batch_data[:]    # x [batch_size, 1, seq_len], y [batch_size, 1, seq_len, 1]
            x = x.detach().cpu().numpy().reshape(-1, config.seq_len)
            y = y.detach().cpu().numpy().reshape(-1, config.seq_len)

            # 数据要做成以下格式：每个方括号是一句话的序列。这里要做成整体的Tensor，而不是每一项单独一个Tensor
            # 长度：length(句子1长度, 句子2长度, 句子3长度...)
            length = torch.tensor(tuple(torch.full([x.shape[0]], x.shape[1])), dtype=torch.long)
            # print(length)

            # 训练数据：sentence([下标序列], [下标序列], [下标序列]...)
            x = torch.tensor(tuple(x), dtype=torch.long)
            # print(x)

            # 标签：tag([id序列], [id序列], [id序列]...)
            y = torch.tensor(tuple(y), dtype=torch.long)
            # print(y)

            loss = model.neg_log_likelihood(x, y, length)

            loss.backward()
            optimizer.step()
        curr_train_loss = loss.cpu().tolist()[0]
        # 按训练集loss的更新保存
        if curr_train_loss < train_loss:
            train_loss = curr_train_loss    # 更新保存的loss


        # 开始eval：对于验证集，算所有的平均损失
        model.eval()
        eval_losses = []
        for i, test_batch in enumerate(test_dataloader):
            x, y = test_batch[:]  # x [batch_size, 1, seq_len], y [batch_size, 1, seq_len, 1]
            x = x.detach().cpu().numpy().reshape(-1, config.seq_len)
            y = y.detach().cpu().numpy().reshape(-1, config.seq_len)

            # 按训练数据的格式要求做
            length = torch.tensor(tuple(torch.full([x.shape[0]], x.shape[1])), dtype=torch.long)
            x = torch.tensor(tuple(x), dtype=torch.long)
            y = torch.tensor(tuple(y), dtype=torch.long)

            loss = model.neg_log_likelihood(x, y, length)
            eval_losses.append(loss.cpu().tolist()[0])

        # 算验证集的的平均损失
        curr_eval_loss = float(sum(eval_losses) / len(eval_losses))
        if curr_eval_loss < eval_loss:
            eval_loss = curr_eval_loss
            torch.save(model.state_dict(), os.path.join(config.model_path, "domain-ner-%d-%.3f-%.3f.pth" % (epoch, curr_train_loss, curr_eval_loss)))

        info = "Epoch: {}\ttrain_loss: {}\teval_loss: {}".format(epoch, train_loss, eval_loss)
        m_progress.set_description(info)












