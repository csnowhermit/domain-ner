import dataset

# 模型的配置
START_TAG = "<START>"
STOP_TAG = "<STOP>"
embedding_dim = 200
hidden_dim = 200

# 数据预处理配置
sent_len = 64
vocab_size = 3000
emb_size = 100
sent_pad = 10
seq_len = sent_len + 2 * sent_pad    # 每个句子的长度


total_epoch = 300    # 总epoch数
batch_size = 16


