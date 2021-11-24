import dataset

# 模型的配置
START_TAG = "<START>"
STOP_TAG = "<STOP>"
embedding_dim = 200
hidden_dim = 200
num_layers = 1    # LSTM的层数
model_path = "./checkpoint/"
word2idx_path = "./word2idx.json"    # 单字和序号的对应关系
mode = "train"    # 模型的模式：train、test

# 数据预处理配置
sent_len = 64
vocab_size = 3000
emb_size = 100
sent_pad = 10
seq_len = sent_len + 2 * sent_pad    # 每个句子的长度


total_epoch = 300    # 总epoch数
batch_size = 16
update_word2idx = False    # 是否需要更新word2idx.json文件

