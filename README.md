# domain-ner

基于BiLSTM+CRF的NER

## 1、背景

开源NER工具（如：LTP）可以实现通用领域的命名实体识别，但对垂直领域无能为力，特实现基于BiLSTM+CRF的NER模型。



## ISSUE

1、训练到最后一个batch时，实际拿到的数据<batch_size，hidden层会报错：

​	RuntimeError: Expected hidden[0] size (2, 6, 100), got [2, 16, 100]
