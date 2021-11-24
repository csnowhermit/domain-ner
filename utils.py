import os

'''
    NER结果后处理
    :param sentence 原文list
    :param pred 预测的list
    :param idx2tag {编号: 实体类别}对应关系dict
'''
def ner_post_process(sentence, pred, idx2tag):
    format_result = {}
    curr_index = 0
    for i in range(len(sentence)):
        if i < curr_index:  # 人名后几个字不用再循环了
            continue
        if pred[i] != 0:
            tmp = pred[i]
            res = sentence[i]
            for j in range(i + 1, len(sentence)):
                if pred[j] == tmp:  # 同类才追加，不同类直接换一次循环
                    res += sentence[j]
                else:
                    curr_index = j
                    break
            if idx2tag[tmp] not in format_result.keys():
                format_result[idx2tag[tmp]] = [res]
            else:
                tmp_res = format_result[idx2tag[tmp]]
                tmp_res.append(res)
                format_result[idx2tag[tmp]] = tmp_res
    return format_result


if __name__ == '__main__':
    sentence = ['张', '三', '叫', '李', '四', '去', '烧', '烤', '，', '李', '四', '说', '他', '得', '糖', '尿', '病', '感', '冒', '冲', '剂', '。']
    paths = [[1, 1, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 0, 0, 3, 3, 3, 4, 4, 4, 4, 0]]
    idx2tag = {1: '人名', 2: '活动', 3: '病名', 4: '药名'}
    format_result = ner_post_process(sentence, paths[0], idx2tag)
    print(format_result)