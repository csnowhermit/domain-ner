import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import matplotlib.pyplot as plt  # 图形处理包
import itertools  # 处理混淆矩阵
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']  # 让图形可以显示中文
plt.rcParams['axes.unicode_minus']=False

'''
    评估NER分类效果
'''

'''
    混淆矩阵画图方法
'''
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

'''
    画出混淆矩阵图
    y: 真实值y（标签/df型）
    y_prob：预测概率
    thres: 阈值，多少以上为预测正确
    png_savename: 保存图片的名字，默认不保存
    return: 输出混淆矩阵图
'''
def metrics_plot(y, y_prob, thres=0.45, png_savename=0):

    y_prediction = y_prob > thres  # 多少以上的概率判定为正
    cnf_matrix = confusion_matrix(y, y_prediction)  # 形成混淆矩阵
    np.set_printoptions(precision=2)  # 设置浮点精度
    vali_recall = '{0:.3f}'.format(cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))  # 召回率
    vali_precision = '{0:.3f}'.format(cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1]))  # 精确率
    print('recall=%s%%，predcision=%s%%' % ('{0:.1f}'.format(float(vali_recall) * 100),
                                        '{0:.1f}'.format(float(vali_precision) * 100)))
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='召回率=%s%% \n 精确率=%s%%' % ('{0:.1f}'.format(float(vali_recall) * 100),
                                                          '{0:.1f}'.format(float(vali_precision) * 100)))
    if png_savename != 0:
        plt.savefig("%s_混淆矩阵.png" % png_savename, dpi=300)  # 保存混淆矩阵图

if __name__ == '__main__':
    y = np.array([0, 0, 0, 1, 1])    # 标签
    y_prob = np.array([0, 0, 1, 1, 1])    # 预测的

    metrics_plot(y, y_prob, thres=0.45, png_savename=0)