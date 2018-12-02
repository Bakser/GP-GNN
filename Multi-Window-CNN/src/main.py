import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve,average_precision_score
def get_one_hot(label):
    return (np.arange(dimE)==label[:,None]).astype(np.integer)
def test():
    model.eval()#model就是你的最佳模型
    preds=[]
    labels=[]
    for words,pos,loc,maskL,maskR,label in dataset.batchs():
        scores=model(words.cuda(),pos.cuda(),loc.cuda(),maskL.cuda(),maskR.cuda())#得到模型预测的概率，softmax之后的
        pred=scores.detach().cpu().numpy()
        preds.append(pred)
        labels.append(get_one_hot(label.numpy()))#把label从数转成one_hot的形式
    preds=np.concatenate(preds,0)
    labels=np.concatenate(labels,0)
    preds=np.reshape(preds[:,1:],(-1))#去掉label=0是因为在这里label=0是NA，即没有关系，对应到我们的数据集里就是没有在edgeSet里标出的边的label要打成NA
    labels=np.reshape(labels[:,1:],(-1))
    p,r,th=precision_recall_curve(labels,preds)
    #r做横坐标，p做纵坐标画图就好了

