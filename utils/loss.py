import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w) 批次大小、类别数、高度、宽度
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)  #布尔掩码，用于过滤掉目标标签的无效值
    target = target[target_mask]  #过滤掉目标标签中无效的像素点，只保留有效的像素点，这样做是为了避免无效标签对损失计算产生干扰
    if not target.data.dim():  #判断目标标签张量是否为空
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()  #将predict张量进行维度转换(n,c,h,w)->(n,h,w,c)
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)  #筛选有效的像素位置
    # print("***********")
    # print(predict.size())
    # print(target.size())
    loss = F.cross_entropy(predict, target, size_average=True)    #预测的张量、真实的标签张量、对每个像素位置的损失值求平均
    return loss

#用于计算概率分布向量的熵损失，它帮助提高模型在目标域上的泛化能力，使预测结果在目标域上分布得比较均匀
def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()  #获取输入概率分布向量的尺寸信息
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))
