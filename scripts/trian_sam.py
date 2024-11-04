import argparse
import numpy as np
import torch
import sys
from torch import optim
from torch.autograd import Variable
from torch.utils import data
import torch.backends.cudnn as cudnn
from advent.model.deeplabv2 import get_deeplab_v2_resnet101

from advent.dataset.infrared import InfraredDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
import os
import os.path as osp
import time
from PIL import Image
import torch.nn as nn
import warnings

from tqdm import tqdm
from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load
from torchsummary import summary

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()
def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

# RESTORE_FROM = './experiments/snapshots/Visible2Infrared_DeepLabv2_AdvEnt/model_7500.pth'
RESTORE_FROM = './new/model_7500.pth'
DATA_DIRECTORY = './data/Infrared'
DATA_LIST_PATH = './advent/dataset/infrared_list/train_new.txt'
# SAVE_PATH = './result/ship2_15000'

IGNORE_LABEL = 255
NUM_CLASSES = 4

MODEL = 'DeeplabMulti'

palette = [0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 0]  # RGB(0,0,0)-黑色背景；（225，225，225）-白色目标

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.tra
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")

    # parser.add_argument("--save", type=str, default=SAVE_PATH,
    #                     help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    # 获取参数
    args = get_arguments()
    # 加载数据集
    trainloader = data.DataLoader(InfraredDataSet(args.data_dir, args.data_list),
                                 batch_size=2, shuffle=True, pin_memory=True)

    # 初始化Deeplab模型
    model = get_deeplab_v2_resnet101(num_classes=args.num_classes)
    # 加载预训练模型权重
    if args.restore_from.endswith('.pth'):
        state_dict = torch.load(args.restore_from)
        model.load_state_dict(state_dict)
    else:
        print("Invalid model file extension. It should be a '.pth' file.")
    model.to(device)
    model.train()
    summary(model, (3, 256, 256))
    cudnn.benchmark = True #自动寻找最适合当前硬件的卷积实现的配置，以加速训练(pytorch每次运行卷积之前自动搜索并选择最佳的卷积算法)
    cudnn.enabled = True  #启动cudnn加速，cudnn是由NVDIA提供的用于深度学习的GPU加速库
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),  # 获取模型中需要进行优化的参数（权重和偏置）
                          lr=cfg.TRAIN.LEARNING_RATE,  # 即每次参数更新时的步长。SGD优化器将根据这个学习率来调整参数的更新幅度
                          momentum=cfg.TRAIN.MOMENTUM,  # 用于在参数更新时考虑历史梯度的影响，动量可以加快收敛速度，并有助于在参数空间跳出局部最优点
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    trainloader_iter = enumerate(trainloader)
    for i_iter in tqdm(range(2000)):  # 适应tqdm库的tqdm函数来迭代训练过程，并在终端中显示进度条
        try:
            _, batch = trainloader_iter.__next__()
        except StopIteration:
            trainloader = data.DataLoader(InfraredDataSet(args.data_dir, args.data_list),
                                          batch_size=2, shuffle=True, pin_memory=True)
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()
        # reset optimizers
        optimizer.zero_grad()  # 优化器所有参数的梯度置零，这是为了在每次迭代开始时清空之前的梯度，以便进行新一轮的参数更新
        adjust_learning_rate(optimizer, i_iter, cfg)  # 根据当前i_iter来调整学习率，这是为了实现在训练过程中逐渐降低学习率的策略，以提高模型的稳定性和收敛性
        images, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images.cuda(device))  # 获得目标域图像在辅助分类器和主分类器上的预测结果
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)  # 插值，将辅助预测结果调整为与源域图像尺寸相同的大小，从而方便与源域图像的标签进行计算损失
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)  # 计算交叉熵损失
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main  # 将主要预测结果的辅助预测结果的交叉熵损失进行加权计算
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()
        optimizer.step()
        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main}
        print_losses(current_losses, i_iter)

        if i_iter % 200 == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR+'new111')
            torch.save(model.state_dict(),
                       osp.join('new111', f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()   #刷新输出缓冲区

if __name__ == '__main__':
    main()
