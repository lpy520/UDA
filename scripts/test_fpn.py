# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os,sys
import os.path as osp
import pprint
from pathlib import Path
import random
import warnings
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils import data
from advent.model.discriminator import get_fc_discriminator
from advent.model.deeplabv2_fpn_fix import get_deeplab_v2_resnet50,get_deeplab_v2_resnet101
from advent.dataset.visible import VisibleDataSet
from advent.dataset.infrared import InfraredDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.losses import MyLoss_correction
from advent.utils.viz_segmask import colorize_mask

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_arguments():
    """
    Parse input arguments 解析输入的命令行参数
    """
    #创建参数解析器对象
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
   #add_argument添加各种命令行参数的定义
    parser.add_argument('--cfg', type=str, default='./configs/advent.yml',
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        # list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
        loss_scalar = to_numpy(loss_value).item()  # 将 numpy 数组转换为标量值
        list_strings.append(f'{loss_name} = {loss_scalar:.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def _init_fn(worker_id):
    np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)
def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK 判别器根据输入的分割图预测出标签，然后与真实域标签进行对比，计算域适应的损失
    # feature-level 创建一个特征级别的鉴别器网络，使其能够区分源域和目标域的特征表示
    #一个用于辅助分类器域适应的判别器
    # d_aux = get_fc_discriminator(num_classes=num_classes)
    # d_aux.train()  # 设置鉴别器模型为训练模式
    # d_aux.to(device)
    # seg maps, i.e. output, level
    #一个用于主分类器域适应的判别器
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    # optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                              betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        # optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        # for param in d_aux.parameters():  # 冻结判别器的参数，防止其在训练过程中被更新
        #     param.requires_grad = False  # 目的是在训练域适应网络中，只更新生成器部分的参数，保持判别器的参数不变，从而帮助生成器更好的适应目标域
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_main = model(images_source.cuda(device))
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_src_aux = interp(pred_src_aux)
        #     loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        # else:
        #     loss_seg_src_aux = 0
        # pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
        loss.backward()  # 将辅助分类器和主分类器的损失加权求和得到总的损失，然后进行反向传播计算生成器的梯度

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_main = model(images.cuda(device))
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_trg_aux = interp_target(pred_trg_aux)
        #     d_out_aux = d_aux(
        #         prob_2_entropy(F.softmax(pred_trg_aux)))  # prob_2_entropy()将概率转换为熵值  --> 得到判别器对目标域辅助分类器输出的判定结果
        #     loss_adv_trg_aux = bce_loss(d_out_aux, source_label)  # 用二进制交叉熵损失函数来计算判别器对目标域辅助分类器输出的判定结果与目标域标签之间的损失
        # else:
        #     loss_adv_trg_aux = 0
        # pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss =  loss_adv_trg_main
        loss = loss  # 总的对抗性损失
        loss.backward()  # 进行反向传播计算判别器的梯度

        # Train discriminator networks
        # enable training mode on discriminator networks
        # for param in d_aux.parameters():  # 解冻判别器的参数
        #     param.requires_grad = True  # 可以在后续的训练过程中对判别器的权重进行更新
        for param in d_main.parameters():  # 以使其更好地判断输入数据来自源域还是目标域
            param.requires_grad = True
        # train with source
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_src_aux = pred_src_aux.detach()
        #     d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
        #     loss_d_aux = bce_loss(d_out_aux, source_label)
        #     loss_d_aux = loss_d_aux / 2
        #     loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_trg_aux = pred_trg_aux.detach()
        #     d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
        #     loss_d_aux = bce_loss(d_out_aux, target_label)
        #     loss_d_aux = loss_d_aux / 2
        #     loss_d_aux.backward()
        # else:
        #     loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()  # 更新模型的参数
        # if cfg.TRAIN.MULTI_LEVEL:
        #     optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_main': loss_seg_src_main,
                          # 'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          # 'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR+'pp')
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()



def main(config_file, exp_suffix):

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg) #美观打印cfg配置字典

    # INIT
    # _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)
        # random.seed(cfg.TRAIN.RANDOM_SEED)
        # np.random.seed(cfg.TRAIN.RANDOM_SEED)
        # torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        # torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)

        # def _init_fn(worker_id):
        #     np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    teacher_model = get_deeplab_v2_resnet101(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
    saved_state_dict = torch.load('../../pretrained_models/DeepLab_resnet_pretrained_imagenet.pth')
    new_params = teacher_model.state_dict().copy()  # 用于保存根据预训练模型更新后的参数
    for i in saved_state_dict:  # 只复制layer5以外的其他权重参数，以确保预训练权重和当前模型的结构一致
        i_parts = i.split('.')  # 因为layer5对应的分类器部分是根据任务而变化的，而当前模型中可能不存在layer5这个子模块
        if not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    teacher_model.load_state_dict(new_params)  # 将更新后的参数加载到当前模型，实现对模型参数的初始化
    print('Model loaded')

#训练过程中，模型会通过源域数据集学习特征表示，并通过目标域数据集进行适应，以提高在目标域上的性能
    # DATALOADERS
    source_dataset = VisibleDataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 set=cfg.TRAIN.SET_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,   #将图像随机的剪裁为指定大小，用于数据增强和训练
                                 mean=cfg.TRAIN.IMG_MEAN)          #用于图像处理的平均值，减去图像像素的均值，以标准化图像
    source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE, #每个批次的样本数
                                    num_workers=cfg.NUM_WORKERS ,              #数据加载并行工作线程数
                                    shuffle=True,               #随机打乱数据的顺序，有助于模型更好的学习数据之间的关联性，防止模型过拟合
                                    pin_memory=True,            #将数据存储在固定的内存区域，这对于GPU加速数据加载有用，可以提高数据传输效率
                                    worker_init_fn=_init_fn)    #每个工作线程初始化,设置随机种子，以确保数据加载的随机性

    target_dataset = InfraredDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,  #数据集附加信息的文件路径
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    # UDA TRAINING
    train_advent(teacher_model, source_loader, target_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)

