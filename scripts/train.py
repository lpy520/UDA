# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import yaml
import torch
from torch.utils import data

from advent.model.deeplabv2 import get_deeplab_v2_resnet50,get_deeplab_v2_resnet101
from advent.dataset.visible import VisibleDataSet
from advent.dataset.infrared import InfraredDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.train_UDA import train_domain_adaptation

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


def _init_fn(worker_id):
    np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

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

        # def _init_fn(worker_id):
        #     np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2_resnet101(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()         #用于保存根据预训练模型更新后的参数
            for i in saved_state_dict:           #只复制layer5以外的其他权重参数，以确保预训练权重和当前模型的结构一致
                i_parts = i.split('.')           #因为layer5对应的分类器部分是根据任务而变化的，而当前模型中可能不存在layer5这个子模块
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)   #将更新后的参数加载到当前模型，实现对模型参数的初始化
        else:   #模型不是基于ImageNet的DeepLabv2,则直接将预训练模型权重参数全部加载到当前模型
            model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    # if cfg.TRAIN.MODEL == 'DeepLabv2':
    #     model = get_deeplab_v2_resnet50(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
    #     saved_state_dict = torch.load('../../pretrained_models/resnet50-19c8e357.pth')
    #     if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
    #         new_params = model.state_dict().copy()  # 用于保存根据预训练模型更新后的参数
    #         for i in saved_state_dict:  # 只复制layer5以外的其他权重参数，以确保预训练权重和当前模型的结构一致
    #             i_parts = i.split('.')  # 因为layer5对应的分类器部分是根据任务而变化的，而当前模型中可能不存在layer5这个子模块
    #             if not i_parts[0] == 'fc':
    #                 new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    #         model.load_state_dict(new_params)  # 将更新后的参数加载到当前模型，实现对模型参数的初始化
    #     else:  # 模型不是基于ImageNet的DeepLabv2,则直接将预训练模型权重参数全部加载到当前模型
    #         model.load_state_dict(saved_state_dict)
    # else:
    #     raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
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
# 将训练过程中使用的各种配置参数保存到一个YAML文件中
    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    train_domain_adaptation(model, source_loader, target_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
