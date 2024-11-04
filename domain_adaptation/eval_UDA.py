# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load


def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,     #使用固定的尺寸对测试数据进行插值
                                verbose=True):            #输出详细的评估结果
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:   #对测试数据进行插值
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)      #从检查点文件中加载模型的权重参数
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))   #使用零矩阵来记录每个类别的像素交叉熵，用于计算最后的平均交并比(mIoU)
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():    #上下文管理器，表示在该代码块不计算梯度，用于在评估过程中节省内存和计算资源
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)   #进行转置操作，将通道维度转移到最后一个维度上
            output = np.argmax(output, axis=2)   #获取每个像素点预测的类别索引，得到最终的预测结果
        label = label.numpy()[0]            #从PyTorch张量转换为NumPy数组，取其中的第一个样本
        # 计算预测结果和标签之间的混淆矩阵;混淆矩阵的行表示真实类别，列表示预测类别;hist[i][j]表示真实类别为i,预测类别为j的像素数量
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)  #label.flatten()将二维数组展平为一维数组、output.flatten()将output从三维展平为一维数组
    inters_over_union_classes = per_class_iu(hist)                               #计算每个类别的交并比
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)  #通过Pickle反序列化加载缓存数据
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):

            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                print(restore_from)
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                image, label, _, name = next(test_iter)
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        if verbose:
            display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)        #加载检查点文件，将保存的模型权重信息加载到 saved_state_dict 变量中
    model.load_state_dict(saved_state_dict)          #将权重信息加载到model中
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
