# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask


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
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()        #设置鉴别器模型为训练模式
    d_aux.to(device)

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
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
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
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():    #冻结判别器的参数，防止其在训练过程中被更新
            param.requires_grad = False     #目的是在训练域适应网络中，只更新生成器部分的参数，保持判别器的参数不变，从而帮助生成器更好的适应目标域
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()       #将辅助分类器和主分类器的损失加权求和得到总的损失，然后进行反向传播计算生成器的梯度

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))  #prob_2_entropy()将概率转换为熵值  --> 得到判别器对目标域辅助分类器输出的判定结果
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)       #用二进制交叉熵损失函数来计算判别器对目标域辅助分类器输出的判定结果与目标域标签之间的损失
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss   # 总的对抗性损失
        loss.backward() # 进行反向传播计算判别器的梯度

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():        #解冻判别器的参数
            param.requires_grad = True          #可以在后续的训练过程中对判别器的权重进行更新
        for param in d_main.parameters():       #以使其更好地判断输入数据来自源域还是目标域
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()   #更新模型的参数
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            #print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), osp.join('../../0816_pth', f'model_{i_iter}.pth'))
            #torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            #torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            #torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            #将图像以及预测结果记录到可视化面板中
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')

# 将图像、预测结果、熵图像在TensorBoard界面进行可视化展示
def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)  #将输入的图像张量转换为一个网络状的图像
    writer.add_image(f'Image - {type_}', grid_image, i_iter)              #将上述生成的图像添加到TensorBoard的日志中
    #将主分类器的预测结果转换为彩色的掩膜图像，并将其转换为TensorBoard所需要的网络状图像数据
    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(          #F.softmax()将主分类器的预测结果转化为概率分布；
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),        #np.argmax()找到每个像素位置上概率最大的类别索引
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,     #colorize_mask()将类别索引转换为彩色掩模图像，以便更好的可视化预测结果
                           normalize=False, range=(0, 255))                             #torch.from_numpy()将上述转换后的图像转换为pyTorch的张量
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)
    #将主分类器的预测结果转换为概率分布，用于计算熵
    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    #计算概率分布熵
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    #将熵值转换为张量，并将其转换为网络状的图像
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def train_minent(model, trainloader, targetloader, cfg):
    ''' UDA training with minEnt
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:  #创建一个SummaryWriter写入器，将训练过程的相关信息写入到指定的TensorBoard日志目录中
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK  (准备模型进行训练之前的操作)
    model.train()       #将模式设置为训练模式
    model.to(device)    #将模型移动到指定设备上
    cudnn.benchmark = True #自动寻找最适合当前硬件的卷积实现的配置，以加速训练(pytorch每次运行卷积之前自动搜索并选择最佳的卷积算法)
    cudnn.enabled = True  #启动cudnn加速，cudnn是由NVDIA提供的用于深度学习的GPU加速库

    # OPTIMIZERS (实现在训练过程中自动更新模型的权重和偏置，从而最小化损失函数并提高模型性能)
    # segnet's optimizer 用于创建一个随机梯度（SGD）优化器对象，用于优化模型的参数
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE), #获取模型中需要进行优化的参数（权重和偏置）
                          lr=cfg.TRAIN.LEARNING_RATE,                     #即每次参数更新时的步长。SGD优化器将根据这个学习率来调整参数的更新幅度
                          momentum=cfg.TRAIN.MOMENTUM,                    #用于在参数更新时考虑历史梯度的影响，动量可以加快收敛速度，并有助于在参数空间跳出局部最优点
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)            #权重衰减的系数，是一种防止过拟合的技术。它通过在损失函数中添加一个正则化项来惩罚较大的权重值

    # interpolate output segmaps

    #对源域和目标域的分割图进行插值，使其与输入图像的尺寸相同
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',  #bilinear双线性插值法来推断未知点的值
                         align_corners=True)                                                  #基于距离的插值方法，通过对四个最近邻数据点的加权平均来推断未知的点，从而得到平滑的插值结果
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)                                            #align_corners=true表示在插值过程中对边缘像素的处理方式，使得插值的结果更加准确
#  将数据加载器转换为一个枚举对象，在进行UDA训练过程中，可以同时迭代源域数据和目标域数据
#  并使用他们来更新模型的参数，从而实现对目标域的适应和优化
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):   #适应tqdm库的tqdm函数来迭代训练过程，并在终端中显示进度条

        # reset optimizers
        optimizer.zero_grad()  #优化器所有参数的梯度置零，这是为了在每次迭代开始时清空之前的梯度，以便进行新一轮的参数更新

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)  #根据当前i_iter来调整学习率，这是为了实现在训练过程中逐渐降低学习率的策略，以提高模型的稳定性和收敛性

    #辅助分类器目的是在源域和目标域之间共享特征，并帮助提高源域(目标域)图像的分类性能。
    #它主要用于域自适应方法中，用于减轻源域和目标域之间的域偏移问题
    #主分类器目的是对源域(目标域)图像进行语义分割，将每个像素分配到不同的语义类别。
    #它的输出通常是最终的预测结果，用于生成源域(目标域)图像的语义分割图
        # UDA Training
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch   #  images_source :[2, 3, 256, 256];labels :[2, 256, 256]
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)   #插值，将辅助预测结果调整为与源域图像尺寸相同的大小，从而方便与源域图像的标签进行计算损失
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device) #计算交叉熵损失
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)     #插值，将主要预测结果调整为与源域图像尺寸相同的大小
        # print("************")
        # print(pred_src_main.size())   [2, 4, 256, 256]
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main     #将主要预测结果的辅助预测结果的交叉熵损失进行加权计算
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()         #用于计算当前损失对于模型参数的梯度，并把这些梯度存储在模型参数的“.grad”属性中
                                #这些梯度告诉我们损失函数相对于每个参数的变化情况，从而可以用梯度下降等优化算法来更新参数，让损失函数尽可能减小，从而优化模型

        # adversarial training with minent
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))  #获得目标域图像在辅助分类器和主分类器上的预测结果
        pred_trg_aux = interp_target(pred_trg_aux)
        pred_trg_main = interp_target(pred_trg_main)
        pred_prob_trg_aux = F.softmax(pred_trg_aux)     #softmax函数将预测结果转换为概率分布
        pred_prob_trg_main = F.softmax(pred_trg_main)   #得到它们在各个类别上的概率预测结果

        loss_target_entp_aux = entropy_loss(pred_prob_trg_aux)
        loss_target_entp_main = entropy_loss(pred_prob_trg_main)
        loss = (cfg.TRAIN.LAMBDA_ENT_AUX * loss_target_entp_aux
                + cfg.TRAIN.LAMBDA_ENT_MAIN * loss_target_entp_main)
        loss.backward()
        optimizer.step()  #根据累积的梯度对模型参数进行更新，以使得损失函数尽可能减小，从而优化模型

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_aux': loss_target_entp_aux,
                          'loss_ent_main': loss_target_entp_main}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), osp.join('../../0816_minent_pth', f'model_{i_iter}.pth'))
            #torch.save(model.state_dict(),
            #           osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()   #刷新输出缓冲区

        # Visualize with tensorboard
        if viz_tensorboard:                 #将当前的损失值current_losses记录到TensorBoard的日志中，从而可也在TensorBoard中查看损失函数在训练过程中的变化情况
            log_losses_tensorboard(writer, current_losses, i_iter)
            # 将图像以及预测结果记录到可视化面板中
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')
def train_minent_unet(model, trainloader, targetloader, cfg):

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:  #创建一个SummaryWriter写入器，将训练过程的相关信息写入到指定的TensorBoard日志目录中
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK  (准备模型进行训练之前的操作)
    model.train()       #将模式设置为训练模式
    model.to(device)    #将模型移动到指定设备上
    cudnn.benchmark = True #自动寻找最适合当前硬件的卷积实现的配置，以加速训练(pytorch每次运行卷积之前自动搜索并选择最佳的卷积算法)
    cudnn.enabled = True  #启动cudnn加速，cudnn是由NVDIA提供的用于深度学习的GPU加速库

    # OPTIMIZERS (实现在训练过程中自动更新模型的权重和偏置，从而最小化损失函数并提高模型性能)
    # segnet's optimizer 用于创建一个随机梯度（SGD）优化器对象，用于优化模型的参数
    # optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE), #获取模型中需要进行优化的参数（权重和偏置）
    #                       lr=cfg.TRAIN.LEARNING_RATE,                     #即每次参数更新时的步长。SGD优化器将根据这个学习率来调整参数的更新幅度
    #                       momentum=cfg.TRAIN.MOMENTUM,                    #用于在参数更新时考虑历史梯度的影响，动量可以加快收敛速度，并有助于在参数空间跳出局部最优点
    #                       weight_decay=cfg.TRAIN.WEIGHT_DECAY)            #权重衰减的系数，是一种防止过拟合的技术。它通过在损失函数中添加一个正则化项来惩罚较大的权重值
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    # interpolate output segmaps

    #对源域和目标域的分割图进行插值，使其与输入图像的尺寸相同
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',  #bilinear双线性插值法来推断未知点的值
                         align_corners=True)                                                  #基于距离的插值方法，通过对四个最近邻数据点的加权平均来推断未知的点，从而得到平滑的插值结果
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)                                            #align_corners=true表示在插值过程中对边缘像素的处理方式，使得插值的结果更加准确
#  将数据加载器转换为一个枚举对象，在进行UDA训练过程中，可以同时迭代源域数据和目标域数据
#  并使用他们来更新模型的参数，从而实现对目标域的适应和优化
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):   #适应tqdm库的tqdm函数来迭代训练过程，并在终端中显示进度条

        # reset optimizers
        optimizer.zero_grad()  #优化器所有参数的梯度置零，这是为了在每次迭代开始时清空之前的梯度，以便进行新一轮的参数更新

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)  #根据当前i_iter来调整学习率，这是为了实现在训练过程中逐渐降低学习率的策略，以提高模型的稳定性和收敛性
        # UDA Training
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch   #  images_source :[2, 3, 256, 256];labels :[2, 256, 256]
        pred_src_main = model(images_source.cuda(device))
        pred_src_main = interp(pred_src_main)     #插值，将主要预测结果调整为与源域图像尺寸相同的大小
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss =loss_seg_src_main     #将主要预测结果的辅助预测结果的交叉熵损失进行加权计算
        loss.backward()         #用于计算当前损失对于模型参数的梯度，并把这些梯度存储在模型参数的“.grad”属性中
                                #这些梯度告诉我们损失函数相对于每个参数的变化情况，从而可以用梯度下降等优化算法来更新参数，让损失函数尽可能减小，从而优化模型

        # adversarial training with minent
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_main = model(images.cuda(device))  #获得目标域图像在辅助分类器和主分类器上的预测结果
        pred_trg_main = interp_target(pred_trg_main)
        pred_prob_trg_main = F.softmax(pred_trg_main)   #得到它们在各个类别上的概率预测结果
        loss_target_entp_main = entropy_loss(pred_prob_trg_main)
        loss = loss_target_entp_main
        loss.backward()
        optimizer.step()  #根据累积的梯度对模型参数进行更新，以使得损失函数尽可能减小，从而优化模型

        current_losses = {'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_main': loss_target_entp_main}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()   #刷新输出缓冲区


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

#将PyTorch张量（tensor）转换为NumPy数组
def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
