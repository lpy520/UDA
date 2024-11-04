import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
# from model.deeplab_multi import DeeplabMulti
from advent.model.deeplabv2 import get_deeplab_v2_resnet50
# from dataset.test_infrared import Infrared
# from dataset.infrared_dataset import InfraredDataSet
from advent.dataset.infrared import InfraredDataSet
import os
import os.path as osp
import time
from PIL import Image
import torch.nn as nn
import warnings

from tqdm import tqdm
from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

# RESTORE_FROM = './experiments/snapshots/Visible2Infrared_DeepLabv2_AdvEnt'
# RESTORE_FROM = './new_resnet50'
# RESTORE_FROM = './new_sum'
# RESTORE_FROM = './new_end_0402'
# RESTORE_FROM = './new_main'
RESTORE_FROM = './new_end_0528_0720'
DATA_DIRECTORY = './data/Infrared'
DATA_LIST_PATH = './advent/dataset/infrared_list/val.txt'
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
      A list of parsed arguments.
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
    testloader = data.DataLoader(InfraredDataSet(args.data_dir, args.data_list),
                                 batch_size=1, shuffle=False, pin_memory=True)

    # 初始化Deeplab模型
    model = get_deeplab_v2_resnet50(num_classes=args.num_classes)

    # 模型加载参
    start_iter = 200  # 从第 轮开始保存
    step = 200  # 每隔 轮保存一次
    max_iter = 4000  # 最多保存到 轮
    cache_path = osp.join(args.restore_from, 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''

    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(args.restore_from, f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            print('Waiting for model..!')
            while not osp.exists(restore_from):
                time.sleep(5)
        print("Evaluating model", restore_from)

        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(model, restore_from, device)
            # eval
            hist = np.zeros((args.num_classes, args.num_classes))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(testloader)
            for index in tqdm(range(len(testloader))):
                image, label, _, name = next(test_iter)
                with torch.no_grad():

                    pred_main = model(image.cuda(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                label = label.numpy()[0]
                # 把lebel从RGB变成二维标签形式
                hist += fast_hist(label.flatten(), output.flatten(), args.num_classes)
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
        display_stats(args, ["bg", "qt", "qzj", "qzc"], inters_over_union_classes)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(args, name_classes, inters_over_union_classes):
    for ind_class in range(args.num_classes):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))

# def display_stats(args, name_classes, inters_over_union_classes):
#     for ind_class in range(args.num_classes):
#         if name_classes[ind_class] != "bg":
#             print(name_classes[ind_class] + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
if __name__ == '__main__':
    main()
