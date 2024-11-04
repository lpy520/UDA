import torch
import torch.nn as nn
import numpy as np
from advent.utils.func import loss_calc, bce_loss
from . import base

class MyLoss_correction(base.Loss):
    def __init__(self,weight=None,att_depth=None,out_channels=None,patch_size=None):
        super().__init__()
        self.nll = nn.NLLLoss(weight=weight)
        self.mseloss=nn.MSELoss()

        self.att_depth=att_depth
        self.patch_size=patch_size
        self.out_channels=out_channels
        self.class_values = [0,1,2,3]

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size))

    def forward(self, y_pr, y_gt, attentions):
        y_gt = [(y_gt == v) for v in self.class_values]

        y_gt = np.stack(y_gt, axis=-1).astype('float')

        y_gt = y_gt.transpose(0, 3, 1, 2).astype('float32')
        y_gt = torch.from_numpy(y_gt).to(y_pr.device)
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                             stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels, bias=False)
        conv_feamap_size.weight = nn.Parameter(torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(y_pr.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False


        y_gt_conv=conv_feamap_size(y_gt)/(2 ** self.att_depth*2 ** self.att_depth)
        attentions_gt=[]
        for i in range(y_gt_conv.size()[1]):
            unfold_y_gt = self.unfold(y_gt[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_y_gt_conv = self.unfold(y_gt_conv[:, i:i + 1, :, :])
            att=torch.matmul(unfold_y_gt,unfold_y_gt_conv)/(self.patch_size*self.patch_size)
            att=torch.unsqueeze(att,dim=1)
            attentions_gt.append(att)

        attentions_gt=torch.cat(attentions_gt,dim=1)

        y_gt=torch.argmax(y_gt,dim=-3)

        # loss_entropy=self.nll(y_pr,y_gt)
        loss_entropy = loss_calc(y_pr, y_gt, y_pr.device)
        loss_mse = self.mseloss(attentions, attentions_gt)
        loss=loss_entropy*3+loss_mse
        # loss = loss_entropy
        return loss
