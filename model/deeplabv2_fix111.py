import torch
import torch.nn as nn

from advent.utils.modules import Activation

affine_par = True

#它的实例将用于构建ResNet架构的深层模型，并用于进行图像特征的提取和残差连接
class Bottleneck(nn.Module):
    expansion = 4  #每个残差块中的特征图尺寸的倍增因子

#   inplanes:输入特征图的通道数；planes:卷积核的数量（残差块中卷积层的输出通道数）；
#   stride:卷积的步幅； dilation:卷积的膨胀率；
#   downsample:降采样层，用于调整残差块的维度，以便于输入特征图的维度匹配；
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        #这些卷积层用于实现对输入特征图进行加工和组合，生成残差块输出的功能。
        super(Bottleneck, self).__init__()
    # change
        #此1*1的卷积层卷积层用于将输入特征图（inplanes通道）变换成输出特征图（planes通道）
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        #它表示2D批归一化层，用于对self.conv1输出特征图进行归一化，以加速模型的收敛和提高模型的稳定性
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False                        #将参数设置为不需要计算梯度，因为批归一化层的参数
        padding = dilation                                 #在训练过程一般是固定的，不需要通过梯度更新
    # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,         #进一步进行卷积操作
                               padding=padding, bias=False, dilation=dilation)  #从而提取更丰富的特征
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        #ReLU激活函数层，对self.conv3输出特征图进行非线性激活操作
        self.relu = nn.ReLU(inplace=True)
        #用于存储降采样层的引用，降采样层用于调整残差块的维度，以便与输入特征图的维度匹配，实现残差连接
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

#一个用于多尺度分类的模块
#它在给定的不同尺度上使用多个卷积层来对输入特征进行分类，并生成对应尺度的分类结果
class ClassifierModule(nn.Module):    #卷积层的空洞率 ；卷积层的填充大小
    def __init__(self, inplanes, dilation_series, padding_series, num_classes,activation=None):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()  #用以存储多个卷积层
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=8)
        self.out_channels = num_classes
        self.att_depth = 3
        self.patch_size = 5
        self.activation = nn.LogSoftmax(dim=1)
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))
        #对每个卷积层进行初始化
        #将它们的权重参数初始化为从均值为0、标准差为0.01的正态分布中采用得到的值
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

        self.project = nn.Sequential(
            nn.Conv2d(4 * num_classes, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))

    #模型通过多个卷积层进行处理，然后将这些处理后的特征图逐个相加
    def forward(self, x, attentions):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels,
                                      kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                                     stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels,
                                     bias=False)
        conv_feamap_size.weight = nn.Parameter(
            torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(out.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False
        out = self.upsampling(out)
        fold_layer = torch.nn.Fold(output_size=(out.size()[-2], out.size()[-1]),
                                   kernel_size=(self.patch_size, self.patch_size),
                                   stride=(self.patch_size, self.patch_size))
        correction = []
        out_argmax = torch.argmax(out, dim=1)

        pr_temp = torch.zeros(out.size()).to(out.device)
        src = torch.ones(out.size()).to(out.device)
        out_softmax = pr_temp.scatter(dim=1, index=out_argmax.unsqueeze(1), src=src)
        argout_feamap = conv_feamap_size(out_softmax) / (2 ** self.att_depth * 2 ** self.att_depth)
        for i in range(out.size()[1]):
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001, dim=-1)

            att = torch.matmul(attentions[:, i:i + 1, :, :] / non_zeros,
                               torch.unsqueeze(self.unfold(argout_feamap[:, i:i + 1, :, :]), dim=1).transpose(-1, -2))

            att = torch.squeeze(att, dim=1)

            att = fold_layer(att.transpose(-1, -2))

            correction.append(att)

        correction = torch.cat(correction, dim=1)

        out = correction * out + out

        # out = self.activation(out)
        # attentions= []
        return out,attentions


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self._out_channels = (3, 64, 256, 512, 1024, 2048)
        self._attention_on_depth = 3
        self.patch_size = 5
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
    #负责对图像进行初步处理
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        #最大池化层，它用于将特征图划分为不重叠的区域，并在每个区域中选择最大的值作为输出，从而减少特征图的空间尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # change
    #负责构建深层网络，逐步提取输入特征图的高级特征
        #创建多层的ResNet结构，每层由多个Bottleneck块（包含一系列卷积、批归一化和激活操作）构成深层次网络的基本模块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:  #是否使用多级别的特征图进行语义分割
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():   #权重初始化
            if isinstance(m, nn.Conv2d):            #Conv2d:将其权重参数进行高斯分布初始化
                m.weight.data.normal_(0, 0.01)      #使用均值为0、标准差为0.01的
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)              #权重参数设置为1、偏置参数设置为0
                m.bias.data.zero_()
        self.conv_feamap = nn.Sequential(
            nn.Conv2d(self._out_channels[self._attention_on_depth], num_classes, kernel_size=(1, 1), stride=1)
        )
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))
        self.conv_img = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3),

            nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=1)
        )
        self.resolution_trans = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

                    # Bottleneck、输出通道数、Bottleneck的个数、步幅、空洞率
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        #判断是否需要添加下采样层
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        attentions = []
        ini_img = self.conv_img(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)  #[2,7,32,32]
        for i in range(feamap.size()[1]):
            unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)  #[2,625,100]
            unfold_img = self.resolution_trans(unfold_img)

            unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])   #[2,100,9]
            unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)
            att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)
            att = torch.unsqueeze(att, 1)
            attentions.append(att)

        attentions = torch.cat(attentions, dim=1)

        x = self.layer3(x)
        if self.multi_level:
            prediction1, attentions1 = self.layer5(x, attentions)  # produce segmap 2
        x = self.layer4(x)
        prediction, attentions = self.layer6(x,attentions)  # produce segmap 2
        return prediction,prediction1,attentions

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


def get_deeplab_v2_resnet101(num_classes=19, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    # model = ResNetMulti(Bottleneck, [3, 4, 6, 3], num_classes, multi_level)
    return model
def get_deeplab_v2_resnet50(num_classes=19, multi_level=True):
    # model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    model = ResNetMulti(Bottleneck, [3, 4, 6, 3], num_classes, multi_level)
    return model