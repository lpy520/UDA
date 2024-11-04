from torch import nn

# 创建一个基于全连接层的鉴别器网络的函数
# 目标是通过学习源域和目标域之间的特征差异，对输入的特征图进行分类，并输出一个标量值，表示输入特征图属于源域还是目标域的概率
def get_fc_discriminator(num_classes, ndf=64):
    return nn.Sequential(                   #pyTorch中的一个容器，用于按顺序封装多个神经网络层，构建一个简单的神经网络模型
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),   #从输入图像中提取特征并构建高级的特征表示
        nn.LeakyReLU(negative_slope=0.2, inplace=True),          #非线性变换使得鉴别器能够学习更复杂的特征映射，使其能够更好地区分源域和目标域之间的特征表示
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),            #在域适应方法中，多次下采样可以使特征图具有更大的感受野和更强的抽象能力
        nn.LeakyReLU(negative_slope=0.2, inplace=True),                             #从而能够更好地捕捉图像的全局和局部特征，帮助模型学习到更具有区分性的特征表示，提高域适应的性能。
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )
