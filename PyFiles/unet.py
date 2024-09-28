import torch
from torch import nn
from torch.nn import functional as Func


# 构建神经网络模块，继承于nn.Module
# 主要包括两个方法:
# __init__方法：这个方法是类的构造函数，在创建类的实例时被调用。
# forward方法：这个方法定义了数据在神经网络中的传递方式。
class Conv_Block(nn.Module):
    # 初始化
    def __init__(self, in_channel, out_channel):
        # 继承，确保子类继承父类的属性和方法
        super(Conv_Block, self).__init__()
        # 卷积块的具体结构，例如添加卷积层、激活函数和其他操作，并实现前向传播方法forward()
        # 第一次卷积增加通道数，第二次卷积不改变通道数
        self.layer = nn.Sequential(
            # 并未改变图像尺寸，但是改变了通道数
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            # 对特征通道进行标准化处理，用于加速神经网络的训练
            nn.BatchNorm2d(out_channel),
            # 防止过拟合
            nn.Dropout2d(0.3),
            # 激活函数
            nn.LeakyReLU(),
            # 二次卷积，但没有改变通道数和尺寸，（起到个滤波的作用
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块

# Dropout是一种有效的正则化方法（提高模型的泛化能力），但在下采样层通常不需要使用，因为下采样层本身不容易过拟合，并且添加Dropout会增加计算
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # 未改变图像通道数，stride为2的目的是减小图像的尺寸，实现下采样
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 上采样模块
#
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        # 改变特征通道,使输出通道减半
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        # 将X尺寸变为原来两倍
        up = Func.interpolate(x, scale_factor=2, mode='nearest')
        # 通道数变为原来的一半
        out = self.layer(up)
        # 特征图联级操作
        return torch.cat((out, feature_map), dim=1)


# Unet定义
class UNet(nn.Module):
    def __init__(self, input_channel):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(input_channel, 32)
        self.d1 = DownSample(32)
        self.c2 = Conv_Block(32, 64)
        self.d2 = DownSample(64)
        self.c3 = Conv_Block(64, 128)
        self.d3 = DownSample(128)
        self.c4 = Conv_Block(128, 256)
        self.d4 = DownSample(256)
        self.c5 = Conv_Block(256, 512)
        # 上采样时会有个特征图的联级操作，导致通道数翻倍，所以需要输出通道减半
        self.u1 = UpSample(512)
        self.c6 = Conv_Block(512, 256)
        self.u2 = UpSample(256)
        self.c7 = Conv_Block(256, 128)
        self.u3 = UpSample(128)
        self.c8 = Conv_Block(128, 64)
        self.u4 = UpSample(64)
        self.c9 = Conv_Block(64, 32)
        # 理论上也可以写成 self.out=Conv_Block(32,1)
        self.out = nn.Conv2d(32, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()

    # 反向传播
    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        return self.Th(self.out(O4))


if __name__ == '__main__':
    input = torch.randn(1, 2, 256, 256)
    net = UNet(2)
    res = net(input)
    print(res.shape)
