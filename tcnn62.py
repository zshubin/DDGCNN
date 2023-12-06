"""
一句话：构建网络
"""
import torch
import torch.nn as nn


class tCNN(nn.Module):
    def __init__(self, win_train):
        super(tCNN, self).__init__()

        # 第一个卷积，卷积后的数据格式：16@1Xwin_train
        self.conv1 = nn.Conv2d(1, 16, (64, 1))
        self.bn1 = nn.BatchNorm2d(16, momentum=0.99, eps=0.001)
        # self.bn1 = nn.BatchNorm2d(16, momentum=0.99)
        self.elu1 = nn.ELU(alpha=1)
        self.dropout1 = nn.Dropout(0.4)

        # 第二个卷积，卷积后的数据格式：16@1X10
        self.conv2 = nn.Conv2d(16, 16, (1, win_train), stride=(5, 5), padding=(0, 23))
        self.bn2 = nn.BatchNorm2d(16, momentum=0.99, eps=0.001)
        # self.bn2 = nn.BatchNorm2d(16, momentum=0.99)
        self.elu2 = nn.ELU(alpha=1)
        self.dropout2 = nn.Dropout(0.4)

        # 第三个卷积，卷积后的数据格式：16@1X6
        self.conv3 = nn.Conv2d(16, 16, (1, 5))
        self.bn3 = nn.BatchNorm2d(16, momentum=0.99, eps=0.001)
        # self.bn3 = nn.BatchNorm2d(16, momentum=0.99)
        self.elu3 = nn.ELU(alpha=1)
        self.dropout3 = nn.Dropout(0.4)

        # @ 第四个卷积，卷积后的数据格式：32@1X1
        self.conv4 = nn.Conv2d(16, 32, (1, 6))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.99, eps=0.001)
        # self.bn4 = nn.BatchNorm2d(32, momentum=0.99)
        self.elu4 = nn.ELU(alpha=1)
        self.dropout4 = nn.Dropout(0.4)

        # dropout
        self.dropout5 = nn.Dropout(0.4)

        # 全连接
        self.linear = nn.Linear(32, 40)

    def forward(self, x):
        # 第一个卷积, 卷积后的数据格式：16@1X50
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.dropout1(x)

        # 第二个卷积, 卷积后的数据格式：16@1X10
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.dropout2(x)

        # 第三个卷积, 卷积后的数据格式：16@1X6
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout3(x)

        # 第四个卷积, 卷积后的数据格式：32@1X1
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu4(x)
        x = self.dropout4(x)

        # 打平
        x = torch.flatten(x, 1, 3)

        # 全连接
        out = self.linear(x)

        return out

