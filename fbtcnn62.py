"""
一句话：构建网络
"""
import torch
import torch.nn as nn


class tCNN(nn.Module):
    def __init__(self, win_train):
        super(tCNN, self).__init__()

        # 第一个卷积，卷积后的数据格式：16@1Xwin_train
        self.conv1 = nn.Conv2d(1, 16, (62, 1))
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

        self.sub_band = nn.Sequential(self.conv1,self.bn1,self.elu1,self.dropout1,
                                      self.conv2,self.bn2,self.elu2,self.dropout2,
                                      self.conv3,self.bn3,self.elu3,self.dropout3,)

        # @ 第四个卷积，卷积后的数据格式：32@1X1
        self.conv4 = nn.Conv2d(16, 32, (1, 6))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.99, eps=0.001)
        # self.bn4 = nn.BatchNorm2d(32, momentum=0.99)
        self.elu4 = nn.ELU(alpha=1)
        self.dropout4 = nn.Dropout(0.4)

        # dropout
        self.dropout5 = nn.Dropout(0.4)

        # 全连接
        self.linear = nn.Linear(32, 4)

    def forward(self, x1, x2, x3, x4):
        x1 = self.sub_band(x1)
        x2 = self.sub_band(x2)
        x3 = self.sub_band(x3)
        x4 = self.sub_band(x4)

        # 第三个卷积, 卷积后的数据格式：16@1X6
        x = x1 + x2 + x3 + x4
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

