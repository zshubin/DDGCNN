import torch
from torch import nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False, dropout=0.2, trans_class='DCD',device='cuda'):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.device = device
        self.num_out = num_out
        if trans_class=='nomal_conv':
            self.conv = nn.Conv2d(num_in, num_out, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=bias).to(self.device)
        elif trans_class=='linear':
            self.conv = Linear(num_in,num_out).to(self.device)
        elif trans_class=='DCD':
            self.conv = GDCD(num_in, num_out, stride=(1, 1), bias=bias, dropout=dropout).to(self.device)


    def forward(self, x, adj):
        out = torch.einsum('ijkl,kk->ijkl',[x,adj])
        self.conv = self.conv.to(self.device)
        out = self.conv(out)
        return out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


def conv3x1(in_planes, out_planes, stride=1):
    """3x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=stride, bias=False)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class GDCD(nn.Module):
    def __init__(self, inplanes, planes, stride, bias=True, dropout=0.2):
        super(GDCD, self).__init__()

        self.conv = conv3x1(inplanes, planes, stride)
        self.dim = int(math.sqrt(inplanes * 4)) #16
        squeeze = max(inplanes * 4, self.dim ** 2) // 16 # 16
        if squeeze < 4:
            squeeze = 4
        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=bias)
        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=bias)
        self.drop_layer1 = nn.Dropout(dropout)
        self.drop_layer2 = nn.Dropout(dropout)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)
        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=bias),
            SEModule_small(squeeze))
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=bias)
        self.fc_scale = nn.Linear(squeeze, planes, bias=bias)
        self.hs = Hsigmoid()

    def forward(self, x):
        r = self.conv(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.drop_layer1(self.fc_phi(y)).view(b, self.dim, self.dim)
        scale = self.hs(self.drop_layer2(self.fc_scale(y))).view(b, -1, 1, 1)
        r = scale.expand_as(r) * r
        out = self.bn1(self.q(x))
        _, _, h, w = out.size()
        out = out.view(b, self.dim, -1)
        out = self.bn2((torch.matmul(phi, out))) + out
        out = out.view(b, -1, h, w)
        out = self.p(out) + r
        return out

if __name__ =="__main__":
    input = torch.rand((16,512,62,1))
    model = GDCD(inplanes=512, planes=512, stride=1)
    output = model(input)
    print(output.shape)
    model = GDCD(inplanes=512, planes=256, stride=1)
    output = model(input)
    print(output.shape)