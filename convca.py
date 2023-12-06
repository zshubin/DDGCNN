import torch
import torch.nn as nn
import torch.nn.functional as F


class convca(nn.Module):
    def __init__(self, channel,tw,cl):
        super(convca, self).__init__()
        self.tw = tw
        self.cl = cl
        self.conv11 = nn.Conv2d(1, 16, (9,channel),stride=(1,1),padding=(4,32))#62:(4,30) 64:(4,32)
        self.conv12 = nn.Conv2d(16, 1, (1,channel),stride=(1,1),padding=(0,31))
        self.conv13 = nn.Conv2d(1, 1, (1,channel))
        self.drop1 = nn.Dropout(0.75)

        self.conv21 = nn.Conv2d(channel, 40,(9,1),padding=(4,0))
        self.conv22 = nn.Conv2d(40,1,(9,1),padding=(4,0))
        self.drop2 = nn.Dropout(0.15)

        self.dense = nn.Linear(cl, cl)

    def corr(self, input):
        x = input[0].squeeze(dim=1)
        t = input[1].squeeze(dim=1)
        t_ = t.view(-1,self.tw,self.cl)

        corr_xt = torch.einsum('ijk,ijl->ilk',[x,t_])
        corr_xx = torch.einsum('ijk,ijk->ik',[x,x])
        corr_tt = torch.einsum('ijl,ijl->il',[t_,t_])
        corr = torch.squeeze(corr_xt)/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)
        return corr

    def forward(self, sig, temp):
        sig = self.conv11(sig)
        sig = self.conv12(sig)
        sig = self.conv13(sig)
        sig = self.drop1(sig)

        temp = self.conv21(temp)
        temp = self.conv22(temp)
        temp = self.drop2(temp)

        corr = self.corr([sig, temp])

        out = self.dense(corr)
        return out




