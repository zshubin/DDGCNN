import torch
import torch.nn as nn
import torch.nn.functional as F
from DDGCNN.layers import GraphConvolution,Linear
from DDGCNN.utils import normalize_A, generate_cheby_adj


class DCDGCN(nn.Module):
    def __init__(self, num_nodes, xdim, K, num_out, dropout=0., bias=True, norm='batch', act='relu',trans_class='DCD', device='cuda'):
        super(DCDGCN, self).__init__()
        self.device = device
        self.K = K
        self.gc = nn.ModuleList()
        self.norm, self.act = None, None
        self.A = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes).to(self.device))
        nn.init.xavier_normal_(self.A)
        self.droplayer=None
        if dropout != 0.:
            self.droplayer = nn.Dropout(dropout)
        for i in range(K):
            self.gc.append(GraphConvolution(xdim, num_out, bias=bias, dropout=dropout, trans_class=trans_class,device=self.device))
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(num_out)
        if norm == 'layer':
            self.norm = nn.LayerNorm([num_out, num_nodes, 1], elementwise_affine=True)
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(num_out, affine=True)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'Leakyrelu':
            self.act = nn.LeakyReLU(0.2, inplace=False).to(self.device)
        else:
            self.act = nn.PReLU(1, 0.2).to(self.device)

    def forward(self, x):
        L = normalize_A(self.A, self.device)
        adj = generate_cheby_adj(L, self.K, self.device)
        for i in range(self.K):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        if self.norm:
            self.norm = self.norm.to(self.device)
            result = self.norm(result)
        if self.act:
            result = self.act(result)
        if self.droplayer:
            result = self.droplayer(result)
        return result

class DenseGCNNblock(nn.Module):
    def __init__(self, num_nodes, in_num_layers, out_num_layers, K, dropout, bias=True, norm='batch', act='relu',trans_class='DCD', device='cuda'):
        super(DenseGCNNblock, self).__init__()
        self.Block_seq = nn.ModuleList()
        self.Block_transfer = nn.ModuleList()
        self.Block_exp = nn.ModuleList()
        self.drop = None
        if dropout is not None and dropout > 0:
            self.drop = nn.Dropout(dropout)
        self.block1 = DCDGCN(num_nodes, in_num_layers, K, in_num_layers//2, dropout=dropout,
                        bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)
        self.block2 = DCDGCN(num_nodes, in_num_layers // 2, K, in_num_layers // 2, dropout=dropout,
                        bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)
        self.block3 = DCDGCN(num_nodes, in_num_layers // 2, K, out_num_layers, dropout=dropout,
                        bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)

    def forward(self, x):
        result = self.block1(x)
        result = self.block2(result)
        result = self.block3(result)
        result = torch.cat((x,result), dim=1)
        return result

class DenseDDGCNN(nn.Module):
    def __init__(self, xdim, k_adj, num_out, dropout, n_blocks=0, nclass=4, bias=True, norm='batch', act='relu',trans_class='DCD', device='cuda'):
        super(DenseDDGCNN, self).__init__()
        self.K = k_adj
        self.num_out = num_out
        self.n_blocks = n_blocks
        self.bias = bias
        self.norm = norm
        self.dropout = dropout
        self.act = act
        self.trans_class = trans_class
        self.device = device
        self.num_nodes = xdim[2]
        self.num_features = xdim[1]
        self.bottle_neck = DCDGCN(self.num_nodes, self.num_features, self.K, num_out, dropout=0.5,
                               bias=bias, norm=norm, act=act,trans_class=trans_class, device=device)
        self.droplayer=None
        if dropout is not None and dropout > 0:
            self.droplayer = nn.Dropout(self.dropout)

        self.fc1 = Linear(self.num_nodes * self.num_out * self.n_blocks, 64)
        self.fc2 = Linear(64, nclass)
        self.blocks = self._make_layer()

    def _make_layer(self):
        blocks = nn.ModuleList()
        for i in range(1, self.n_blocks):
            blocks.append(DenseGCNNblock(self.num_nodes, self.num_out*i, self.num_out, self.K, dropout=self.dropout,
                                       bias=self.bias, norm=self.norm, act=self.act, trans_class=self.trans_class, device=self.device))
        return blocks

    def forward(self, x):
        result = self.bottle_neck(x)
        for i in range(self.n_blocks-1):
            result = self.blocks[i](result)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        if self.droplayer:
            result = self.droplayer(result)
        result = self.fc2(result)
        return result

