import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DDGCNN.normalization import fetch_normalization


def normalize_A(A, device='cuda', symmetry=False):
    A = F.relu(A).to(device)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K,device='cuda'):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(device))
        elif i == 1:
            support.append(A.to(device))
        else:
            temp = torch.matmul(support[-1].cuda(), A.to(device))
            support.append(temp.to(device))
    return support


def randomedge_drop(percent, adj, normalization):
    """
    Randomly drop edge and drop percent% edges.
    """
    "Opt here"
    coo_a = adj.to_sparse()
    indices = coo_a._indices()
    nnz = coo_a._nnz()
    perm = torch.randperm(nnz)
    preserve_nnz = int(nnz * (1.-percent))
    perm = perm[:preserve_nnz]
    i = indices[:, perm]
    v = coo_a._values()[perm]
    r_adj = torch.sparse_coo_tensor(i, v, adj.shape).to_dense()
    r_adj = preprocess_adj(normalization, r_adj)
    return r_adj


def preprocess_adj(normalization, adj):
    adj_normalizer = fetch_normalization(normalization)
    r_adj = adj_normalizer(adj)
    return r_adj


if __name__ == '__main__':
    adj = torch.rand((64, 64))
    percent = 0.2
    normalization = 'AugNormAdj'
    adj = randomedge_drop(percent, adj, normalization)
