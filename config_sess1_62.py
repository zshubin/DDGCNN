import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import logging.config
import os
import sys


class OptInit:
    def __init__(self, sub, logger,sess):
        self.logger = logger
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')

        # base
        parser.add_argument('--use_cpu', default=False, help='use cpu?')

        # dataset args
        parser.add_argument('--data_dir', type=str, default='./processed_ssvep_data/{}/{}/'.format(sess,sub))
        parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default:32)')                             #32
        parser.add_argument('--sample_freq', default=1000, type=int, help='bci sample frequency')
        parser.add_argument('--down_sample', default=4, type=int, help='down sample rate')

        # train args
        parser.add_argument('--total_epochs', default=4000, type=int, help='number of total epochs to run')                        #2600
        parser.add_argument('--save_freq', default=1, type=int, help='save model per num of epochs')
        parser.add_argument('--iter', default=0, type=int, help='number of iteration to start')
        parser.add_argument('--seed', type=int, default=2022, help='random seed')
        parser.add_argument('--weight_decay', type=float, default=0.001, help='L2 weight decay')                                   #0.001
        parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate')                                        #1e-3
        parser.add_argument('--lr_decay_rate', default=0.5, type=float, help='learning rate decay')                               #0.3/0.75
        parser.add_argument('--optim_patience', default=200, type=int, help='learning rate decay patience epoch')                  #200
        parser.add_argument('--time_win', type=int, default=0.8, help='time window')

        # model args
        parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default=None)
        parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
        parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
        parser.add_argument('--act', default='leakyrelu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='layer', type=str, help='{batch, layer, instance} normalization')
        parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')       #32
        parser.add_argument('--k_adj', type=int, default=3, help='adj order')
        parser.add_argument('--n_blocks', default=3, type=int, help='number of basic blocks')
        parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
        parser.add_argument('--dropedge', default=0., type=float, help='ratio of dropedge')
        parser.add_argument('--res_scale', default=0.2, type=float, help='ratio of residual module')
        parser.add_argument('--dropedge_norm_style', default='AugNormAdj', type=str, help='Normalization way of drop adj-matrix')

        args = parser.parse_args()

        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        args.in_channels=int(args.sample_freq/args.down_sample)
        args.save_dir = os.path.join('./tCNN62_session1/', sub)
        self.args = args
        self._set_seed(self.args.seed)

        self.args.writer = SummaryWriter(log_dir=self.args.save_dir + '/log/', comment='comment',
                                         filename_suffix="_test_your_filename_suffix")
        # loss
        self.args.epoch = 0
        self.args.step = -1

        # self._configure_logger()
        self._print_args()

    def get_args(self):
        return self.args

    def _print_args(self):
        self.logger.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            self.logger.info("{}:{}".format(arg, content))
        self.logger.info("==========     args END    =============")
        self.logger.info("\n")



    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



