import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os


class OptInit:
    def __init__(self, sub, logger):
        self.logger = logger
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')

        # base
        parser.add_argument('--use_cpu', default=False, help='use cpu?')

        # dataset args
        parser.add_argument('--data_dir', type=str, default='./40targetdata/{}/'.format(sub))
        parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default:32)')
        parser.add_argument('--sample_freq', default=1000, type=int, help='bci sample frequency')
        parser.add_argument('--down_sample', default=4, type=int, help='down sample rate')
        parser.add_argument('--eeg_channel', type=int, default=64, help='eeg_channel')
        parser.add_argument('--class_num', type=int, default=40, help='ssvep class')
        parser.add_argument('--time_win', type=int, default=0.2, help='time window')

        # train args
        parser.add_argument('--total_epochs', default=8000, type=int, help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2023, help='random seed')
        parser.add_argument('--weight_decay', type=float, default=0.001, help='L2 weight decay')#0.001
        parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')#5e-4
        parser.add_argument('--lr_decay_rate', default=0.75, type=float, help='learning rate decay')#0.75
        parser.add_argument('--optim_patience', default=3000, type=int, help='learning rate decay patience epoch')#500

        # model args
        parser.add_argument('--trans_class', default='DCD', type=str, help='{DCD, linear, normal_conv}')
        parser.add_argument('--act', default='leakyrelu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='layer', type=str, help='{batch, layer, instance} normalization')
        parser.add_argument('--bias', default=False,  type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_filters', default=128, type=int, help='number of channels of deep features')#64
        parser.add_argument('--k_adj', type=int, default=3, help='adj order')#3
        parser.add_argument('--n_blocks', default=3, type=int, help='number of basic blocks')#3
        parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')#0.5

        args = parser.parse_args()

        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        args.in_channels=int(args.sample_freq/args.down_sample*args.time_win)
        args.save_dir = os.path.join('./DDGCNN/save_model/', sub)
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



