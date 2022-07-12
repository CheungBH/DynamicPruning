import argparse
import os.path
import sys
import torch
import models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import tqdm
import utils.flopscounter as flopscounter
import utils.utils as utils
import utils.viz as viz
from torch.backends import cudnn as cudnn
from simple_args import SimpleArguments
cudnn.benchmark = True
device = 'cuda'



def main():
    global iteration
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with sparse masks')

    # basic
    parser.add_argument('--dataset-root', default='/esat/visicsrodata/datasets/ilsvrc2012/', type=str, metavar='PATH',
                        help='ImageNet dataset root')
    parser.add_argument('--batchsize', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--budget', default=-1, type=float,
                        help='computational budget (between 0 and 1) (-1 for no sparsity)')
    parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
    parser.add_argument('--res', default=224, type=int, help='number of epochs')

    # learning strategy
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=[30, 60, 90], nargs='+', type=int, help='learning rate decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--optim', type=str, default='sgd', help='network model name')
    parser.add_argument('--scheduler', type=str, default='step', help='network model name')

    # loss
    parser.add_argument('--net_weight', default=10, type=float, help='weight of network sparsity')
    parser.add_argument('--layer_weight', default=10, type=float, help='weight of layer sparsity')
    parser.add_argument('--sparse_strategy', type=str, default='static', help='Type of mask')
    parser.add_argument('--valid_range', type=float, default=0.33, help='Type of mask')
    parser.add_argument('--static_range', type=float, default=0, help='Type of mask')
    parser.add_argument('--layer_loss_method', type=str, default='flops', help='Calculation for layer-wise methods')
    parser.add_argument('--unlimited_lower', action='store_true', help='loss without lower constraints')

    # channel arguments
    parser.add_argument('--group_size', type=int, default=1, help='The number for grouping channel pruning')
    parser.add_argument('--pooling_method', type=str, default='max', help='Maxpooling or AveragePooling')
    parser.add_argument('--channel_budget', default=-1, type=float,
                        help='computational budget (between 0 and 1) (-1 for no sparsity)')
    parser.add_argument('--channel_unit_type', type=str, default='fc', help='Type of mask')
    parser.add_argument('--channel_stage', nargs="+", type=int, help='target stage for pretrain mask')
    parser.add_argument('--lasso_lambda', type=float, default=1e-8)

    # model
    parser.add_argument('--model', type=str, default='resnet101', help='network model name')
    parser.add_argument('--model_cfg', type=str, default='baseline', help='network model name')
    parser.add_argument('--conv1_act', type=str, default='relu', help='the activation function of conv1')
    parser.add_argument('--resolution_mask', action='store_true', help='share a mask within a same resolution')
    # Negative value for directly skip; Positive for using formula to skip
    parser.add_argument('--mask_type', type=str, default='conv', help='Type of mask')
    parser.add_argument('--mask_kernel', default=3, type=int, help='number of epochs')
    parser.add_argument('--no_attention', action='store_true', help='run without attention')
    parser.add_argument('--input_resolution', action='store_true',
                        help='The mask resolution is based on the input size')
    # special args
    parser.add_argument('--mask_thresh', default=0.5, type=float, help='The numerical threshold of mask')
    parser.add_argument('--target_stage', nargs="+", type=int, help='target stage for pretrain mask')
    parser.add_argument('--random_mask_stage', nargs="+", type=int, default=[-1], help='target stage for pretrain mask')
    parser.add_argument('--individual_forward', action='store_true',
                        help='for stat mask: Treating each sample individually')
    parser.add_argument('--skip_layer_thresh', default=0, type=float, help='The numerical threshold of mask')
    parser.add_argument('--dropout_stages', nargs="+", type=int, default=[-1], help='target stage for pretrain mask')
    parser.add_argument('--dropout_ratio', default=0, type=float, help='The numerical threshold of mask')
    # mobilenet args
    parser.add_argument('--final_activation', default="linear", type=str, help='The numerical threshold of mask')
    parser.add_argument('--use_downsample', action='store_true', help='run without attention')

    # file management
    parser.add_argument('-s', '--save_dir', type=str, default='', help='directory to save model')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--load', type=str, default='', help='load model path')
    parser.add_argument('--auto_resume', action='store_true', help='plot ponder cost')

    # evaluation
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    parser.add_argument('--feat_save_dir', default='', help='plot ponder cost')
    parser.add_argument('--plot_save_dir', default='', help='plot ponder cost')

    # simple arguments

    parser.add_argument('--model_args', type=str, default='', help='load model path')
    parser.add_argument('--loss_args', type=str, default='', help='load model path')
    args = parser.parse_args()
    print('Args:', args)
    args = SimpleArguments().update(args)
    res = args.res

    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0, model_cfg=args.model_cfg, resolution_mask=args.resolution_mask,
                       mask_type=args.mask_type, momentum=args.momentum, budget=args.budget,
                       mask_kernel=args.mask_kernel, no_attention=args.no_attention,
                       individual_forward=args.individual_forward, save_feat=args.feat_save_dir,
                       target_stage=args.target_stage, mask_thresh=args.mask_thresh,
                       random_mask_stage=args.random_mask_stage, skip_layer_thresh=args.skip_layer_thresh,
                       input_resolution=args.input_resolution, conv1_act=args.conv1_act, group_size=args.group_size,
                       pooling_method=args.pooling_method, channel_budget=args.channel_budget,
                       channel_unit_type=args.channel_unit_type, channel_stage=args.channel_stage,
                       dropout_stages=args.dropout_stages, dropout_ratio=args.dropout_ratio,
                       use_downsample=args.use_downsample, final_activation=args.final_activation).to(device=device)

    meta = {'masks': [], 'device': device, 'gumbel_temp': 5.0, 'gumbel_noise': False, 'epoch': 0,
            "feat_before": [], "feat_after": [], "lasso_sum": 0, "channel_prediction": {}}

    bs = 6

    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()
    if args.budget != -1:
        update_sparsity(args, model)
    _ = model(torch.rand((bs, 3, res, res)).cuda(), meta)
    model.stop_flops_count()
    if args.save_dir:
        with open(args.save_dir, "a+") as f:
            if len(sys.argv) < 15:
                print("{}: Baseline ".format(sys.argv[4]), file=f)
            else:
                print("{}: s{}-c{} ".format(sys.argv[4], sys.argv[6], sys.argv[8]), file=f)
            print(model.compute_average_flops_cost()[0], file=f)
            f.write("\n")
    else:
        print(model.compute_average_flops_cost()[0])


def update_sparsity(args, model):
    spatial_sparsity, channel_sparsity = args.budget, args.channel_budget
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if "layer" in name and "downsample" not in name and "masker" not in name:
                module.__mask__ = spatial_sparsity
            # if "layer" in name:
            #     if "conv2" in name:
            #         module.__input_ratio__ = channel_sparsity
            #         module.__output_ratio__ = channel_sparsity
            #     elif "conv1" in name:
            #         module.__output_ratio__ = channel_sparsity
            #     elif "conv3" in name:
            #         module.__input_ratio__ = channel_sparsity
            #     else:
            #         pass


if __name__ == "__main__":
    main()
