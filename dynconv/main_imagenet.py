import argparse
import os.path
import sys
import matplotlib.pyplot as plt
import h5py, shutil

import dataloader.imagenet
import dynconv
import torch
import models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import tqdm
import utils.flopscounter as flopscounter
import utils.logger as logger
import utils.utils as utils
import utils.viz as viz
from torch.backends import cudnn as cudnn
from simple_args import SimpleArguments

from apex import amp
mix_precision = False

cudnn.benchmark = True
device='cuda'
iteration = 0


def main():
    global iteration
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with sparse masks')

    # basic
    parser.add_argument('--dataset-root', default='/esat/visicsrodata/datasets/ilsvrc2012/', type=str, metavar='PATH',
                    help='ImageNet dataset root')
    parser.add_argument('--batchsize', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--budget', default=-1, type=float, help='computational budget (between 0 and 1) (-1 for no sparsity)')
    parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
    parser.add_argument('--res', default=224, type=int, help='number of epochs')

    # learning strategy
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=[30,60,90], nargs='+', type=int, help='learning rate decay epochs')
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
    parser.add_argument('--channel_budget', default=-1, type=float, help='computational budget (between 0 and 1) (-1 for no sparsity)')
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
    parser.add_argument('--input_resolution', action='store_true', help='The mask resolution is based on the input size')
    # special args
    parser.add_argument('--mask_thresh', default=0.5, type=float, help='The numerical threshold of mask')
    parser.add_argument('--target_stage', nargs="+", type=int, help='target stage for pretrain mask')
    parser.add_argument('--random_mask_stage', nargs="+", type=int, default=[-1], help='target stage for pretrain mask')
    parser.add_argument('--individual_forward', action='store_true', help='for stat mask: Treating each sample individually')
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
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
            "feat_before": [], "feat_after": [], "lasso_sum": torch.zeros(1).cuda(), "channel_prediction": {}}
    model.eval()
    _ = model(torch.rand((2, 3, res, res)).cuda(), meta)


    ## CRITERION
    class Loss(nn.Module):
        def __init__(self, budget, net_weight, block_weight, tensorboard_folder="", channel_budget=0, **kwargs):
            super(Loss, self).__init__()
            self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.sparsity_loss = dynconv.SparsityCriterion(args.budget, **kwargs) \
                if args.budget >= 0 and "stat" not in args.mask_type else None
            self.budget = budget
            self.net_weight = net_weight
            self.block_weight = block_weight
            if tensorboard_folder:
                os.makedirs(tensorboard_folder, exist_ok=True)
            self.tb_writer = SummaryWriter(tensorboard_folder) if tensorboard_folder else ""
            self.channel_budget = channel_budget

        def forward(self, output, target, meta, phase="train"):
            global iteration
            task_loss, loss_block, loss_net, spatial_percents = self.task_loss(output, target), torch.zeros(1).cuda(), \
                                                              torch.zeros(1).cuda(), []
            if self.sparsity_loss is not None:
                loss_net, loss_block, spatial_percents = self.sparsity_loss(meta)
            spatial_loss = loss_block * self.block_weight + loss_net * self.net_weight
            channel_loss, channel_percents = self.get_channel_loss(meta)
            
            if self.tb_writer and phase == "train":
                self.tb_writer.add_scalar("{}/task loss".format(phase), task_loss, iteration)
                self.tb_writer.add_scalar("{}/network loss".format(phase), loss_net, iteration)
                self.tb_writer.add_scalar("{}/block loss".format(phase), loss_block, iteration)
                self.tb_writer.add_scalar("{}/channel loss".format(phase), channel_loss, iteration)
                iteration += 1

            return task_loss, spatial_loss, spatial_percents, channel_loss, channel_percents

        def get_channel_loss(self, meta):
            channel_loss, channel_percents = torch.zeros(1).cuda(), []
            if 0 < self.channel_budget < 1:
                for _, vector in meta["channel_prediction"].items():
                    layer_percent = torch.true_divide(vector.sum(), vector.numel())
                    channel_percents.append(layer_percent)
                    assert layer_percent >= 0 and layer_percent <= 1, layer_percent
                    channel_loss += max(0, layer_percent - self.channel_budget) ** 2
            else:
                channel_loss = meta["lasso_sum"]
            return channel_loss, channel_percents

    tb_folder = os.path.join(args.save_dir, "tb") if not args.evaluate else ""
    channel_gumbel = args.channel_budget if "gumbel" in args.channel_unit_type else -1
    criterion = Loss(args.budget, net_weight=args.net_weight, block_weight=args.layer_weight, num_epochs=args.epochs,
                     strategy=args.sparse_strategy, valid_range=args.valid_range, static_range=args.static_range,
                     tensorboard_folder=tb_folder, unlimited_lower=args.unlimited_lower,
                     layer_loss_method=args.layer_loss_method, channel_budget=channel_gumbel)

    if not args.evaluate:
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(res),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if len(args.save_dir) > 0:
            if not os.path.exists(os.path.join(args.save_dir)):
                os.makedirs(os.path.join(args.save_dir))
        ## DATA
        trainset = dataloader.imagenet.IN1K(root=args.dataset_root, split='train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)
        args.feat_save_dir = ""
    else:
        args.save_dir = ""
        if args.plot_save_dir:
            os.makedirs(args.plot_save_dir, exist_ok=True)
            args.batchsize = 1
        if args.feat_save_dir:
            os.makedirs(args.feat_save_dir, exist_ok=True)
            args.batchsize = 1

    transform_val = transforms.Compose([
        transforms.Resize(int(res / 0.875)),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        normalize,
    ])

    valset = dataloader.imagenet.IN1K(root=args.dataset_root, split='val', transform=transform_val)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)

    file_path = os.path.join(args.save_dir, "log.txt")
    if args.save_dir:
        cmd = utils.generate_cmd(sys.argv[1:])
        with open(file_path, "a+") as f:
            f.write(cmd + "\n")
            print('Args:', args, file=f)
            f.write("\n")

    ## OPTIMIZER
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    ## CHECKPOINT
    start_epoch, best_prec1 = -1, 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            # print('check', checkpoint)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            try:
                iteration = checkpoint['iteration']
            except:
                iteration = args.batchsize * start_epoch
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)
    elif args.auto_resume:
        assert args.save_dir, "Please specify the auto resuming folder"
        resume_path = os.path.join(args.save_dir, "checkpoint.pth")
        if os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path)
            # print('check', checkpoint)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{resume_path}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(resume_path)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)
    elif args.load:
        try:
            checkpoint_dict = torch.load(args.load, map_location=device)['state_dict']
        except:
            checkpoint_dict = torch.load(args.load, map_location=device)

        model_dict = model.state_dict()
        # update_dict = {k: v for k, v in model_dict.items() if k in checkpoint_dict.keys()}
        update_keys = [k for k, v in model_dict.items() if k in checkpoint_dict.keys()]
        update_dict = {k: v for k, v in checkpoint_dict.items() if k in update_keys}
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)

    if args.scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_decay, last_epoch=start_epoch)
    elif args.scheduler == "exp":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(args.lr_decay[0]), last_epoch=start_epoch)
    elif args.scheduler == "cosine_anneal_warmup":
        pass
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, gamma=float(args.lr_decay[0]), last_epoch=start_epoch)
    else:
        raise NotImplementedError

    start_epoch += 1
            
    ## Count number of params
    print("* Number of trainable parameters:", utils.count_parameters(model))

    if mix_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    ## EVALUATION
    if args.evaluate:
        # evaluate on validation set
        print(f"########## Evaluation ##########")
        prec1, MMac = validate(args, val_loader, model, criterion, start_epoch)
        if args.mask_type == "stat_mom":
            print("The threshold for each layer is {}".format(
                ",".join([str(round(v.data.squeeze().tolist(), 4)) for k, v in model.named_parameters()
                          if "threshold" in k])))
        return

    ## TRAINING
    best_epoch, best_MMac = start_epoch, -1
    for epoch in range(start_epoch, args.epochs):
        print(f"########## Epoch {epoch} ##########")

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, file_path)
        if args.scheduler != "cosine_anneal_warmup":
            lr_scheduler.step()

        # evaluate on validation set
        prec1, MMac = validate(args, val_loader, model, criterion, epoch, file_path)
        if args.mask_type == "stat_mom":
            print("The threshold for each layer is {}".format(
                ",".join([str(round(v.data.squeeze().tolist(), 4)) for k, v in model.named_parameters()
                          if "threshold" in k])))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_epoch, best_MMac = epoch, MMac
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
            "iteration": iteration
        }, folder=args.save_dir, is_best=is_best)
        with open(file_path, "a+") as f:
            print(f" *Currently Best prec1: {best_prec1}\n-------------------------------------------------\n", file=f)

    with open(file_path, "a+") as f:
        print(f" * Best prec1: {best_prec1}, Epoch {best_epoch}, MMac {best_MMac}", file=f)



def train(args, train_loader, model, criterion, optimizer, epoch, file_path):
    """
    Run one train epoch
    """
    model.train()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    task_loss_record = utils.AverageMeter()
    spatial_loss_record = utils.AverageMeter()
    channel_loss_record = utils.AverageMeter()
    layer_cnt = utils.layer_count(args)

    spatial_sparsity_records = [utils.AverageMeter() for _ in range(layer_cnt)]
    channel_sparsity_records = [utils.AverageMeter() for _ in range(layer_cnt)]

    if args.scheduler == "cosine_anneal_warmup":
        utils.adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=args.epochs, lr_min=0.00001,
                             lr_max=args.lr)
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

    if epoch < 0.5 * args.epochs:
        gumbel_temp = 2.5
    elif epoch < 0.8 * args.epochs:
        gumbel_temp = 1
    else:
        gumbel_temp = 0.6667
    gumbel_noise = False if epoch > 0.8*args.epochs else True

    num_step =  len(train_loader)
    for input, target, _ in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch,
                "lasso_sum": torch.zeros(1).cuda(), "channel_prediction": {}}
        output, meta = model(input, meta)
        t_loss, s_loss, s_percents, c_loss, c_percents = criterion(output, target, meta)
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        task_loss_record.update(t_loss.item(), input.size(0))
        spatial_loss_record.update(s_loss.item(), input.size(0))
        channel_loss_record.update(c_loss.item(), input.size(0))

        for s_per, recorder in zip(s_percents, spatial_sparsity_records):
            recorder.update(s_per.item(), 1)
        for c_per, recorder in zip(c_percents, channel_sparsity_records):
            recorder.update(c_per.item(), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss = s_loss + t_loss + args.lasso_lambda * c_loss #if s_loss else t_loss

        if mix_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

    spatial_layer_str = ",".join([str(round(recorder.avg, 4)) for recorder in spatial_sparsity_records])
    channel_layer_str = ",".join([str(round(recorder.avg, 4)) for recorder in channel_sparsity_records])
    # print("* Spatial Percentage are: {}".format(spatial_layer_str))
    # print("* Channel Percentage are: {}".format(spatial_layer_str))
    if file_path:
        with open(file_path, "a+") as f:
            logger.tick(f)
            f.write("Train: Epoch {}, Prec@1 {}, Prec@5 {}, task loss {}, sparse loss {}\n".format
                    (epoch, round(top1.avg, 4), round(top5.avg, 4), round(task_loss_record.avg, 4),
                     round(spatial_loss_record.avg, 4)))
            f.write("Train Spatial Percentage: {}\n".format(spatial_layer_str))
            f.write("Train Channel Percentage: {}\n".format(channel_layer_str))

    if criterion.tb_writer:
        criterion.tb_writer.add_scalar("train/TASK LOSS-EPOCH", task_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar("train/Prec@1-EPOCH", top1.avg, epoch)
        criterion.tb_writer.add_scalar('train/SPATIAL LOSS-EPOCH', spatial_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar('train/CHANNEL LOSS-EPOCH', channel_loss_record.avg, epoch)
        # criterion.tb_writer.add_scalar("train/MMac-EPOCH", model.compute_average_flops_cost()[0]/1e6, epoch)
        for idx, recorder in enumerate(spatial_sparsity_records):
            criterion.tb_writer.add_scalar("train/SPATIAL LAYER {}-EPOCH".format(idx+1), recorder.avg, epoch)
        for idx, recorder in enumerate(channel_sparsity_records):
            criterion.tb_writer.add_scalar("train/CHANNEL LAYER {}-EPOCH".format(idx+1), recorder.avg, epoch)


def validate(args, val_loader, model, criterion, epoch, file_path=None):
    """
    Run evaluation
    """
    top1, top5 = utils.AverageMeter(), utils.AverageMeter()
    task_loss_record = utils.AverageMeter()
    spatial_loss_record = utils.AverageMeter()
    channel_loss_record = utils.AverageMeter()
    layer_cnt = utils.layer_count(args)

    spatial_sparsity_records = [utils.AverageMeter() for _ in range(layer_cnt)]
    channel_sparsity_records = [utils.AverageMeter() for _ in range(layer_cnt)]

    channel_files = []
    target_stages = [(3, 1), (3, 2)]

    def record_channels(img_path, channels, channel_files):
        for channel_file, stage in zip(channel_files, target_stages):
            for path, channel in zip(img_path, channels[stage]):
                channel_file[path.split("/")[-1]] = channel.detach().cpu()

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target, img_path in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            for file in channel_files:
                file.close()
            channel_files = []
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch,
                    "feat_before": [], "feat_after": [], "lasso_sum": torch.zeros(1).cuda(), "channel_prediction": {}}
            output, meta = model(input, meta)
            record_channels(img_path, meta['channel_prediction'], channel_files)
            output = output.float()
            t_loss, s_loss, s_percents, c_loss, c_percents = criterion(output, target, meta)
            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            task_loss_record.update(t_loss.item(), input.size(0))
            spatial_loss_record.update(s_loss.item(), input.size(0))
            channel_loss_record.update(c_loss.item(), input.size(0))

            for s_per, recorder in zip(s_percents, spatial_sparsity_records):
                recorder.update(s_per.item(), 1)
            for c_per, recorder in zip(c_percents, channel_sparsity_records):
                recorder.update(c_per.item(), 1)
            # measure accuracy and record loss
            # prec1 = utils.accuracy(output.data, target)[0]
            # top1.update(prec1.item(), input.size(0))

            if args.feat_save_dir:
                viz.save_feat(meta["feat_before"], args.feat_save_dir, img_path[0].split("/")[-1], "before")
                viz.save_feat(meta["feat_after"], args.feat_save_dir, img_path[0].split("/")[-1], "after")

            if args.plot_ponder:
                if args.plot_save_dir and os.path.exists(os.path.join(args.plot_save_dir, img_path[0].split("/")[-1])):
                    pass
                else:
                    save_path = os.path.join(args.plot_save_dir, img_path[0].split("/")[-1].split(".")[0]) \
                        if args.plot_save_dir else ""
                    os.makedirs(save_path, exist_ok=True)
                    shutil.copy(os.path.join(args.dataset_root, img_path[0]), os.path.join(save_path, "raw_image.jpg"))
                    # viz.plot_image(input, save_path)
                    viz.plot_paper_masks(meta['masks'], save_path)
                    # viz.plot_ponder_cost(meta['masks'])
                    if args.resolution_mask:
                        viz.plot_masks(meta['masks'], save_path=os.path.join(save_path, "mask_sum.jpg"))
                    else:
                        viz.plot_masks(meta['masks'], save_path=os.path.join(save_path, "mask_sum.jpg"), WIDTH=4)
                    viz.showKey()

    print(f'* Epoch {epoch} - Prec@1 {top1.avg:.3f} - Prec@5 {top5.avg:.3f}')
    print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost()[0]/1e6:.6f} MMac')
    spatial_layer_str = ",".join([str(round(recorder.avg, 4)) for recorder in spatial_sparsity_records])
    channel_layer_str = ",".join([str(round(recorder.avg, 4)) for recorder in channel_sparsity_records])
    print("* Spatial Percentage are: {}".format(spatial_layer_str))
    print("* Channel Percentage are: {}".format(channel_layer_str))
    model.stop_flops_count()
    if file_path:
        with open(file_path, "a+") as f:
            f.write("Validation: Epoch {}, Prec@1 {}, Prec@5 {}, task loss {}, sparse loss {}, ave FLOPS per image: {} MMac\n".
                    format(epoch, round(top1.avg, 4), round(top1.avg, 4), round(top5.avg, 4),
                           round(task_loss_record.avg, 4), round(spatial_loss_record.avg, 4),
                           round(model.compute_average_flops_cost()[0]/1e6), 6))
            f.write("Validation Spatial percentage: {}\n".format(spatial_layer_str))
            f.write("Validation Channel percentage: {}\n".format(channel_layer_str))

    if criterion.tb_writer:
        criterion.tb_writer.add_scalar("valid/TASK LOSS-EPOCH", task_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar("valid/Prec@1-EPOCH", top1.avg, epoch)
        criterion.tb_writer.add_scalar('valid/SPATIAL LOSS-EPOCH', spatial_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar('valid/CHANNEL LOSS-EPOCH', channel_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar("valid/MMac-EPOCH", model.compute_average_flops_cost()[0]/1e6, epoch)
        for idx, recorder in enumerate(spatial_sparsity_records):
            criterion.tb_writer.add_scalar("valid/SPATIAL LAYER {}-EPOCH".format(idx+1), recorder.avg, epoch)
        for idx, recorder in enumerate(channel_sparsity_records):
            criterion.tb_writer.add_scalar("valid/CHANNEL LAYER {}-EPOCH".format(idx+1), recorder.avg, epoch)

    return top1.avg, model.compute_average_flops_cost()[0]/1e6


if __name__ == "__main__":
    main()    
