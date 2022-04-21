import argparse
import os.path

import matplotlib.pyplot as plt

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


from apex import amp
mix_precision = True

cudnn.benchmark = True
device='cuda'
iteration = 0


def main():
    global iteration
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with sparse masks')
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=[30,60,90], nargs='+', help='learning rate decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--sparse_weight', default=10, type=float, help='weight of network sparsity')
    parser.add_argument('--layer_weight', default=10, type=float, help='weight of layer sparsity')
    parser.add_argument('--sparse_strategy', type=str, default='static', help='Type of mask')
    parser.add_argument('--valid_range', type=float, default=0.33, help='Type of mask')
    parser.add_argument('--static_range', type=float, default=0.2, help='Type of mask')
    parser.add_argument('--target_stage', nargs="+", type=int, help='target stage for pretrain mask')
    parser.add_argument('--batchsize', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--mask_thresh', default=0.5, type=float, help='The numerical threshold of mask')

    parser.add_argument('--model', type=str, default='resnet101', help='network model name')
    parser.add_argument('--model_cfg', type=str, default='baseline', help='network model name')
    parser.add_argument('--load', type=str, default='', help='load model path')
    parser.add_argument('--mask_type', type=str, default='conv', help='Type of mask')
    parser.add_argument('--mask_kernel', default=3, type=int, help='number of epochs')
    parser.add_argument('--no_attention', action='store_true', help='run without attention')
    parser.add_argument('--individual_forward', action='store_true', help='run without attention')

    parser.add_argument('--budget', default=-1, type=float, help='computational budget (between 0 and 1) (-1 for no sparsity)')
    parser.add_argument('-s', '--save_dir', type=str, default='', help='directory to save model')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--dataset-root', default='/esat/visicsrodata/datasets/ilsvrc2012/', type=str, metavar='PATH',
                    help='ImageNet dataset root')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    parser.add_argument('--resolution_mask', action='store_true', help='share a mask within a same resolution')
    parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    parser.add_argument('--feat_save_dir', default='', help='plot ponder cost')
    parser.add_argument('--plot_save_dir', default='', help='plot ponder cost')
    parser.add_argument('--auto_resume', action='store_true', help='plot ponder cost')
    parser.add_argument('--optim', type=str, default='sgd', help='network model name')
    parser.add_argument('--scheduler', type=str, default='step', help='network model name')
    parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
    args =  parser.parse_args()
    print('Args:', args)

    res = 224

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0, model_cfg=args.model_cfg, resolution_mask=args.resolution_mask,
                       mask_type=args.mask_type, momentum=args.momentum, budget=args.budget,
                       mask_kernel=args.mask_kernel, no_attention=args.no_attention,
                       individual_forward=args.individual_forward, save_feat=args.feat_save_dir,
                       target_stage=args.target_stage, mask_thresh=args.mask_thresh).to(device=device)

    meta = {'masks': [], 'device': device, 'gumbel_temp': 5.0, 'gumbel_noise': False, 'epoch': 0,
            "feat_before": [], "feat_after": []}
    _ = model(torch.rand((1, 3, res, res)).cuda(), meta)


    ## CRITERION
    class Loss(nn.Module):
        def __init__(self, budget, net_weight, block_weight, tensorboard_folder="", **kwargs):
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

        def forward(self, output, target, meta, phase="train"):
            global iteration
            task_loss, loss_block, loss_net, layer_percents = self.task_loss(output, target), torch.zeros(1).cuda(), \
                                                              torch.zeros(1).cuda(), []
            if self.sparsity_loss is not None:
                loss_net, loss_block, layer_percents = self.sparsity_loss(meta)
            sparse_loss = loss_block * self.block_weight + loss_net * self.net_weight

            if self.tb_writer and phase == "train":
                self.tb_writer.add_scalar("{}/task loss".format(phase), task_loss, iteration)
                self.tb_writer.add_scalar("{}/network loss".format(phase), loss_net, iteration)
                self.tb_writer.add_scalar("{}/block loss".format(phase), loss_block, iteration)
                iteration += 1
            return task_loss, sparse_loss, layer_percents

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
        tb_folder = os.path.join(args.save_dir, "tb")
        args.feat_save_dir = ""
    else:
        tb_folder = ""
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
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=False)

    file_path = os.path.join(args.save_dir, "log.txt")
    criterion = Loss(args.budget, net_weight=args.sparse_weight, block_weight=args.layer_weight, num_epochs=args.epochs,
                     strategy=args.sparse_strategy, valid_range=args.valid_range, static_range=args.static_range,
                     tensorboard_folder=tb_folder)

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
    else:
        raise NotImplementedError

    start_epoch += 1
            
    ## Count number of params
    print("* Number of trainable parameters:", utils.count_parameters(model))

    ## EVALUATION
    if args.evaluate:
        # evaluate on validation set
        print(f"########## Evaluation ##########")
        prec1, MMac = validate(args, val_loader, model, criterion, start_epoch)
        return

    if mix_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        
    ## TRAINING
    best_epoch, best_MMac = start_epoch, -1
    for epoch in range(start_epoch, args.epochs):
        print(f"########## Epoch {epoch} ##########")

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(args, train_loader, model, criterion, optimizer, epoch, file_path)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, MMac = validate(args, val_loader, model, criterion, epoch, file_path)

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
    task_loss_record = utils.AverageMeter()
    sparse_loss_record = utils.AverageMeter()
    layer_sparsity_records = [utils.AverageMeter() for _ in range(16)]

    if float(args.lr_decay[0]) > 1:
        if epoch < args.lr_decay[0]:
            gumbel_temp = 5.0
        elif epoch < args.lr_decay[1]:
            gumbel_temp = 2.5
        else:
            gumbel_temp = 1
    else:
        if epoch < 0.5*args.epochs:
            gumbel_temp = 5.0
        elif epoch < 0.8*args.epochs:
            gumbel_temp = 2.5
        else:
            gumbel_temp = 1
    gumbel_noise = False if epoch > 0.8*args.epochs else True

    num_step =  len(train_loader)
    for input, target, _ in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        output, meta = model(input, meta)
        t_loss, s_loss, layer_percents = criterion(output, target, meta)
        prec1 = utils.accuracy(output.data, target)[0]
        top1.update(prec1.item(), input.size(0))
        task_loss_record.update(t_loss.item(), input.size(0))
        sparse_loss_record.update(s_loss.item(), input.size(0))

        for layer_per, recorder in zip(layer_percents, layer_sparsity_records):
            recorder.update(layer_per.item(), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss = s_loss + t_loss if s_loss else t_loss

        if mix_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

    layer_str = ",".join([round(recorder.avg, 4) for recorder in layer_sparsity_records])
    if file_path:
        with open(file_path, "a+") as f:
            logger.tick(f)
            f.write("Train: Epoch {}, Prec@1 {}, task loss {}, sparse loss {}\n".format
                    (epoch, round(top1.avg, 4), round(task_loss_record.avg, 4), round(sparse_loss_record.avg, 4)))
            f.write("Train Layer Percentage: {}".format(layer_str))

    if criterion.tb_writer:
        criterion.tb_writer.add_scalar("train/TASK LOSS-EPOCH", task_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar("train/Prec@1-EPOCH", top1.avg, epoch)
        criterion.tb_writer.add_scalar('train/SPARSE LOSS-EPOCH', sparse_loss_record.avg, epoch)
        for idx, recorder in enumerate(layer_sparsity_records):
            criterion.tb_writer.add_scalar("train/LAYER {}-EPOCH".format(idx+1), recorder.avg, epoch)


def validate(args, val_loader, model, criterion, epoch, file_path=None):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()
    task_loss_record = utils.AverageMeter()
    sparse_loss_record = utils.AverageMeter()
    layer_sparsity_records = [utils.AverageMeter() for _ in range(16)]

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target, img_path in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch,
                    "feat_before": [], "feat_after": []}
            output, meta = model(input, meta)
            output = output.float()
            t_loss, s_loss, layer_percents = criterion(output, target, meta, phase="")
            task_loss_record.update(t_loss.item(), input.size(0))
            sparse_loss_record.update(s_loss.item(), input.size(0))
            for layer_per, recorder in zip(layer_percents, layer_sparsity_records):
                recorder.update(layer_per.item(), 1)

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))

            if args.feat_save_dir:
                viz.save_feat(meta["feat_before"], args.feat_save_dir, img_path[0].split("/")[-1], "before")
                viz.save_feat(meta["feat_after"], args.feat_save_dir, img_path[0].split("/")[-1], "after")

            if args.plot_ponder:
                if args.plot_save_dir and os.path.exists(os.path.join(args.plot_save_dir, img_path[0].split("/")[-1])):
                    pass
                else:
                    viz.plot_image(input)
                    try:
                        viz.plot_ponder_cost(meta['masks'])
                    except:
                        pass
                    save_path = os.path.join(args.plot_save_dir, img_path[0].split("/")[-1]) if args.plot_save_dir else ""
                    if args.resolution_mask:
                        viz.plot_masks(meta['masks'], save_path=save_path)
                    else:
                        viz.plot_masks(meta['masks'], save_path=save_path, WIDTH=4)
                    viz.showKey()

    print(f'* Epoch {epoch} - Prec@1 {top1.avg:.3f}')
    print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost()[0]/1e6:.6f} MMac')
    layer_str = ",".join([round(recorder.avg, 4) for recorder in layer_sparsity_records])
    print("* Layer Percentage are: {}".format(layer_str))
    model.stop_flops_count()
    if file_path:
        with open(file_path, "a+") as f:
            f.write("Validation: Epoch {}, Prec@1 {}, task loss {}, sparse loss {}, ave FLOPS per image: {} MMac\n".
                    format(epoch, round(top1.avg, 4), round(task_loss_record.avg, 4), round(sparse_loss_record.avg, 4),
                           round(model.compute_average_flops_cost()[0]/1e6), 6))
            f.write("Validation Layer percentage: {}".format(layer_str))
    if criterion.tb_writer:
        criterion.tb_writer.add_scalar("valid/TASK LOSS-EPOCH", task_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar("valid/Prec@1-EPOCH", top1.avg, epoch)
        criterion.tb_writer.add_scalar('valid/SPARSE LOSS-EPOCH', sparse_loss_record.avg, epoch)
        criterion.tb_writer.add_scalar("valid/MMac-EPOCH", model.compute_average_flops_cost()[0]/1e6, epoch)
        for idx, recorder in enumerate(layer_sparsity_records):
            criterion.tb_writer.add_scalar("valid/LAYER {}-EPOCH".format(idx+1), recorder.avg, epoch)
    return top1.avg, model.compute_average_flops_cost()[0]/1e6


if __name__ == "__main__":
    main()    
