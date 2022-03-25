import argparse
import os.path

import matplotlib.pyplot as plt

import dynconv
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import utils.flopscounter as flopscounter
import utils.logger as logger
import utils.utils as utils
import utils.viz as viz
from torch.backends import cudnn as cudnn
import models

cudnn.benchmark = True

device = 'cuda:0'

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with sparse masks')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=[150,250], nargs='+', type=int, help='learning rate decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--stat_thresh', default=0, type=float, help='Threshold of stat mask')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=350, type=int, help='number of epochs')
    parser.add_argument('--model', type=str, default='resnet32', help='network model name')
    parser.add_argument('--mask_type', type=str, default='conv', help='Type of mask')
    parser.add_argument('--load', type=str, default='', help='load model path')
    parser.add_argument('--mask_kernel', default=3, type=int, help='number of epochs')
    parser.add_argument('--no_attention', action='store_true', help='run without attention')
    parser.add_argument('--individual_forward', action='store_true', help='run without attention')

    # parser.add_argument('--resnet_n', default=5, type=int, help='number of layers per resnet stage (5 for Resnet-32)')
    parser.add_argument('--budget', default=-1, type=float, help='computational budget (between 0 and 1) (-1 for no sparsity)')
    parser.add_argument('-s', '--save_dir', type=str, default='', help='directory to save model')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    parser.add_argument('--workers', default=4, type=int, help='number of dataloader workers')
    args =  parser.parse_args()
    print('Args:', args)


    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ## DATA
    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)

    valset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0, mask_type=args.mask_type, budget=args.budget,
                       momentum=args.momentum, mask_kernel=args.mask_kernel, no_attention=args.no_attention,
                       individual_forward=args.individual_forward).to(device=device)

    file_path = os.path.join(args.save_dir, "log.txt")

    ## CRITERION
    class Loss(nn.Module):
        def __init__(self, budget=1):
            super(Loss, self).__init__()
            self.task_loss = nn.CrossEntropyLoss().to(device=device)
            self.sparsity_loss = dynconv.SparsityCriterion(args.budget, args.epochs) \
                if args.budget >= 0 and "stat" not in args.mask_type else None

        def forward(self, output, target, meta):
            task_loss, sparse_loss = self.task_loss(output, target), torch.zeros(1).cuda()
            logger.add('loss_task', task_loss.item())
            if self.sparsity_loss is not None:
                sparse_loss = 10*self.sparsity_loss(meta)
            return task_loss, sparse_loss
    
    criterion = Loss(args.budget)

    ## OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0

    if not args.evaluate and len(args.save_dir) > 0:
        if not os.path.exists(os.path.join(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir))

    if args.resume:
        resume_path = args.resume
        if not os.path.isfile(resume_path):
            resume_path = os.path.join(resume_path, 'checkpoint.pth')
        if os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path, map_location=device)
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
        checkpoint_dict = torch.load(args.load, map_location=device)['state_dict']
        model_dict = model.state_dict()
        update_dict = {k: v for k, v in model_dict.items() if k in checkpoint_dict.keys()}
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)

    try:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=args.lr_decay, last_epoch=start_epoch)
    except:
        print('Warning: Could not reload learning rate scheduler')
    start_epoch += 1
            
    ## Count number of params
    print("* Number of trainable parameters:", utils.count_parameters(model))


    ## EVALUATION
    if args.evaluate:
        print(f"########## Evaluation ##########")
        prec1 = validate(args, val_loader, model, criterion, start_epoch)
        return
        
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

    # set gumbel temp
    # disable gumbel noise in finetuning stage
    gumbel_temp = 1.0
    gumbel_noise = False if epoch > 0.8*args.epochs else True

    num_step =  len(train_loader)
    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        output, meta = model(input, meta)
        t_loss, s_loss = criterion(output, target, meta)
        prec1 = utils.accuracy(output.data, target)[0]
        top1.update(prec1.item(), input.size(0))
        task_loss_record.update(t_loss.item(), input.size(0))
        sparse_loss_record.update(s_loss.item(), input.size(0))
        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss = s_loss + t_loss if s_loss else t_loss
        loss.backward()
        optimizer.step()

    if file_path:
        with open(file_path, "a+") as f:
            logger.tick(f)
            f.write("Train: Epoch {}, Prec@1 {}, task loss {}, sparse loss {}\n".format
                    (epoch, round(top1.avg, 4), round(task_loss_record.avg, 4), round(sparse_loss_record.avg, 4)))

def validate(args, val_loader, model, criterion, epoch, file_path=None):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()
    task_loss_record = utils.AverageMeter()
    sparse_loss_record = utils.AverageMeter()

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
            output, meta = model(input, meta)
            output = output.float()
            t_loss, s_loss = criterion(output, target, meta)
            task_loss_record.update(t_loss.item(), input.size(0))
            sparse_loss_record.update(s_loss.item(), input.size(0))

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))

            if args.plot_ponder:
                viz.plot_image(input)
                viz.plot_ponder_cost(meta['masks'])
                viz.plot_masks(meta['masks'])
                plt.show()

    print(f'* Epoch {epoch} - Prec@1 {top1.avg:.3f}')
    print(f'* average FLOPS (multiply-accumulates, MACs) per image:  {model.compute_average_flops_cost()[0]/1e6:.6f} MMac')
    model.stop_flops_count()
    if file_path:
        with open(file_path, "a+") as f:
            f.write("Validation: Epoch {}, Prec@1 {}, task loss {}, sparse loss {}, ave FLOPS per image: {} MMac\n".
                    format(epoch, round(top1.avg, 4), round(task_loss_record.avg, 4), round(sparse_loss_record.avg, 4),
                           round(model.compute_average_flops_cost()[0] / 1e6), 6))

    return top1.avg, model.compute_average_flops_cost()[0] / 1e6


if __name__ == "__main__":
    main()    
