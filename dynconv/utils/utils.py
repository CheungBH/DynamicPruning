import os.path

import torch
from torchvision import transforms
from math import cos, pi


def set_gumbel(intervals, temps, epoch_ratio, remove_gumbel):
    assert len(intervals) == len(temps), "Please reset your gumbel"
    len_gumbel = len(intervals)
    gumbel_temp = temps[-1]
    for idx in range(len(intervals)):
        if intervals[len_gumbel-idx-1] > epoch_ratio:
            gumbel_temp = temps[len_gumbel-idx-1]
        else:
            break
    gumbel_noise = False if epoch_ratio > remove_gumbel else True
    return gumbel_temp, gumbel_noise


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0.0, lr_max=0.1, warmup_epoch=5):
    if current_epoch < warmup_epoch:
        lr = (lr_max-lr_min) * (current_epoch+1) / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def layer_count(args):
    if args.model == "resnet50":
        layer_cnt = 16
    elif args.model == "resnet101":
        layer_cnt = 36
    elif args.model == "MobileNetV2":
        layer_cnt = 17
    elif args.model == "resnet32":
        layer_cnt = 15
    elif args.model == "MobileNetV2_32x32":
        layer_cnt = 12
    else:
        layer_cnt = 20
    return layer_cnt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, folder, is_best):
    """
    Save the training model
    """
    if len(folder) == 0:
        print('Did not save model since no save directory specified in args!')
        return
        
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, 'checkpoint.pth')
    print(f" => Saving {filename}")
    torch.save(state, filename)

    if is_best:
        filename = os.path.join(folder, 'checkpoint_best.pth')
        print(f" => Saving {filename}")
        torch.save(state, filename)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        assert tensor.shape[0] == 3
        return self.unnormalize(tensor)


def generate_cmd(ls):
    string = ""
    for idx, item in enumerate(ls):
        string += item
        string += " "
    return string[:-1] + "\n"
