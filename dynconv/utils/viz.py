import math
import matplotlib.pyplot as plt
import utils.utils as utils
import dynconv
import torch
import torch.nn as nn
import os
from torchvision import utils as vutils

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
unnormalize = utils.UnNormalize(mean, std)


def save_feat(feats, dir, name, prefix, stages=(3,4,6,3)):
    def get_stage(curr_idx):
        s = 0
        for i, stage in enumerate(stages):
            s += stage
            if curr_idx < s:
                return i, curr_idx - (s - stage)

    os.makedirs(os.path.join(dir, name), exist_ok=True)
    for idx, feat in enumerate(feats):
        stage, layer = get_stage(idx)
        vutils.save_image(
            nn.functional.upsample_nearest(torch.sum(feat, 1).unsqueeze(dim=1), scale_factor=8*pow(2, (stage-1))),
            os.path.join(dir, name, "stage_{}_{}_{}_relu.jpg".format(stage, layer, prefix)),
            normalize=True)


def plot_image(input):
    ''' shows the first image of a 4D pytorch batch '''
    assert input.dim() == 4
    plt.figure('Image')
    im = unnormalize(input[0]).cpu().numpy().transpose(1,2,0)
    plt.imshow(im)


def plot_ponder_cost(masks):
    ''' plots ponder cost
    argument masks is a list with masks as returned by the network '''
    assert isinstance(masks, list)
    plt.figure('Ponder Cost')
    ponder_cost = dynconv.ponder_cost_map(masks)
    plt.imshow(ponder_cost, vmin=0, vmax=len(masks))
    plt.colorbar()

def plot_masks(masks, save_path=""):
    ''' plots individual masks as subplots 
    argument masks is a list with masks as returned by the network '''
    nb_mask = len(masks)
    WIDTH = 2
    HEIGHT = math.ceil(nb_mask / WIDTH)
    f, axarr = plt.subplots(HEIGHT, WIDTH)

    for i, mask in enumerate(masks):
        x = i % WIDTH
        y = i // WIDTH

        m = mask['std'].hard[0].cpu().numpy().squeeze(0)

        assert m.ndim == 2
        axarr[y,x].imshow(m, vmin=0, vmax=1)
        axarr[y,x].axis('off')
    
    for j in range(i+1, WIDTH*HEIGHT):
        x = j % WIDTH
        y = j // WIDTH
        f.delaxes(axarr[y,x])

    plt.savefig(save_path)

def showKey(show=False):
    ''' 
    shows a plot, closable by pressing a key 
    '''
    if show:
        plt.draw()
        plt.pause(1)
        input("<Hit Enter To Close>")
    plt.clf()
    plt.cla()
    plt.close('all')