#-*-coding:utf-8-*-

cmds = [

    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --model_cfg hardware -s exp/diff_strategy/higher --model resnet50 --budget 0.5 --dataset-root /home/user/Documents/imagenet --batchsize 96 --mask_kernel 1 --no_attention --valid_range 1 --sparse_strategy higher --sparse_weight 0 --load exp/resnet50_hardware_imagenet/baseline/checkpoint_best.pth",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_cfg hardware -s exp/diff_strategy/lower --model resnet50 --budget 0.5 --dataset-root /home/user/Documents/imagenet --batchsize 96 --mask_kernel 1 --no_attention --valid_range 1 --sparse_strategy lower --sparse_weight 0 --load exp/resnet50_hardware_imagenet/baseline/checkpoint_best.pth",
    "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_cfg hardware -s exp/diff_strategy/static --model resnet50 --budget 0.5 --dataset-root /home/user/Documents/imagenet --batchsize 96 --mask_kernel 1 --no_attention --valid_range 1 --sparse_strategy static --sparse_weight 0 --load exp/resnet50_hardware_imagenet/baseline/checkpoint_best.pth",
    "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_cfg hardware -s exp/diff_strategy/static_range10 --model resnet50 --budget 0.5 --dataset-root /home/user/Documents/imagenet --batchsize 96 --mask_kernel 1 --no_attention --valid_range 1 --sparse_strategy static_range --static_range 0.1 --sparse_weight 0 --load exp/resnet50_hardware_imagenet/baseline/checkpoint_best.pth",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
