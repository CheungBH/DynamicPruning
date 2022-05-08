#-*-coding:utf-8-*-

cmds = [
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --model_cfg hardware --mask_kernel 1 --model resnet50 --no_attention --input_resolution --dataset-root /media/hkuit155/NewDisk/imagenet --batchsize 64 --budget 0.75 --load exp/imagenet_resnet50_hardware/baseline/checkpoint_best.pth --sparse_strategy static --sparse_weight 0 --valid_range 1 -s exp/add_conv1_mask/s75",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_cfg hardware --mask_kernel 1 --model resnet50 --no_attention --input_resolution --dataset-root /media/hkuit155/NewDisk/imagenet --batchsize 64 --budget 0.50 --load exp/imagenet_resnet50_hardware/baseline/checkpoint_best.pth --sparse_strategy static --sparse_weight 0 --valid_range 1 -s exp/add_conv1_mask/s50",
    "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_cfg hardware --mask_kernel 1 --model resnet50 --no_attention --input_resolution --dataset-root /media/hkuit155/NewDisk/imagenet --batchsize 64 --budget 0.25 --load exp/imagenet_resnet50_hardware/baseline/checkpoint_best.pth --sparse_strategy static --sparse_weight 0 --valid_range 1 -s exp/add_conv1_mask/s75",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
