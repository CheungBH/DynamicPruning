#-*-coding:utf-8-*-

cmds = [
    "python main_imagenet.py -e --model MobileNetV2 --model_cfg baseline_full --lr 0.000001 --epochs 12 --lr_decay 8 --layer_weight 1 --load weights/spatial_channel_mobile/s75_relu-C_s75_ave_g64_stage23-layerblock_continue/checkpoint_best.pth --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower --group_size 32 --channel_budget 0.75 --channel_stage 6 -1",
    "python main_imagenet.py -e --model MobileNetV2 --model_cfg baseline_full --lr 1e-6 --load weights/spatial_channel_mobile/s50_relu-C_s50_ave_g64_stage23-layerblock/checkpoint_best.pth --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower --group_size 32 --channel_budget 0.5 --channel_stage 6 -1  --layer_weight 1"
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
