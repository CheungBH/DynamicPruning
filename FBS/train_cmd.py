#-*-coding:utf-8-*-

cmds = [
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --group_size 64 --budget 0.25 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root /media/hkuit164/Elements1/imagenet -s exp/channel/s25_ave_stage23 --load ",
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --group_size 64 --budget 0.25 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root /media/hkuit164/Elements1/imagenet -s exp/channel/s25_max_stage23 --load ",
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --group_size 64 --budget 0.5 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method max --dataset-root /media/hkuit164/Elements1/imagenet -s exp/channel/s50_ave_stage23 --load ",
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --group_size 64 --budget 0.5 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root /media/hkuit164/Elements1/imagenet -s exp/channel/s50_max_stage23 --load ",
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
