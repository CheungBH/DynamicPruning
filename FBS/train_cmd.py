#-*-coding:utf-8-*-

cmds = [
    "python main_imagenet.py -e --group_size 64 --load exp/channel/first/checkpoint.pth --dataset-root /media/hkuit164/Elements1/imagenet --budget 0.5 --workers 0 --model resnet50 --model_cfg hardware --mask_type fc"
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
