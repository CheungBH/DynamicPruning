#-*-coding:utf-8-*-

cmds = [
    # "python main_cifar.py --model resnet32 --save_dir exp/cifar10_no_dilation/pretrain --budget -1",
    # "python main_cifar.py --model resnet32 --save_dir exp/cifar10_no_dilation/s90 --budget 0.9 "
    # "--pretrain exp/cifar10_no_dilation/pretrain/checkpoint.pth",
    "python main_imagenet.py --model resnet50 --save_dir exp/imagenet_no_dilation/s100 --budget 1 "
    "--dataset-root /home/user/Documents/imagenet",
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
