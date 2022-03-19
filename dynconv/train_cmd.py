#-*-coding:utf-8-*-

cmds = [
    "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_baseline --budget -1",
    "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s100_stat --mask_type stat --budget 1 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s50_stat --mask_type stat --budget 0.5 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s75_stat --mask_type stat --budget 0.75 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",

    "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s100_conv --budget 1 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s50_conv --budget 0.5 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s75_conv --budget 0.75 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
