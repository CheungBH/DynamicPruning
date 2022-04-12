#-*-coding:utf-8-*-

cmds = [

    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py -s exp/diff_strategy/higher --model resnet50 --budget 0.5 --dataset-root /media/hkuit155/NewDisk/imagenet --batchsize 64 --mask_kernel 1 --no_attention --valid_range 1 --sparse_strategy higher --sparse_weight 0 --load ",
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py -s exp/diff_strategy/higher --model resnet50 --budget 0.5 --dataset-root /media/hkuit155/NewDisk/imagenet --batchsize 64 --mask_kernel 1 --no_attention --valid_range 1 --sparse_strategy lower --sparse_weight 0 --load ",
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py -s exp/diff_strategy/higher --model resnet50 --budget 0.5 --dataset-root /media/hkuit155/NewDisk/imagenet --batchsize 64 --mask_kernel 1 --no_attention --valid_range 1 --sparse_strategy static --sparse_weight 0 --load ",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
