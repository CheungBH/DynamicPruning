#-*-coding:utf-8-*-

cmds = [
    "python ManiDP/main.py --ngpu=1 --arch=dyresnet20 --dataset=cifar10 --target_remain_rate=0.6 --lambda_lasso 5e-3 "
    "--lambda_graph 1.0 --pretrain_path weights/cifar10_pretrain_dyresnet20/model_best.pth.tar "
    "--data_path=/media/hkuit155/Windows1/research/ViT-cifar10-pruning/data --save_path weights/cifar10_dyresnet20_60",

    "python ManiDP/main.py --ngpu=1 --arch=dyresnet20 --dataset=cifar10 --target_remain_rate=0.9 --lambda_lasso 5e-3 "
    "--lambda_graph 1.0 --pretrain_path weights/cifar10_pretrain_dyresnet20/model_best.pth.tar "
    "--data_path=/media/hkuit155/Windows1/research/ViT-cifar10-pruning/data --save_path weights/cifar10_dyresnet20_90",

    "python ManiDP/main.py --ngpu=1 --arch=dyresnet20 --dataset=cifar10 --target_remain_rate=0.8 --lambda_lasso 5e-3 "
    "--lambda_graph 1.0 --pretrain_path weights/cifar10_pretrain_dyresnet20/model_best.pth.tar "
    "--data_path=/media/hkuit155/Windows1/research/ViT-cifar10-pruning/data --save_path weights/cifar10_dyresnet20_80",

    "python ManiDP/main.py --ngpu=1 --arch=dyresnet20 --dataset=cifar10 --target_remain_rate=0.7 --lambda_lasso 5e-3 "
    "--lambda_graph 1.0 --pretrain_path weights/cifar10_pretrain_dyresnet20/model_best.pth.tar "
    "--data_path=/media/hkuit155/Windows1/research/ViT-cifar10-pruning/data --save_path weights/cifar10_dyresnet20_70",

    "python ManiDP/main.py --ngpu=1 --arch=dyresnet20 --dataset=cifar10 --target_remain_rate=0.5 --lambda_lasso 5e-3 "
    "--lambda_graph 1.0 --pretrain_path weights/cifar10_pretrain_dyresnet20/model_best.pth.tar "
    "--data_path=/media/hkuit155/Windows1/research/ViT-cifar10-pruning/data --save_path weights/cifar10_dyresnet20_50",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)