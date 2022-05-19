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


'''
RESNET101
python test.py --model R110_C10 --load cv/finetuned/R110_C10_gamma_10/ckpt_E_2000_A_0.936_R_1.95E-01_S_16.93_#_469.t7
python test.py --model R110_C10

RESNET34
python test.py --model R32_C10 --load cv/finetuned/R32_C10_gamma_5/ckpt_E_730_A_0.913_R_2.73E-01_S_6.92_#_53.t7
python test.py --model R32_C10 
'''