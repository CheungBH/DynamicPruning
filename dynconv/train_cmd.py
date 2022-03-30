#-*-coding:utf-8-*-

cmds = [

    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --budget -1 -s exp/imagenet_1x1/baseline --no_attention --mask_kernel 1 --batchsize 320 --dataset-root /media/ssd0/imagenet/ --model resnet50",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_baseline --budget -1",
    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/baseline_k1 --mask_kernel 1 --budget -1 --workers 8",
    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/baseline_NoAtt --mask_kernel 3 --no_attention --budget -1 --workers 8",
    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/baseline_k1_NoAtt --mask_kernel 1 --no_attention --budget -1 --workers 8",

    "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/k1_s100 --mask_kernel 1 --budget 1 --load exp/diff_mask_mobile/baseline_k1/checkpoint_best.pth",
    "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/NoAtt_s100 --mask_kernel 3 --no_attention --budget 1 --load exp/diff_mask_mobile/baseline_NoAtt/checkpoint_best.pth",
    "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/k1_NoAtt_s100 --mask_kernel 1 --no_attention --budget 1 --load exp/diff_mask_mobile/baseline_k1_NoAtt/checkpoint_best.pth",

    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/baseline_k1 --mask_kernel 1 --budget -1",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/baseline_NoAtt --mask_kernel 3 --no_attention --budget -1",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/baseline_k1_NoAtt --mask_kernel 1 --no_attention --budget -1"
    #
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/k1_s50 --budget 0.5 --load exp/diff_mask/baseline_k1/checkpoint_best.pth --mask_kernel 1",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/NoAtt_s50 --budget 0.5 --load exp/diff_mask/baseline_NoAtt/checkpoint_best.pth --mask_kernel 3 --no_attention",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/k1_NoAtt_s50 --budget 0.5 --load exp/diff_mask/baseline_k1_NoAtt/checkpoint_best.pth --mask_kernel 1 --no_attentio"

    # "python main_cifar.py --model MobileNetV2_32x32 --individual_forward -s exp/diff_mask_mobile/s50_stat_mom_indfor --mask_type stat_mom --budget 0.5 --load exp/diff_mask_mobile/baseline/checkpoint_best.pth",
    # "python main_cifar.py --model MobileNetV2_32x32 --individual_forward -s exp/diff_mask_mobile/s75_stat_mom_indfor --mask_type stat_mom --budget 0.75 --load exp/diff_mask_mobile/baseline/checkpoint_best.pth",
    # "python main_cifar.py --model MobileNetV2_32x32 --individual_forward -s exp/diff_mask_mobile/s25_stat_mom_indfor --mask_type stat_mom --budget 0.25 --load exp/diff_mask_mobile/baseline/checkpoint_best.pth",

    #"python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s75_stat_mom --mask_type stat_mom --budget 0.75 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    #"python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s25_stat_mom --mask_type stat_mom --budget 0.25 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",

    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/s50_stat_mom --mask_type stat_mom --budget 0.5 --load  exp/diff_mask_mobile/baseline/checkpoint_best.pth",
    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/s75_stat_mom --mask_type stat_mom --budget 0.75 --load exp/diff_mask_mobile/baseline/checkpoint_best.pth",
    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/s25_stat_mom --mask_type stat_mom --budget 0.25 --load exp/diff_mask_mobile/baseline/checkpoint_best.pth",

    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/s50_conv --budget 0.5 --load  exp/diff_mask_mobile/baseline/checkpoint_best.pth",
    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/s75_conv --budget 0.75 --load exp/diff_mask_mobile/baseline/checkpoint_best.pth",
    # "python main_cifar.py --model MobileNetV2_32x32 -s exp/diff_mask_mobile/s25_conv --budget 0.25 --load exp/diff_mask_mobile/baseline/checkpoint_best.pth",

    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s75_stat --mask_type stat --budget 0.75 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s25_stat --mask_type stat --budget 0.25 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s100_stat --mask_type stat --budget 1 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s100_conv --budget 1 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s50_conv --budget 0.5 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",
    # "python main_cifar.py --model resnet32_BN -s exp/diff_mask/r32BN_s25_conv --budget 0.25 --load exp/diff_mask/r32BN_baseline/checkpoint_best.pth",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
