#-*-coding:utf-8-*-

cmds = [
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s75_lr0.01_relu_layerloss",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --batchsize 80 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_baseline",

    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_relu_layerloss",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --loss_args flops -s exp/add_conv1_mask/s50_lr0.01_relu_flops",
    # "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_relu_layerloss_front --layer_loss_method front_mask",
    # "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_relu_layerloss_later --layer_loss_method later_mask --dataset-root ~/imagenet",

    # "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act none --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_none_layerloss",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act none --loss_args flops -s exp/add_conv1_mask/s50_lr0.01_none_flops",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act none --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_none_layerloss_front --layer_loss_method front_mask",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act none --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_none_layerloss_later --layer_loss_method later_mask",

    # "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_lrelu_layerloss",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args flops -s exp/add_conv1_mask/s50_lr0.01_lrelu_flops",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_lrelu_layerloss_front --layer_loss_method front_mask",
    # "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_lrelu_layerloss_later --layer_loss_method later_mask",

    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_relu_layerloss_resmask",
    "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --loss_args flops -s exp/add_conv1_mask/s50_lr0.01_relu_flops_resmask",
    "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_relu_layerloss_later_resmask --layer_loss_method later_mask",

    "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_lrelu_layerloss_resmask",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args flops -s exp/add_conv1_mask/s50_lr0.01_lrelu_flops_resmask",
    "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_lrelu_layerloss_later_resmask --layer_loss_method later_mask",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
