#-*-coding:utf-8-*-

cmds = [
    "python main_imagenet.py --budget -1 --model MobileNetV2 -s exp/baseline/mobilenet_full --model_cfg baseline_full --batchsize 128",
    "python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.001 --epochs 160 --scheduler cosine_anneal_warmup --input_resolution --batchsize 72 --channel_unit_type fc_gumbel --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_gumbel_mobile/s50_c50_ave_g32 --group_size 32 --channel_budget 0.5 --channel_stage 6 -1 --lasso_lambda 1 --epochs 120 --no_attention --mask_kernel 1 --load ",
    "python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --pooling_method ave --batchsize 256 --epochs 160 --scheduler cosine_anneal_warmup --channel_unit_type fc_gumbel --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_gumbel/s50_c50_ave_g64_stage23-layerblock --lasso_lambda 1 --group_size 64 --lasso_lambda 1 --channel_budget 0.5 --channel_stage 2 3 --load "

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
