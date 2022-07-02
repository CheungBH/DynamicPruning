#-*-coding:utf-8-*-

cmds = [
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --input_resolution --batchsize 80 --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s75_lr0.01_relu_layerloss",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --lr 0.01 --batchsize 80 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_baseline",
    #
    # "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_lrelu_layerloss_resmask",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args flops -s exp/add_conv1_mask/s50_lr0.01_lrelu_flops_resmask",
    # "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_resnet50 --resolution_mask --lr 0.01 --input_resolution --batchsize 80 --budget 0.5 --conv1_act leaky_relu --loss_args layer_wise --unlimited_lower -s exp/add_conv1_mask/s50_lr0.01_lrelu_layerloss_later_resmask --layer_loss_method later_mask",

    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --budget -1 --model resnet50 --dataset-root /media/hkuit164/Elements1/imagenet --mask_kernel 1 --no_attention --batchsize 32 --input_resolution --mask_type none --model_cfg hardware_static_s50_bl -s exp/baseline/channel_static_s50",
    # "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --budget -1 --model resnet50 --dataset-root /media/hkuit164/Elements1/imagenet --mask_kernel 1 --no_attention --batchsize 32 --input_resolution --mask_type none --model_cfg hardware_static_s75_bl -s exp/baseline/channel_static_s75",


    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.25 --loss_args layer_wise -s exp/spatial_channel_2048/s25_relu_-C_s25_max_stage23-layerblock --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.25 --loss_args layer_wise -s exp/spatial_channel_2048/s25_relu-C_s50_max_stage23-layerblock --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.25 --loss_args layer_wise -s exp/spatial_channel_2048/s25_relu-C_s75_max_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.5 --loss_args layer_wise -s exp/spatial_channel_2048/s50_relu-C_s25_max_stage23-layerblock --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.5 --loss_args layer_wise -s exp/spatial_channel_2048/s50_relu-C_s75_max_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.75 --loss_args layer_wise -s exp/spatial_channel_2048/s75_relu-C_s25_max_stage23-layerblock --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.75 --loss_args layer_wise -s exp/spatial_channel_2048/s75_relu-C_s50_max_stage23-layerblock --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --budget 0.75 --loss_args layer_wise -s exp/spatial_channel_2048/s75_relu-C_s75_max_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3",

    # "python main_imagenet.py --budget -1 --model MobileNetV2 -s exp/baseline/mobilenet_full --model_cfg baseline_full"

    # "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --batchsize 72 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_resmask/s50_layerloss_later_resmask --layer_loss_method later_mask",
    # "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --batchsize 72 --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_resmask/s50_layerloss_later_resmask --layer_loss_method later_mask",
    # "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --batchsize 72 --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_resmask/s50_layerloss_later_resmask --layer_loss_method later_mask",

    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s50_relu-C_s50_ave_g32_stage23-layerblock --group_size 32 --channel_budget 0.5 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s25_relu-C_s25_ave_g32_stage23-layerblock --group_size 32 --channel_budget 0.25 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s75_relu-C_s75_ave_g32_stage23-layerblock --group_size 32 --channel_budget 0.75 --channel_stage 2 3",
    #
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s50_relu-C_s50_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s25_relu-C_s25_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s75_relu-C_s75_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    #
    # "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s50_relu-C_s50_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s75_relu-C_s75_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    # "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s25_relu-C_s25_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.25 --channel_stage 2 3",

    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py -s exp/cifar_spatial_channel/baseline --mask_type none --auto_resume",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.50 --unlimited_lower -s exp/cifar_ave_spatial_channel/s50_c75_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.75 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.50 --unlimited_lower -s exp/cifar_ave_spatial_channel/s50_c50_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.50 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.50 --unlimited_lower -s exp/cifar_ave_spatial_channel/s50_c25_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.25 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.75 --unlimited_lower -s exp/cifar_ave_spatial_channel/s75_c75_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.75 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.75 --unlimited_lower -s exp/cifar_ave_spatial_channel/s75_c50_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.50 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.75 --unlimited_lower -s exp/cifar_ave_spatial_channel/s75_c25_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.25 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.25 --unlimited_lower -s exp/cifar_ave_spatial_channel/s25_c75_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.75 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.25 --unlimited_lower -s exp/cifar_ave_spatial_channel/s25_c50_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.50 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --input_resolution --budget 0.25 --unlimited_lower -s exp/cifar_ave_spatial_channel/s25_c25_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.25 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
                                          
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.50 -s exp/cifar_ave_spatial_channel/s50_c75_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.75 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.50 -s exp/cifar_ave_spatial_channel/s50_c50_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.5 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.50 -s exp/cifar_ave_spatial_channel/s50_c25_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.25 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.75 -s exp/cifar_ave_spatial_channel/s75_c75_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.75 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.75 -s exp/cifar_ave_spatial_channel/s75_c50_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.5 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.75 -s exp/cifar_ave_spatial_channel/s75_c25_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.25 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.25 -s exp/cifar_ave_spatial_channel/s25_c75_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.75 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.25 -s exp/cifar_ave_spatial_channel/s25_c50_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.5 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",
    "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --pooling_method ave --resolution_mask --input_resolution  --unlimited_lower --budget 0.25 -s exp/cifar_ave_spatial_channel/s25_c25_resmask_ave --load exp/cifar_spatial_channel/baseline/checkpoint_best.pth --group_size 8 --channel_budget 0.25 --channel_stage 1 2 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention",

    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048/s50_relu-C_s50_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.5 --channel_stage 6 -1 --model MobileNetV2 --mask_kernel 1 --no_attention",
    #
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --channel_stage 0 -1 --model MobileNetV2_32x32 -s exp/cifar_mobile_spatial_channel/baseline --mask_type none",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.25 -s exp/cifar_mobile_spatial_channel/s25_c25_ave --group_size 8 --channel_budget 0.25 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.25 -s exp/cifar_mobile_spatial_channel/s25_c50_ave_retrain --group_size 8 --channel_budget 0.50 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.25 -s exp/cifar_mobile_spatial_channel/s25_c75_ave --group_size 8 --channel_budget 0.75 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.50 -s exp/cifar_mobile_spatial_channel/s50_c25_ave --group_size 8 --channel_budget 0.25 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.50 -s exp/cifar_mobile_spatial_channel/s50_c50_ave --group_size 8 --channel_budget 0.50 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.50 -s exp/cifar_mobile_spatial_channel/s50_c75_ave --group_size 8 --channel_budget 0.75 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.75 -s exp/cifar_mobile_spatial_channel/s75_c25_ave --group_size 8 --channel_budget 0.25 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.75 -s exp/cifar_mobile_spatial_channel/s75_c50_ave --group_size 8 --channel_budget 0.50 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --input_resolution --budget 0.75 -s exp/cifar_mobile_spatial_channel/s75_c75_ave --group_size 8 --channel_budget 0.75 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    #
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.25 -s exp/cifar_mobile_spatial_channel/s25_c25_ave_resmask --group_size 8 --channel_budget 0.25 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.25 -s exp/cifar_mobile_spatial_channel/s25_c50_ave_resmask --group_size 8 --channel_budget 0.50 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.25 -s exp/cifar_mobile_spatial_channel/s25_c75_ave_resmask --group_size 8 --channel_budget 0.75 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.50 -s exp/cifar_mobile_spatial_channel/s50_c25_ave_resmask --group_size 8 --channel_budget 0.25 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.50 -s exp/cifar_mobile_spatial_channel/s50_c50_ave_resmask --group_size 8 --channel_budget 0.50 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.50 -s exp/cifar_mobile_spatial_channel/s50_c75_ave_resmask --group_size 8 --channel_budget 0.75 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.75 -s exp/cifar_mobile_spatial_channel/s75_c25_ave_resmask --group_size 8 --channel_budget 0.25 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.75 -s exp/cifar_mobile_spatial_channel/s75_c50_ave_resmask --group_size 8 --channel_budget 0.50 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    # "CUDA_VISIBLE_DEVICES=0 python main_cifar.py --load exp/cifar_mobile_spatial_channel/baseline/checkpoint.pth --resolution_mask --input_resolution --budget 0.75 -s exp/cifar_mobile_spatial_channel/s75_c75_ave_resmask --group_size 8 --channel_budget 0.75 --channel_stage 4 -1 --mask_type conv --net_weight 0 --valid_range 1 --mask_kernel 1 --no_attention --model MobileNetV2_32x32",
    #
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s50_relu-C_s50_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.5 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s25_relu-C_s25_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.25 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s75_relu-C_s75_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.75 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s50_relu-C_s50_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.5 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s25_relu-C_s25_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.25 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s75_relu-C_s75_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.75 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s50_relu-C_s50_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.5 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s25_relu-C_s25_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.25 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s75_relu-C_s75_ave_g64_stage23-layerblock --group_size 32 --channel_budget 0.75 --channel_stage 6 -1",
    #
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --resolution_mask --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s50_relu-C_s50_ave_g64_stage23-layerblock_resmask --group_size 32 --channel_budget 0.5 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --resolution_mask --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s25_relu-C_s25_ave_g64_stage23-layerblock_resmask --group_size 32 --channel_budget 0.25 --channel_stage 6 -1",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --resolution_mask --model MobileNetV2 --model_cfg baseline_full --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_mobile/s75_relu-C_s75_ave_g64_stage23-layerblock_resmask --group_size 32 --channel_budget 0.75 --channel_stage 6 -1",

    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s75_relu-C_s50_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s75_relu-C_s50_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s50_relu-C_s50_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.50 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s50_relu-C_s50_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s50_relu-C_s75_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.50 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s50_relu-C_s75_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s75_relu-C_s75_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s75_relu-C_s75_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s75_relu-C_s25_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s75_relu-C_s25_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s25_relu-C_s75_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s25_relu-C_s75_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s25_relu-C_s25_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s25_relu-C_s25_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s25_relu-C_s50_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s25_relu-C_s50_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_individual/s50_relu-C_s25_ave_g64_stage23-layerblock/checkpoint_best.pth --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.50 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s50_relu-C_s25_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --epochs 12 --lr_decay 5 9",

    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_resolution/s25_relu-C_s25_ave_g64_stage23-layerblock_resmask/checkpoint_best.pth --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.25 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s25_relu-C_s25_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.25 --channel_stage 2 3 --epochs 10 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_resolution/s50_relu-C_s50_ave_g64_stage23-layerblock_resmask/checkpoint_best.pth --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.50 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s50_relu-C_s50_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.50 --channel_stage 2 3 --epochs 10 --lr_decay 5 9",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --load weights/spatial_channel_2048/resnet50_resolution/s75_relu-C_s75_ave_g64_stage23-layerblock_resmask/checkpoint_best.pth --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix_ft/s75_relu-C_s75_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.75 --channel_stage 2 3 --epochs 10 --lr_decay 5 9",

    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --epochs 120 --scheduler cosine_anneal_warmup --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 66 --pooling_method ave --budget 0.50 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix/s50_relu-C_s50_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.50 --channel_stage 2 3",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --epochs 120 --scheduler cosine_anneal_warmup --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 66 --pooling_method ave --budget 0.50 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix/s50_relu-C_s75_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --epochs 120 --scheduler cosine_anneal_warmup --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 66 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix/s75_relu-C_s75_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --epochs 120 --scheduler cosine_anneal_warmup --model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 66 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix/s75_relu-C_s50_ave_g64_stage23-layerblock --group_size 64 --channel_budget 0.50 --channel_stage 2 3",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --epochs 120 --scheduler cosine_anneal_warmup --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 66 --budget 0.50 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix/s50_relu-C_s50_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.50 --channel_stage 2 3",
    "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --epochs 120 --scheduler cosine_anneal_warmup --model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 66 --budget 0.75 --loss_args layer_wise --unlimited_lower -s exp/spatial_channel_2048_no_mix/s75_relu-C_s75_ave_g64_stage23-layerblock_resmask --group_size 64 --channel_budget 0.75 --channel_stage 2 3",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
