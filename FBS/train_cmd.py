#-*-coding:utf-8-*-

cmds = [
    # "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --group_size 64 --budget 0.25 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root /home/user/Documents/imagenet -s exp/channel/s25_ave_stage23 --load ../dynconv/exp/imagenet_1x1/baseline/checkpoint_best.pth",
    # "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --group_size 64 --budget 0.25 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method max --dataset-root /home/user/Documents/imagenet -s exp/channel/s25_max_stage23 --load ../dynconv/exp/resnet50_hardware_imagenet/baseline/checkpoint_best.pth",
    # "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --group_size 64 --budget 0.5 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method max --dataset-root /home/user/Documents/imagenet -s exp/channel/s50_max_stage23 --load ../dynconv/exp/resnet50_hardware_imagenet/baseline/checkpoint_best.pth",

    # "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --batchsize 80 --group_size 64 --budget 0.5 --model resnet50 --model_cfg hardware --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root  ~/Documents/imagenet/ -s exp/channel/s50_ave_stage23 --load ../dynconv/exp/imagenet_1x1/baseline/checkpoint_best.pth",

    "CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --batchsize 80 --group_size 64 --budget 0.5 --model resnet50 --model_cfg hardware_2048 --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root  ~/Documents/imagenet/ -s exp/channel/s50_ave_stage23_g64 --load ../dynconv/weights/hardware_2048/checkpoint_best.pth",
    "CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --batchsize 80 --group_size 32 --budget 0.5 --model resnet50 --model_cfg hardware_2048 --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root  ~/Documents/imagenet/ -s exp/channel/s50_ave_stage23_g32 --load ../dynconv/weights/hardware_2048/checkpoint_best.pth",
    "CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --batchsize 80 --group_size 16 --budget 0.5 --model resnet50 --model_cfg hardware_2048 --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root  ~/Documents/imagenet/ -s exp/channel/s50_ave_stage23_g16 --load ../dynconv/weights/hardware_2048/checkpoint_best.pth",
    "CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --batchsize 80 --group_size 8 --budget 0.5 --model resnet50 --model_cfg hardware_2048 --mask_type fc --target_stage 2 3 --pooling_method ave --dataset-root  ~/Documents/imagenet/ -s exp/channel/s50_ave_stage23_g8 --load ../dynconv/weights/hardware_2048/checkpoint_best.pth",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)
