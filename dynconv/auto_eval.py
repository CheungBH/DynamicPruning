import os

cmd_predix = 'CUDA_VISIBLE_DEVICES=0 python main_imagenet.py -e '
ind_dir = "weights/spatial_channel_2048/resnet50_individual"
res_dir = "weights/spatial_channel_2048/resnet50_resolution"
resolutions = [352, 384, 416]

base_cmd = cmd_predix + "--model resnet50 --model_cfg hardware_2048 --load exp/hardware_2048/checkpoint_best.pth --mask_type none --budget -1 --batchsize 72 "

ind_cmds = {
    "s25_relu-C_s25_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    "s25_relu-C_s50_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    "s25_relu-C_s75_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.25 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    "s50_relu-C_s25_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    "s50_relu-C_s50_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    "s50_relu-C_s75_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.5 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
    "s75_relu-C_s25_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    "s75_relu-C_s50_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    "s75_relu-C_s75_ave_g64_stage23-layerblock": "--model_args hardware_2048 --lr 0.01 --input_resolution --batchsize 72 --pooling_method ave --budget 0.75 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
}
res_cmds = {
    "s25_relu-C_s25_ave_g64_stage23-layerblock_resmask": "--model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.25 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.25 --channel_stage 2 3",
    "s50_relu-C_s50_ave_g64_stage23-layerblock_resmask": "--model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.5 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.5 --channel_stage 2 3",
    "s75_relu-C_s75_ave_g64_stage23-layerblock_resmask": "--model_args hardware_2048 --resolution_mask --lr 0.01 --input_resolution --pooling_method ave --batchsize 72 --budget 0.75 --loss_args layer_wise --unlimited_lower --group_size 64 --channel_budget 0.75 --channel_stage 2 3",
}

cmds = [base_cmd]
for model, cmd in ind_cmds.items():
    cmds.append(cmd_predix + cmd + " --load {}/{}/checkpoint_best.pth".format(ind_dir, model))

for model, cmd in res_cmds.items():
    cmds.append(cmd_predix + cmd + " --load {}/{}/checkpoint_best.pth".format(res_dir, model))

for resolution in resolutions:
    os.makedirs("resolution2/{}".format(resolution), exist_ok=True)
    for cmd in cmds:
        print(cmd)
        print("Processing {} with resolution {}".format(cmd.split("/")[-2], resolution))
        if not os.path.exists("resolution2/{0}/{1}.txt".format(resolution, cmd.split("/")[-2])):
            os.system(cmd + " --res {0} > resolution2/{0}/{1}.txt".format(resolution, cmd.split("/")[-2]))
