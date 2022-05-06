
import os

pretrain_path = "weight/resnet50/resnet50-19c8e357.pth"
ratios = [0.4, 0.6, 0.1, 0.25, 0.5, 0.75, 0.9]
mask_types = ["sum", "abs_sum"]
execute = True
data_root = "/media/hkuit164/Elements1/imagenet"
stages = [[0], [1], [2], [3], [0,1,2], [0,1,3], [0,2,3], [1,2,3], [0,1,2,3]]
bs = 32
target_folder = "feature_analysis"
os.makedirs(target_folder, exist_ok=True)

for mask_type in mask_types:
    for ratio in ratios:
        for stage in stages:
            stage_str = list(map(lambda x:str(x), stage))
            stage_cmd_str, stage_file_str = " ".join(stage_str), ",".join(stage_str)
            cmd = "CUDA_VISBLE_DEVICES=0 python main_imagenet.py --budget -1 --model resnet50 --mask_type {} --mask_thresh {} " \
                  "--target_stage {} --dataset-root {} --load {} -e --batchsize {}".\
                format(mask_type, ratio, stage_cmd_str, data_root, pretrain_path, bs)
            print(cmd)
            if execute:
                target_file = os.path.join(target_folder, "{}-{}-target{}.txt".format(mask_type, ratio, stage_file_str))
                if not os.path.exists(target_file):
                    os.system(cmd + " > " + target_file)
