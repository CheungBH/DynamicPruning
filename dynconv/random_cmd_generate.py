
import os

# pretrain_path = "weights/classification_modified_resnet50/conv_s75/checkpoint_best.pth"
# ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
repeats = 3
pretrain_ratio = {0.75: "weights/classification_modified_resnet50/conv_s75/checkpoint_best.pth",
                  0.25: "weights/classification_modified_resnet50/conv_s25/checkpoint_best.pth"}
execute = True
data_root = "/media/hkuit155/NewDisk/imagenet"
stages = [[0,1,2], [0,1,3],[0,2,3],[1,2,3], [0], [1], [2], [3], [0,1,2,3]]
model_cfg = "hardware"
bs = 32
mask_kernel = 1
target_folder = "random_mask"
os.makedirs(target_folder, exist_ok=True)

for repeat in range(repeats):
    for ratio, pretrain in pretrain_ratio.items():
        for stage in stages:
            stage_str = list(map(lambda x:str(x), stage))
            stage_cmd_str, stage_file_str = " ".join(stage_str), ",".join(stage_str)
            cmd = "python main_imagenet.py --budget {} --model resnet50 --random_mask_stage {} --no_attention " \
                  "--dataset-root {} --load {} -e --batchsize {} --mask_kernel {}".\
                format(ratio, stage_cmd_str, data_root, pretrain, bs, mask_kernel)
            if model_cfg:
                cmd += " --model_cfg {}".format(model_cfg)
            target_file = os.path.join(target_folder, "FLOPs_loss-repeat{}-s{}-target{}.txt".format(repeat, ratio, stage_file_str))
            print(cmd + " > " + target_file)
            # print(cmd)
            if execute:
                if not os.path.exists(target_file):
                    os.system(cmd + " > " + target_file)

