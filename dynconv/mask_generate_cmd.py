import os

dataset_root = "/media/hkuit155/NewDisk/imagenet"
checkpoint_folder = "weights/classification_mask_strategy"
conv_kernel = 1
mask_folder = "/media/hkuit155/NewDisk/research/diff_strategy_mask"
execute = True

for folder_name in os.listdir(checkpoint_folder):
    if "s50" in folder_name or "narrower_s25" in folder_name:
        continue
    checkpoint_path = os.path.join(checkpoint_folder, folder_name, "checkpoint_best.pth")
    mask_path = os.path.join(mask_folder, folder_name)
    # if os.path.exists(mask_path):
    #     continue
    os.makedirs(mask_path, exist_ok=True)
    cmd = "python main_imagenet.py --model_cfg hardware --mask_kernel {} --model resnet50 --no_attention --dataset-root {} --batchsize 64 --load {} -e --plot_save_dir {} --plot_ponder".format(conv_kernel, dataset_root, checkpoint_path, mask_path)
    if "s75" in folder_name:
        cmd += " --budget 0.75"
    elif "s50" in folder_name:
        cmd += " --budget 0.5"
    elif "s25" in folder_name:
        cmd += " --budget 0.25"

    if "resmask" in folder_name:
        cmd += " --resolution_mask"
    if "stat_" in folder_name:
        cmd += " --mask_type stat_mom"
    print(cmd)
    if execute:
        os.system(cmd)
