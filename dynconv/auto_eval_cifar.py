import os

cmd_predix = 'CUDA_VISIBLE_DEVICES=0 python main_cifar.py -e '
ind_dir = "exp/cifar_spatial_channel"
exps = os.listdir(ind_dir)


def get_sparsity(name):
    if "c25" in name:
        return 0.25
    elif "c50" in name:
        return 0.5
    elif "c75" in name:
        return 0.75
    else:
        raise ValueError

cmds = []
for exp in exps:
    if "resmask" in exp:
        cmds.append("python main_cifar.py -e --resolution_mask --input_resolution --budget 0.50 --load {} --group_size"
                    " 8 --channel_budget {} --channel_stage 1 2 --mask_type conv --mask_kernel 1 --no_attention "
            .format(os.path.join(ind_dir, exp, "checkpoint_best.pth"), get_sparsity(exp)))
    elif "baseline" in exp:
        cmds.append("python main_cifar.py -e --mask_type none --load {}".format(
            os.path.join(ind_dir, exp, "checkpoint_best.pth")))
    else:
        cmds.append("python main_cifar.py -e --input_resolution --budget 0.50 --load {} --group_size 8 --channel_budget"
                    " {} --channel_stage 1 2 --mask_type conv --mask_kernel 1 --no_attention".
            format(os.path.join(ind_dir, exp, "checkpoint_best.pth"), get_sparsity(exp)))

# print(cmds)
os.makedirs("cifar_result", exist_ok=True)
for cmd in cmds:
    print(cmd)
    os.system(cmd + " > cifar_result/{}.txt".format( cmd.split("/")[-2]))
