import os
from collections import defaultdict


def calculate_reduction(metric_dict, metric_name):
    baselines = metric_dict[("baseline", )]
    drop_dict = defaultdict(list)
    drop_dict[("baseline", )] = ["-" for _ in range(len(baselines))]
    for ex, metrics in metric_dict.items():
        if ex not in drop_dict:
            for metric,  baseline in zip(metrics, baselines):
                metric, baseline = float(metric), float(baseline)
                if metric_name == "acc":
                    drop_dict[ex].append(str(round((metric - baseline), 4)))
                elif metric_name == "Mac":
                    drop_dict[ex].append(str(round((baseline/metric), 4)-1))
    return drop_dict


def recognize_sparsity(name):
    spatial = name[0:3]
    channel = name[11:14]
    res = "" if "resmask" not in name else "resmask"
    if "hardware" in name:
        return ("baseline", )
    return spatial, channel, res


def get_profile(file):
    with open(file, "r") as f:
        lines = f.readlines()
        if "Evaluation" in lines[-1]:
            print(file)
            return "-1"
        acc = lines[-3].split(" ")[-1][:-1] if "Prec@1" in lines[-3] else lines[-2].split(" ")[-1][:-1]
        flops = lines[-2].split(" ")[-2] if "MACs" in lines[-2] else lines[-1].split(" ")[-2]
    return acc, flops


def tuple2str(tp):
    if isinstance(tp, tuple):
        s = "-".join(tp)
        return "({})".format(s)
    else:
        return tp


src_folder = "resolution2"
acc_dict, MMac_dict, acc_reduction, MMac_reduction = {}, {}, {}, {}
resolutions = [i for i in os.listdir(src_folder) if "txt" not in i]
resolutions.sort()
result_names = ["acc", "MMac", "acc_drop", "MMac_reduction"]
# result_names = ["acc", "MMac"]
result_files = [open(os.path.join(src_folder, "resolution_{}.txt".format(file_name)), "w") for file_name in result_names]
for result_file in result_files:
    result_file.write("resolution," + ",".join(resolutions) + "\n")
exps = os.listdir(os.path.join(src_folder, resolutions[0]))
exps.sort()
for exp in exps:
    acc, mac = [], []
    for resolution in resolutions:
        info = get_profile(os.path.join(src_folder, resolution, exp))
        acc.append(info[0])
        mac.append(info[-1])
    # summary_dict[recognize_sparsity(exp)] = tmp
    acc_dict[recognize_sparsity(exp)] = acc
    MMac_dict[recognize_sparsity(exp)] = mac

acc_drop_dict = calculate_reduction(acc_dict, "acc")
MMac_drop_dict = calculate_reduction(MMac_dict, "Mac")
for file, i_dict in zip(result_files, [acc_dict, MMac_dict, acc_drop_dict, MMac_drop_dict]):
    for k, v in i_dict.items():
        file.write(tuple2str(k))
        # result_file.write(",")
        v_str = [tuple2str(tp) for tp in v]
        file.write("," + ",".join(v_str) + "\n")


# if __name__ == '__main__':
#     file_path = 'feature_analysis/abs_sum-0.1-target0.txt'
#     print(get_profile(file_path))
