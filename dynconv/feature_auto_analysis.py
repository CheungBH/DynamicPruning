import os

file_folder = "random_mask"
result_file = "random_mask.txt"

def get_acc(file):
    with open(file, "r") as f:
        lines = f.readlines()
        if "Evaluation" in lines[-1]:
            print(file)
            return "-1"
        acc = lines[-3].split(" ")[-1][:-1] if "Prec@1" in lines[-3] else lines[-2].split(" ")[-1][:-1]
    return acc


acc_dict = {file_name.split(".txt")[0]: get_acc(os.path.join(file_folder, file_name))
            for file_name in os.listdir(file_folder)}
print(acc_dict)

percents = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
targets = ["0,1,2", "0,1,3", "0,2,3", "0", "1,2,3", "1", "2", "0,1,2,3", "1,2,3", "2,3", "3"]

percents_ls = []
for percent in percents:
    percent_ls = []
    for target in targets:
        for k, v in acc_dict.items():
            if "target" + target in k and str(percent) in k:
                percent_ls.append(v)
    percents_ls.append(percent_ls)

print(percents_ls)


def change_char(char):
    return char.replace(",", "-")


with open(result_file, "w") as f:
    f.write(",".join(["Name"] + ["target"+change_char(item) for item in targets]) + "\n")
    # f.write(",".join(list(map(lambda x: x.replace("," "-"), ["Name"] + targets)))+"\n")
    for idx, perc in enumerate(percents_ls):
        f.write(str(percents[idx])+","+",".join(perc)+"\n")

