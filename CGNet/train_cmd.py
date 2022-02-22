#-*-coding:utf-8-*-

cmds = [
    # "python cg-cifar10.py -lr 0.08 -wd 3e-4 -gi 0.0 -gt 2 -pt 8 -cg --sparse_bp --use_group -s weights/t2_cg",
    "python cg-cifar10.py -lr 0.08 -wd 3e-4 -gi 0.0 -gt 2 -pt 8 --sparse_bp --use_group -s weights/t2_bl -gpu 1",

    # "python cg-cifar10.py -lr 0.08 -wd 3e-4 -gi 0.0 -gt 3 -pt 16 -cg --sparse_bp --use_group -s weights/t3_cg",
    "python cg-cifar10.py -lr 0.08 -wd 3e-4 -gi 0.0 -gt 3 -pt 16 --sparse_bp --use_group -s weights/t3_bl -gpu 2",
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)