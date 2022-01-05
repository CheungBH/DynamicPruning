#-*-coding:utf-8-*-

cmds = [
    "python cg-cifar10.py -gpu 1 -lr 0.08 -wd 3e-4 -gi 0.0 -gt 0.6 -pt 2 -cg --use_postact --sparse_bp -shuffle --use_group -s weights/CG_60_shuf",
    "python cg-cifar10.py -gpu 1 -lr 0.08 -wd 3e-4 -gi 0.0 -gt 0.6 -pt 2 -cg --use_postact --sparse_bp --use_group -s weights/CG_60",
    "python cg-cifar10.py -gpu 0 -gi 0.0 -gt 1.0 -pt 4 --alpha 2.0 -cg --sparse_bp --use_group -s weights/CGpost",
    "python cg-cifar10.py -gpu 0 -gi 0.0 -gt 1.0 -pt 4 --alpha 2.0 -cg --sparse_bp --use_group -s weights/CGpost_shuf -shuffle",
    "python cg-cifar10.py -gpu 1 -lr 0.08 -wd 3e-4 -gi 0.0 -gt 0.9 -pt 2 -cg --use_postact --sparse_bp --use_group -s weights/CG_90",
    "python cg-cifar10.py -gpu 1 -lr 0.08 -wd 3e-4 -gi 0.0 -gt 0.8 -pt 2 -cg --use_postact --sparse_bp --use_group -s weights/CG_80",
    "python cg-cifar10.py -gpu 1 -lr 0.08 -wd 3e-4 -gi 0.0 -gt 0.7 -pt 2 -cg --use_postact --sparse_bp --use_group -s weights/CG_70",

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    # print("Processing cmd {}".format(idx))
    os.system(cmd)