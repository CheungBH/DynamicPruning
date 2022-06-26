import h5py
import dataloader.imagenet
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File("tb/channels_3_2.h5", "r")
transform_val = transforms.Compose([
])
valset = dataloader.imagenet.IN1K(root="/media/hkuit164/Elements1/imagenet", split='val', transform=transform_val)
labels, img_names = valset.labels, valset.imgs

class_channels = defaultdict(list)
for label, img_name in zip(labels, img_names):
    class_channels[label].append(f[img_name.split("/")[-1]].value)
print(class_channels)

class_percents = {}
for k, v in class_channels.items():
    class_percents[k] = np.array(v).sum(axis=0)/len(v)
print(class_percents)


def obtain_short_classes(length=10):
    chosen_class = []
    while len(chosen_class) < 10:
        remaining_num = 10 - len(chosen_class)
        chosen_class += random.sample(range(0, 1000), remaining_num)
        classes_word = [valset.class_to_word[valset.idx_to_class[cls]] for cls in chosen_class]
        bad_classes_idx = [idx for idx, class_word in enumerate(classes_word) if
                           len(class_word) > length or "," in class_word]
        for bad_idx in sorted(bad_classes_idx, reverse=True):
            chosen_class.remove(chosen_class[bad_idx])
    return chosen_class

import random
chosen_class = obtain_short_classes()
classes_word = [valset.class_to_word[valset.idx_to_class[cls]] for cls in chosen_class]
percent_array = np.array([channel for idx, channel in class_percents.items() if idx in chosen_class])
percent_array = percent_array.repeat(3, axis=1).repeat(2, axis=0)

x_tick = [3*x+1 for x in range(16)]
y_tick = [2*x+0.5 for x in range(10)]
fig = plt.figure()
plt.xticks(x_tick, [str(idx) for idx in range(16)], fontweight='bold', fontsize=24)
plt.xlabel("channel groups", fontweight='bold', fontsize=24)
plt.yticks(y_tick, classes_word, fontweight='bold', fontsize=24)
# plt.ylabel("classes")
plt.imshow(percent_array, cmap="binary")
cb = plt.colorbar(ticks=range(2), orientation="vertical")
cb.ax.set_yticklabels(labels=["0.0", "1.0"], weight='bold', fontsize=24)
cb.set_label(label="Selection Ratio", weight='bold', fontsize=24)
plt.draw()
plt.pause(0)

