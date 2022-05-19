#-*-coding:utf-8-*-
import os
import cv2
from pycocotools.coco import COCO
import random
import numpy as np
from collections import defaultdict

img_folder = "/media/hkuit155/NewDisk/imagenet/val"
fm_folder = "/media/hkuit155/NewDisk/feat"


def padded(image, num, h, w):
    if num == 0:
        return image
    tmp = np.full((3, h, w*num), 0)
    return np.concatenate((image, tmp), axis=1)


def draw_fm(fm_ls, name):
    WIDTH = 4
    h, w, _ = fm_ls[0].shape
    image = np.concatenate(fm_ls[:WIDTH], axis=1)
    column = len(fm_ls) // WIDTH
    for c in range(column-1):
        tmp_img = np.concatenate(fm_ls[WIDTH*c: WIDTH*(c+1)], axis=1)
        if c+2 == column:
            tmp_img = padded(tmp_img, len(fm_ls)%WIDTH, h, w)
        image = np.concatenate((image, tmp_img), axis=0)
    cv2.imshow(name, cv2.resize(image, (w, h)))


img_names = [name for name in os.listdir(fm_folder)]


for i, img_name in enumerate(img_names):
    mask_dir = os.path.join(fm_folder, img_name)
    image_path = os.path.join(img_folder, img_name)
    raw_img = cv2.imread(image_path)
    cv2.imshow("raw_image", raw_img)

    fm_dict_before, fm_dict_after = [], []
    for idx, item in enumerate(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, item)
        if "before" in mask_path:
            fm_dict_before.append(cv2.imread(mask_path))
        elif "after" in mask_path:
            fm_dict_after.append(cv2.imread(mask_path))
    draw_fm(fm_dict_before, "before")
    draw_fm(fm_dict_after, "after")

    cv2.waitKey(0)

