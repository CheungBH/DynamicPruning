import os
import cv2

image_path = "/media/hkuit164/Elements1/imagenet/val"
mask_paths = ["plot/s50"]

img_names = [name for name in os.listdir(mask_paths[0])]

for img_name in img_names:
    raw_img = cv2.imread(os.path.join(image_path, img_name))

    cv2.imshow("raw", raw_img)
    for mask_path in mask_paths:
        mask_img = cv2.imread(os.path.join(mask_path, img_name))
        cv2.imshow("{}".format(mask_path.split("/")[-1]), mask_img)
    cv2.waitKey(0)


