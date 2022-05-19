import os
import cv2
import shutil

image_path = "/media/hkuit155/NewDisk/imagenet/val"
mask_paths = ["/media/hkuit155/NewDisk/mask/s50", "/media/hkuit155/NewDisk/mask/s75", "/media/hkuit155/NewDisk/mask/s25",
              "/media/hkuit155/NewDisk/mask/s50_resmask", "/media/hkuit155/NewDisk/mask/s75_resmask", "/media/hkuit155/NewDisk/mask/s25_resmask"]
move_dest = "sample/masks"

img_names = [name for name in os.listdir(mask_paths[0])]

for img_name in img_names:
    raw_img = cv2.imread(os.path.join(image_path, img_name))
    if move_dest:
        os.makedirs(os.path.join(move_dest, img_name), exist_ok=True)
        shutil.copy(os.path.join(image_path, img_name), os.path.join(move_dest, img_name, "raw.jpg"))

    cv2.imshow("raw", raw_img)
    for mask_path in mask_paths:
        mask_img = cv2.imread(os.path.join(mask_path, img_name))
        cv2.imshow("{}".format(mask_path.split("/")[-1]), mask_img)
        if move_dest:
            shutil.copy(os.path.join(mask_path, img_name),
                        os.path.join(move_dest, img_name, "{}.jpg".format(mask_path.split("/")[-1])))
    cv2.waitKey(0)


