import os
import cv2
import shutil

image_path = "/media/hkuit164/Elements1/imagenet/ILSVRC2012_img_val"
mask_paths = ["/media/hkuit164/Elements1/feature_mask"]
move_dest = ""
show_label = True
if show_label:
    import torchvision.transforms as transforms
    import dataloader.imagenet
    transform_val = transforms.Compose([
    ])
    valset = dataloader.imagenet.IN1K(root="/media/hkuit164/Elements1/imagenet", split='val', transform=transform_val)
    labels, names = valset.labels, valset.imgs

img_names = [name for name in os.listdir(mask_paths[0])]

for img_name in img_names:
    if show_label:
        print("{}: {}".format(img_name, valset.class_to_word[valset.idx_to_class[
            labels[names.index("ILSVRC2012_img_val/{}.JPEG".format(img_name))]]]))
    raw_img = cv2.imread(os.path.join(image_path, img_name + ".JPEG"))
    if move_dest:
        os.makedirs(os.path.join(move_dest, img_name), exist_ok=True)
        shutil.copy(os.path.join(image_path, img_name), os.path.join(move_dest, img_name, "raw.jpg"))

    cv2.imshow("raw", raw_img)
    for mask_path in mask_paths:
        mask_img = cv2.imread(os.path.join(mask_path, img_name, "mask_sum.jpg"))
        cv2.imshow("{}".format(mask_path.split("/")[-1]), mask_img)
        if move_dest:
            shutil.copy(os.path.join(mask_path, img_name),
                        os.path.join(move_dest, img_name, "{}.jpg".format(mask_path.split("/")[-1])))
    cv2.waitKey(0)


