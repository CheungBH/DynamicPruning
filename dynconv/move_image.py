import os, shutil

src_path = "/media/ssd0/imagenet"
dest_path = "/media/ssd0/imagenet/validation"
os.makedirs(dest_path, exist_ok=True)

for name in os.listdir(src_path):
    if "ILSVRC2012_val" in name and ".JPEG" in name:
        shutil.move(os.path.join(src_path, name), os.path.join(dest_path, name))
