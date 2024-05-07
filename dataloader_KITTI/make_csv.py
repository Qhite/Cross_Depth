import os
import csv

os.system("clear")

setting = "train"
# setting = "test"

root_path = "/root/MDE_LiDAR/datasets/kitti/"
date_ = os.listdir(root_path)
date_.sort()

ex = ["2011_09_26_drive_0020_sync",
      "2011_09_26_drive_0060_sync",
      "2011_09_26_drive_0091_sync",
      "2011_09_26_drive_0113_sync",
      
      "2011_09_28_drive_0038_sync",
      "2011_09_28_drive_0070_sync",
      "2011_09_28_drive_0087_sync",
      "2011_09_28_drive_0103_sync",
      "2011_09_28_drive_0128_sync",
      "2011_09_28_drive_0156_sync",
      "2011_09_28_drive_0192_sync",
      "2011_09_28_drive_0199_sync",
      "2011_09_28_drive_0220_sync",

      "2011_09_29_drive_071_sync",
      "2011_09_30_drive_0027_sync",
      "2011_10_03_drive_0034_sync"]

f = open(f"/root/MDE_LiDAR/kitti_{setting}.csv", "w")

fc = csv.writer(f)

for d in date_:
    seq = os.listdir(f"{root_path}{d}/")
    seq.sort()
    for s in seq[:-3]:
        if setting == "train":
            if s in ex:
                continue
        elif setting == "test":
            if not s in ex:
                continue
        files = os.listdir(f"{root_path}{d}/{s}/")
        files.sort()

        img_fnames = f"{root_path}{d}/{s}/{files[2]}/data/"
        depth_fnames = f"{root_path}{d}/{s}/{files[5]}/groundtruth/image_02"
        lidar_fnames = f"{root_path}{d}/{s}/{files[6]}/data/"

        imgs = os.listdir(img_fnames)
        imgs.sort()
        depths = os.listdir(depth_fnames)
        depths.sort()
        lidars = os.listdir(lidar_fnames)
        lidars.sort()
        for im, de, li in zip(imgs[5:-5], depths, lidars[5:-5]):
            ifn = f"/{d}/{s}/{files[2]}/data/{im}"
            dfn = f"/{d}/{s}/{files[5]}/groundtruth/image_02/{de}"
            lfn = f"/{d}/{s}/{files[6]}/data/{li}"
            fc.writerow([ifn,dfn,lfn])
            # print(f"{ifn} {dfn} {lfn}")
f.close()