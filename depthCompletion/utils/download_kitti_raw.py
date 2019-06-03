# code-checked
# server-checked

import os

# NOTE! NOTE! NOTE! make sure you run this code inside the kitti_raw directory (/root/data/kitti_raw)

kitti_depth_path = "/root/data/kitti_depth"

train_dirs = os.listdir(kitti_depth_path + "/train") # (contains "2011_09_26_drive_0001_sync" and so on)
val_dirs = os.listdir(kitti_depth_path + "/val")
dirs = train_dirs + val_dirs

print ("num train dirs: %d" % len(train_dirs))
print ("num val dirs: %d" % len(val_dirs))
print ("num dirs: %d" % len(dirs))

for step, dir_name in enumerate(dirs):
    print ("##################################################################")
    print ("step %d/%d" % (step+1, len(dirs)))
    print (dir_name)

    # (dir_name == "2011_09_26_drive_0001_sync" (for example))

    dir_name_no_sync = dir_name.split("_sync")[0] # (dir_name_no_sync == "2011_09_26_drive_0001")

    # download the zip file:
    os.system("wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/%s/%s.zip" % (dir_name_no_sync, dir_name))

    # unzip:
    os.system("unzip %s.zip" % dir_name)
