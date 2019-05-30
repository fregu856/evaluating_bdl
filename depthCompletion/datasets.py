# code-checked
# server-checked

import cv2
import numpy as np
import os
import random
import torch
from torch.utils import data

################################################################################
# KITTI:
################################################################################
class DatasetKITTIAugmentation(data.Dataset):
    def __init__(self, kitti_depth_path, kitti_rgb_path, max_iters=None, crop_size=(352, 352)):
        self.crop_h, self.crop_w = crop_size

        self.kitti_depth_train_path = kitti_depth_path + "/train"
        self.kitti_rgb_train_path = kitti_rgb_path + "/train"

        train_dir_names = os.listdir(self.kitti_depth_train_path) # (contains "2011_09_26_drive_0001_sync" and so on)

        self.examples = []
        for dir_name in train_dir_names:
            groundtruth_dir_path_02 = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/groundtruth/image_02"
            file_ids_02 = os.listdir(groundtruth_dir_path_02) # (contains e.g. "0000000005.png" and so on)
            for file_id in file_ids_02:
                target_path = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/groundtruth/image_02/" + file_id
                sparse_path = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/velodyne_raw/image_02/" + file_id
                img_path = self.kitti_rgb_train_path + "/" + dir_name + "/image_02/data/" + file_id

                example = {}
                example["img_path"] = img_path
                example["sparse_path"] = sparse_path
                example["target_path"] = target_path
                example["file_id"] = groundtruth_dir_path_02 + "/" + file_id
                self.examples.append(example)

            groundtruth_dir_path_03 = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/groundtruth/image_03"
            file_ids_03 = os.listdir(groundtruth_dir_path_03) # (contains e.g. "0000000005.png" and so on)
            for file_id in file_ids_03:
                target_path = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/groundtruth/image_03/" + file_id
                sparse_path = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/velodyne_raw/image_03/" + file_id
                img_path = self.kitti_rgb_train_path + "/" + dir_name + "/image_03/data/" + file_id

                example = {}
                example["img_path"] = img_path
                example["sparse_path"] = sparse_path
                example["target_path"] = target_path
                example["file_id"] = groundtruth_dir_path_03 + "/" + file_id
                self.examples.append(example)

        print ("DatasetKITTIAugmentation - num unique examples: %d" % len(self.examples))
        if max_iters is not None:
            self.examples = self.examples*int(np.ceil(float(max_iters)/len(self.examples)))
        print ("DatasetKITTIAugmentation - num examples: %d" % len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        sparse_path = example["sparse_path"]
        target_path = example["target_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (375, 1242, 3), dtype: uint8) (or something close to (375, 1242))
        sparse = cv2.imread(sparse_path, -1) # (shape: (375, 1242), dtype: uint16)
        target = cv2.imread(target_path, -1) # (shape: (375, 1242), dtype: uint16)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # crop to the bottom center (352, 1216):
        new_img_h = 352
        new_img_w = 1216 # (this is the image size of all images in the selected val/test sets)

        img_h = img.shape[0]
        img_w = img.shape[1]

        img = img[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216, 3))
        sparse = sparse[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216))
        target = target[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # flip img, sparse and target along the vertical axis with 0.5 probability:
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            sparse = cv2.flip(sparse, 1)
            target = cv2.flip(target, 1)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # select a random (crop_h, crop_w) crop:
        img_h, img_w = sparse.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        img = img[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w, 3))
        sparse = sparse[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w))
        target = target[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # convert sparse and target to meters:
        sparse = sparse/256.0
        sparse = sparse.astype(np.float32)
        target = target/256.0
        target = target.astype(np.float32)

        # convert img to grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (352, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img = img.astype(np.float32)

        return (img.copy(), sparse.copy(), target.copy(), file_id)

class DatasetKITTIVal(data.Dataset):
    def __init__(self, kitti_depth_path):
        self.kitti_depth_val_path = kitti_depth_path + "/depth_selection/val_selection_cropped"

        img_dir = self.kitti_depth_val_path + "/image"
        sparse_dir = self.kitti_depth_val_path + "/velodyne_raw"
        target_dir = self.kitti_depth_val_path + "/groundtruth_depth"

        img_ids = os.listdir(img_dir) # (contains "2011_09_26_drive_0002_sync_image_0000000005_image_02.png" and so on)

        self.examples = []
        for img_id in img_ids:
            # (img_id == "2011_09_26_drive_0002_sync_image_0000000005_image_02.png" (e.g.))

            img_path = img_dir + "/" + img_id

            file_id_start, file_id_end = img_id.split("_sync_image_")
            # (file_id_start == "2011_09_26_drive_0002")
            # (file_id_end == "0000000005_image_02.png")

            sparse_path = sparse_dir + "/" + file_id_start + "_sync_velodyne_raw_" + file_id_end

            target_path = target_dir + "/" + file_id_start + "_sync_groundtruth_depth_" + file_id_end

            example = {}
            example["img_path"] = img_path
            example["sparse_path"] = sparse_path
            example["target_path"] = target_path
            example["file_id"] = img_id
            self.examples.append(example)

        print ("DatasetKITTIVal - num examples: %d" % len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        sparse_path = example["sparse_path"]
        target_path = example["target_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (352, 1216, 3), dtype: uint8))
        sparse = cv2.imread(sparse_path, -1) # (shape: (352, 1216), dtype: uint16)
        target = cv2.imread(target_path, -1) # (shape: (352, 1216), dtype: uint16)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # convert sparse and target to meters:
        sparse = sparse/256.0
        sparse = sparse.astype(np.float32)
        target = target/256.0
        target = target.astype(np.float32)

        # convert img to grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (352, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img = img.astype(np.float32)

        return (img.copy(), sparse.copy(), target.copy(), file_id)

class DatasetKITTIValSeq(data.Dataset):
    def __init__(self, kitti_depth_path, kitti_raw_path, seq="2011_09_26_drive_0002"):
        kitti_depth_val_seq_path = kitti_depth_path + "/val/" + seq + "_sync"

        sparse_dir = kitti_depth_val_seq_path + "/proj_depth/velodyne_raw/image_02"
        target_dir = kitti_depth_val_seq_path + "/proj_depth/groundtruth/image_02"

        seq_date = seq.split("_drive")[0] # (seq_date == "2011_09_26")
        img_dir = kitti_raw_path + "/" + seq_date + "/" + seq + "_sync/image_02/data"

        self.ids = os.listdir(sparse_dir) # (contains "0000000005.png" and so on)

        self.examples = []
        for id in self.ids:
            # (id == "0000000005.png" (e.g.))

            img_path = img_dir + "/" + id
            sparse_path = sparse_dir + "/" + id
            target_path = target_dir + "/" + id

            example = {}
            example["img_path"] = img_path
            example["sparse_path"] = sparse_path
            example["target_path"] = target_path
            example["file_id"] = id
            self.examples.append(example)

        print ("DatasetKITTIValSeq - num examples: %d" % len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        sparse_path = example["sparse_path"]
        target_path = example["target_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (375, 1242, 3), dtype: uint8) (or something close to (375, 1242))
        sparse = cv2.imread(sparse_path, -1) # (shape: (375, 1242), dtype: uint16)
        target = cv2.imread(target_path, -1) # (shape: (375, 1242), dtype: uint16)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # crop to the bottom center (352, 1216):
        new_img_h = 352
        new_img_w = 1216 # (this is the image size of all images in the selected val/test sets)

        img_h = img.shape[0]
        img_w = img.shape[1]

        img = img[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (256, 1216, 3))
        sparse = sparse[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (256, 1216))
        target = target[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (256, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # convert sparse and target to meters:
        sparse = sparse/256.0
        sparse = sparse.astype(np.float32)
        target = target/256.0
        target = target.astype(np.float32)

        # convert img to grayscale:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (352, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        #
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img_gray = img_gray.astype(np.float32)

        return (img_gray.copy(), sparse.copy(), target.copy(), file_id, img)






################################################################################
# virtualKITTI:
################################################################################
class DatasetVirtualKITTIAugmentation(data.Dataset):
    def __init__(self, virtualkitti_path, max_iters=None, crop_size=(352, 352)):
        self.crop_h, self.crop_w = crop_size

        depthgt_path = virtualkitti_path + "/vkitti_1.3.1_depthgt"
        rgb_path = virtualkitti_path + "/vkitti_1.3.1_rgb"

        train_dir_names = ["0001", "0006", "0018", "0020"]

        variation_dir_names = ["15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone", "fog", "morning", "overcast", "rain", "sunset"]

        self.examples = []
        for train_dir_name in train_dir_names:
            ids = os.listdir(depthgt_path + "/" + train_dir_name + "/clone") # (contains "00000.png" and so on)
            for id in ids:
                for variation_dir_name in variation_dir_names:
                    file_id = train_dir_name + "/" + variation_dir_name + "/" + id

                    img_path = rgb_path + "/" + file_id

                    gt_path = depthgt_path + "/" + file_id

                    example = {}
                    example["img_path"] = img_path
                    example["gt_path"] = gt_path
                    example["file_id"] = file_id
                    self.examples.append(example)

        print ("DatasetVirtualKITTIAugmentation - num unique examples: %d" % len(self.examples))
        if max_iters is not None:
            self.examples = self.examples*int(np.ceil(float(max_iters)/len(self.examples)))
        print ("DatasetVirtualKITTIAugmentation - num examples: %d" % len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        gt_path = example["gt_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (375, 1242, 3), dtype: uint8) (or something close to (375, 1242))
        gt = cv2.imread(gt_path, -1) # (shape: (375, 1242), dtype: uint16)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # crop to the bottom center (352, 1216):
        new_img_h = 352
        new_img_w = 1216 # (this is the image size of all images in the selected val/test sets of kitti-depth)

        img_h = img.shape[0]
        img_w = img.shape[1]

        img = img[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216, 3))
        gt = gt[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # flip img and gt along the vertical axis with 0.5 probability:
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            gt = cv2.flip(gt, 1)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # select a random (crop_h, crop_w) crop:
        img_h, img_w = gt.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        img = img[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w, 3))
        gt = gt[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # convert gt to meters:
        gt = gt/100.0
        gt = gt.astype(np.float32)

        # convert img to grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (crop_h, crop_w))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # create sparse and target from gt:
        max_distance = 80.0
        prob_keep = 0.05

        target = gt.copy()
        target[target > max_distance] = 0

        sparse = target.copy()
        mask = np.random.binomial(1, prob_keep, sparse.shape)
        sparse = mask*sparse

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # target = (target/max_distance)*255
        # target = target.astype(np.uint8)
        #
        # sparse = (sparse/max_distance)*255
        # sparse = sparse.astype(np.uint8)
        #
        # sparse_color = cv2.applyColorMap(sparse, cv2.COLORMAP_JET)
        # sparse_color[sparse == 0] = 0
        #
        # target_color = cv2.applyColorMap(target, cv2.COLORMAP_JET)
        # target_color[target == 0] = 0
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        # cv2.imshow("sparse_color", sparse_color)
        # cv2.waitKey(0)
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # cv2.imshow("target_color", target_color)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img = img.astype(np.float32)
        sparse = sparse.astype(np.float32)
        target = target.astype(np.float32)

        return (img.copy(), sparse.copy(), target.copy(), file_id)

class DatasetVirtualKITTIVal(data.Dataset):
    def __init__(self, virtualkitti_path):
        depthgt_path = virtualkitti_path + "/vkitti_1.3.1_depthgt"
        rgb_path = virtualkitti_path + "/vkitti_1.3.1_rgb"

        val_dir_names = ["0002"]

        variation_dir_names = ["15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone", "fog", "morning", "overcast", "rain", "sunset"]

        self.examples = []
        for val_dir_name in val_dir_names:
            ids = os.listdir(depthgt_path + "/" + val_dir_name + "/clone") # (contains "00000.png" and so on)
            for id in ids:
                for variation_dir_name in variation_dir_names:
                    file_id = val_dir_name + "/" + variation_dir_name + "/" + id

                    img_path = rgb_path + "/" + file_id

                    gt_path = depthgt_path + "/" + file_id

                    example = {}
                    example["img_path"] = img_path
                    example["gt_path"] = gt_path
                    example["file_id"] = file_id
                    self.examples.append(example)

        print ("DatasetVirtualKITTIVal - num examples: %d" % len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        gt_path = example["gt_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (375, 1242, 3), dtype: uint8) (or something close to (375, 1242))
        gt = cv2.imread(gt_path, -1) # (shape: (375, 1242), dtype: uint16)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # crop to the bottom center (352, 1216):
        new_img_h = 352
        new_img_w = 1216 # (this is the image size of all images in the selected val/test sets of kitti-depth)

        img_h = img.shape[0]
        img_w = img.shape[1]

        img = img[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216, 3))
        gt = gt[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # convert gt to meters:
        gt = gt/100.0
        gt = gt.astype(np.float32)

        # convert img to grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (crop_h, crop_w))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # create sparse and target from gt:
        max_distance = 80.0
        prob_keep = 0.05

        target = gt.copy()
        target[target > max_distance] = 0

        sparse = target.copy()
        mask = np.random.binomial(1, prob_keep, sparse.shape)
        sparse = mask*sparse

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # target = (target/max_distance)*255
        # target = target.astype(np.uint8)
        #
        # sparse = (sparse/max_distance)*255
        # sparse = sparse.astype(np.uint8)
        #
        # sparse_color = cv2.applyColorMap(sparse, cv2.COLORMAP_JET)
        # sparse_color[sparse == 0] = 0
        #
        # target_color = cv2.applyColorMap(target, cv2.COLORMAP_JET)
        # target_color[target == 0] = 0
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        # cv2.imshow("sparse_color", sparse_color)
        # cv2.waitKey(0)
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # cv2.imshow("target_color", target_color)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img = img.astype(np.float32)
        sparse = sparse.astype(np.float32)
        target = target.astype(np.float32)

        return (img.copy(), sparse.copy(), target.copy(), file_id)

class DatasetVirtualKITTIValSeq(data.Dataset):
    def __init__(self, virtualkitti_path, seq="0002", variation="clone"):
        depthgt_path = virtualkitti_path + "/vkitti_1.3.1_depthgt"
        rgb_path = virtualkitti_path + "/vkitti_1.3.1_rgb"

        self.examples = []
        self.ids = os.listdir(depthgt_path + "/" + seq + "/clone") # (contains "00000.png" and so on)
        for id in self.ids:
            file_id = seq + "/" + variation + "/" + id

            img_path = rgb_path + "/" + file_id

            gt_path = depthgt_path + "/" + file_id

            example = {}
            example["img_path"] = img_path
            example["gt_path"] = gt_path
            example["file_id"] = file_id
            self.examples.append(example)

        print ("DatasetVirtualKITTIValSeq - num examples: %d" % len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        gt_path = example["gt_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (375, 1242, 3), dtype: uint8) (or something close to (375, 1242))
        gt = cv2.imread(gt_path, -1) # (shape: (375, 1242), dtype: uint16)

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # crop to the bottom center (352, 1216):
        new_img_h = 352
        new_img_w = 1216 # (this is the image size of all images in the selected val/test sets of kitti-depth)

        img_h = img.shape[0]
        img_w = img.shape[1]

        img = img[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216, 3))
        gt = gt[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # convert gt to meters:
        gt = gt/100.0
        gt = gt.astype(np.float32)

        # convert img to grayscale:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (crop_h, crop_w))

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (gt.shape)
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("gt", gt)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # create sparse and target from gt:
        max_distance = 80.0
        prob_keep = 0.05

        target = gt.copy()
        target[target > max_distance] = 0

        sparse = target.copy()
        mask = np.random.binomial(1, prob_keep, sparse.shape)
        sparse = mask*sparse

        # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (img.shape)
        # print (sparse.shape)
        # print (target.shape)
        #
        # target = (target/max_distance)*255
        # target = target.astype(np.uint8)
        #
        # sparse = (sparse/max_distance)*255
        # sparse = sparse.astype(np.uint8)
        #
        # sparse_color = cv2.applyColorMap(sparse, cv2.COLORMAP_JET)
        # sparse_color[sparse == 0] = 0
        #
        # target_color = cv2.applyColorMap(target, cv2.COLORMAP_JET)
        # target_color[target == 0] = 0
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("sparse", sparse)
        # cv2.waitKey(0)
        # cv2.imshow("sparse_color", sparse_color)
        # cv2.waitKey(0)
        #
        # cv2.imshow("target", target)
        # cv2.waitKey(0)
        # cv2.imshow("target_color", target_color)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img_gray = img_gray.astype(np.float32)
        sparse = sparse.astype(np.float32)
        target = target.astype(np.float32)

        return (img_gray.copy(), sparse.copy(), target.copy(), file_id, img)
