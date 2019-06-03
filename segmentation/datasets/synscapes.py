import cv2
import numpy as np
import os
import os.path as osp
import random
import torch
from torch.utils import data

import pickle

################################################################################
# Cityscapes
################################################################################







################################################################################
# Synscapes
################################################################################
class DatasetSynscapesAugmentation(data.Dataset):
    def __init__(self, root, root_meta, type="train", max_iters=None, crop_size=(512, 512), ignore_label=255):
        self.root = root
        self.root_meta = root_meta
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label

        if type == "train":
            with open(root_meta + "/train_img_ids.pkl", "rb") as file: # (needed for python3)
                self.img_ids = pickle.load(file)
        elif type == "val":
            with open(root_meta + "/val_img_ids.pkl", "rb") as file: # (needed for python3)
                self.img_ids = pickle.load(file)
        else:
            raise Exception("type must be either 'train' or 'val'!")

        print ("DatasetSynscapesAugmentation - num unique examples: %d" % len(self.img_ids))
        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        print ("DatasetSynscapesAugmentation - num examples: %d" % len(self.img_ids))

        self.files = []
        for img_id in self.img_ids:
            self.files.append({
                "img": self.root + "/img/rgb-2k/" + img_id + ".png",
                "label": self.root_meta + "/gtFine/" + img_id + ".png",
                "name": img_id,
                "weight": 1
            })

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 16)/10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def id2trainId(self, label):
        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        if not os.path.exists(datafiles["img"]): # (26 out of 25000 images are missing)
            return self.__getitem__(0)

        label = self.id2trainId(label)

        size = image.shape
        name = datafiles["name"]
        image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        mean = (102.9801, 115.9465, 122.7717)
        image = image[:,:,::-1]
        image -= mean

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))

        flip = np.random.choice(2)*2 - 1
        image = image[:, :, ::flip]
        label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class DatasetSynscapesEval(data.Dataset):
    def __init__(self, root, root_meta, type="val", ignore_label=255):
        self.root = root
        self.root_meta = root_meta
        self.ignore_label = ignore_label

        if type == "train":
            with open(root_meta + "/train_img_ids.pkl", "rb") as file: # (needed for python3)
                self.img_ids = pickle.load(file)
        elif type == "val":
            with open(root_meta + "/val_img_ids.pkl", "rb") as file: # (needed for python3)
                self.img_ids = pickle.load(file)
        else:
            raise Exception("type must be either 'train' or 'val'!")

        print ("DatasetSynscapesEval - num examples: %d" % len(self.img_ids))

        self.files = []
        for img_id in self.img_ids:
            self.files.append({
                "img": self.root + "/img/rgb-2k/" + img_id + ".png",
                "label": self.root_meta + "/gtFine/" + img_id + ".png",
                "name": img_id,
                "weight": 1
            })

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        if not os.path.exists(datafiles["img"]): # (26 out of 25000 images are missing)
            return self.__getitem__(0)

        label = self.id2trainId(label)

        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)

        mean = (102.9801, 115.9465, 122.7717)
        image = image[:,:,::-1]
        image -= mean

        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
