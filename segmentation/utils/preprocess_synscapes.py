# code-checked
# server-checked

import pickle
import numpy as np
import cv2
import os
from collections import namedtuple
import random

# (NOTE! this is taken from the official Cityscapes scripts:)
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# create a function which maps id to trainId:
id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

synscapes_path = "/home/data/synscapes"
synscapes_meta_path = "/home/data/synscapes_meta"

if not os.path.exists(synscapes_meta_path):
    os.makedirs(synscapes_meta_path)
if not os.path.exists(synscapes_meta_path + "/gtFine"):
    os.makedirs(synscapes_meta_path + "/gtFine")
if not os.path.exists(synscapes_meta_path + "/label_imgs"):
    os.makedirs(synscapes_meta_path + "/label_imgs")

img_h = 720
img_w = 1440

new_img_h = 1024
new_img_w = 2048

################################################################################
# randomly select a subset of 2975 images as train and 500 images as val:
################################################################################
img_ids_float = np.linspace(1, 25000, 25000)
img_ids = []
for img_id_float in img_ids_float:
    img_id_str = str(int(img_id_float))
    img_ids.append(img_id_str)

random.shuffle(img_ids)
random.shuffle(img_ids)
random.shuffle(img_ids)
random.shuffle(img_ids)

train_img_ids = img_ids[0:2975]
print ("num train images: %d" % len(train_img_ids))
with open(synscapes_meta_path + "/train_img_ids.pkl", "wb") as file:
    pickle.dump(train_img_ids, file)

val_img_ids = img_ids[2975:(2975+500)]
print ("num val images: %d" % len(val_img_ids))
with open(synscapes_meta_path + "/val_img_ids.pkl", "wb") as file:
    pickle.dump(val_img_ids, file)

################################################################################
# enlarge all train labels and save to disk:
################################################################################
label_dir = synscapes_path + "/img/class/"
for (step, img_id) in enumerate(train_img_ids):
    if (step % 100) == 0:
        print ("enlarging train labels, step: %d/%d" % (step+1, len(train_img_ids)))

    gtFine_img_path = label_dir + img_id + ".png"
    gtFine_img = cv2.imread(gtFine_img_path, -1) # (shape: (720, 1440))

    # resize gtFine_img without interpolation:
    gtFine_img = cv2.resize(gtFine_img, (new_img_w, new_img_h), interpolation=cv2.INTER_NEAREST) # (shape: (1024, 2048))

    cv2.imwrite(synscapes_meta_path + "/gtFine/" + img_id + ".png", gtFine_img)

    # convert gtFine_img from id to trainId pixel values:
    label_img = id_to_trainId_map_func(gtFine_img) # (shape: (1024, 2048))
    label_img = label_img.astype(np.uint8)

    cv2.imwrite(synscapes_meta_path + "/label_imgs/" + img_id + ".png", label_img)

################################################################################
# enlarge all val labels and save to disk:
################################################################################
label_dir = synscapes_path + "/img/class/"
for (step, img_id) in enumerate(val_img_ids):
    if (step % 100) == 0:
        print ("enlarging val labels, step: %d/%d" % (step+1, len(val_img_ids)))

    gtFine_img_path = label_dir + img_id + ".png"
    gtFine_img = cv2.imread(gtFine_img_path, -1) # (shape: (720, 1440))

    # resize gtFine_img without interpolation:
    gtFine_img = cv2.resize(gtFine_img, (new_img_w, new_img_h), interpolation=cv2.INTER_NEAREST) # (shape: (1024, 2048))

    cv2.imwrite(synscapes_meta_path + "/gtFine/" + img_id + ".png", gtFine_img)

    # convert gtFine_img from id to trainId pixel values:
    label_img = id_to_trainId_map_func(gtFine_img) # (shape: (1024, 2048))
    label_img = label_img.astype(np.uint8)

    cv2.imwrite(synscapes_meta_path + "/label_imgs/" + img_id + ".png", label_img)

################################################################################
# compute the class weigths:
################################################################################
num_classes = 19

trainId_to_count = {}
for trainId in range(num_classes):
    trainId_to_count[trainId] = 0

# get the total number of pixels in all train label_imgs that are of each object class:
for step, img_id in enumerate(train_img_ids):
    if (step % 100) == 0:
        print ("computing class weights, step: %d/%d" % (step+1, len(train_img_ids)))

    label_img_path = synscapes_meta_path + "/label_imgs/" + img_id + ".png"
    label_img = cv2.imread(label_img_path, -1)

    for trainId in range(num_classes):
        # count how many pixels in label_img which are of object class trainId:
        trainId_mask = np.equal(label_img, trainId)
        trainId_count = np.sum(trainId_mask)

        # add to the total count:
        trainId_to_count[trainId] += trainId_count

# compute the class weights according to the ENet paper:
class_weights = []
total_count = sum(trainId_to_count.values())
for trainId, count in trainId_to_count.items():
    trainId_prob = float(count)/float(total_count)
    trainId_weight = 1/np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

print (class_weights)

with open(synscapes_meta_path + "/class_weights.pkl", "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
