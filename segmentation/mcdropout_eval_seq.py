# code-checked
# server-checked

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

import os
import numpy as np
import cv2

from datasets import DatasetCityscapesEvalSeq
from models.model import get_model

from utils.utils import label_img_2_color

model_id = "mcdropout_syn_0"
M = 8

data_dir = "/home/data/cityscapes"
batch_size = 4
num_classes = 19
max_entropy = np.log(num_classes)

output_path = "/home/evaluating_bdl/segmentation/training_logs/%s_M%d_eval_seq" % (model_id, M)
if not os.path.exists(output_path):
    os.makedirs(output_path)

restore_from = "/home/evaluating_bdl/segmentation/trained_models/%s/checkpoint_60000.pth" % model_id
deeplab = get_model(num_classes=num_classes)
deeplab.load_state_dict(torch.load(restore_from))
model = nn.DataParallel(deeplab)
model.eval()
model.cuda()

M_float = float(M)
print (M_float)

demo_sequences = ["00", "01", "02"]
for step, seq in enumerate(demo_sequences):
    print ("##################################################################")
    print ("seq: %d/%d, %s" % (step+1, len(demo_sequences), seq))

    output_path_seq = output_path + "/" + seq
    if not os.path.exists(output_path_seq):
        os.makedirs(output_path_seq)

    eval_dataset = DatasetCityscapesEvalSeq(data_path=data_dir, sequence=seq)
    eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    names = []
    for step, batch in enumerate(eval_loader):
        with torch.no_grad():
            print ("%d/%d" % (step+1, len(eval_loader)))

            image, _, name = batch
            # (image has shape: (batch_size, 3, h, w))

            batch_size = image.size(0)
            h = image.size(2)
            w = image.size(3)

            p = torch.zeros(batch_size, num_classes, h, w).cuda() # (shape: (batch_size, num_classes, h, w))
            for i in range(M):
                logits_downsampled = model(Variable(image).cuda()) # (shape: (batch_size, num_classes, h/8, w/8))
                logits = F.upsample(input=logits_downsampled , size=(h, w), mode='bilinear', align_corners=True) # (shape: (batch_size, num_classes, h, w))
                p_value = F.softmax(logits, dim=1) # (shape: (batch_size, num_classes, h, w))
                p = p + p_value/M_float

            p_numpy = p.cpu().data.numpy().transpose(0, 2, 3, 1) # (array of shape: (batch_size, h, w, num_classes))

            entropy = -np.sum(p_numpy*np.log(p_numpy), axis=3) # (shape: (batch_size, h, w))
            pred_label_imgs_raw = np.argmax(p_numpy, axis=3).astype(np.uint8)
            for i in range(image.size(0)):
                img = image[i].data.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                img = img + np.array([102.9801, 115.9465, 122.7717])
                img = img[:,:,::-1]
                cv2.imwrite(output_path_seq + "/" + name[i] + "_img.png", img)

                pred_label_img = pred_label_imgs_raw[i]
                pred_label_img = pred_label_img.astype(np.uint8)
                pred_label_img_color = label_img_2_color(pred_label_img)[:,:,::-1]
                overlayed_img = 0.30*img + 0.70*pred_label_img_color
                overlayed_img = overlayed_img.astype(np.uint8)
                cv2.imwrite(output_path_seq + "/" + name[i] + "_pred_overlayed.png", overlayed_img)

                entropy_img = entropy[i]
                entropy_img = (entropy_img/max_entropy)*255
                entropy_img = entropy_img.astype(np.uint8)
                entropy_img = cv2.applyColorMap(entropy_img, cv2.COLORMAP_HOT)
                cv2.imwrite(output_path_seq + "/" + name[i] + "_entropy.png", entropy_img)

                names.append(name[i])

            # # # # # # # # # # # # # # # # # # debug START:
            # if step > 0:
            #     break
            # # # # # # # # # # # # # # # # # # debug END:

    # (names contains "stuttgart_00_000000_000030", "stuttgart_00_000000_000031" etc.)
    names_sorted = sorted(names)

    out = cv2.VideoWriter("%s/%s.avi" % (output_path_seq, seq), cv2.VideoWriter_fourcc(*"MJPG"), 20, (2*w, 2*h))
    for step, name in enumerate(names_sorted):
        if step % 10 == 0:
            print ("step: %d/%d" % (step+1, len(names_sorted)))

        img = cv2.imread(output_path_seq + "/" + name + "_img.png", -1)
        pred_overlayed = cv2.imread(output_path_seq + "/" + name + "_pred_overlayed.png", -1)
        entropy = cv2.imread(output_path_seq + "/" + name + "_entropy.png", -1)

        combined_img = np.zeros((2*h, 2*w, 3), dtype=np.uint8)

        combined_img[0:h, int(0.5*w):int(1.5*w)] = img
        combined_img[h:2*h, 0:w] = pred_overlayed
        combined_img[h:2*h, w:2*w] = entropy

        out.write(combined_img)

    out.release()
