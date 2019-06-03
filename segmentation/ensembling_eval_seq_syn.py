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

from datasets import DatasetSynscapesEval
from models.model import get_model

from utils.utils import label_img_2_color, get_confusion_matrix

model_id = "ensembling_syn"
#model_id = "ensembling"
model_is = [0, 1, 2, 3, 4, 5, 6, 7]
print (model_is)

data_dir = "/home/data/synscapes"
synscapes_meta_path = "/home/data/synscapes_meta"
batch_size = 2
num_classes = 19
max_entropy = np.log(num_classes)

eval_dataset = DatasetSynscapesEval(root=data_dir, root_meta=synscapes_meta_path, type="val")
eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

output_path = "/home/evaluating_bdl/segmentation/training_logs/%s_%s_eval_seq_syn" % (model_id, str(model_is))
if not os.path.exists(output_path):
    os.makedirs(output_path)

models = []
for i in model_is:
    restore_from = "/home/evaluating_bdl/segmentation/trained_models/%s_%d/checkpoint_40000.pth" % (model_id, i)
    deeplab = get_model(num_classes=num_classes)
    deeplab.load_state_dict(torch.load(restore_from))
    model = nn.DataParallel(deeplab)
    model.eval()
    model.cuda()
    models.append(model)

M_float = float(len(models))
print (M_float)

names = []
confusion_matrix = np.zeros((num_classes, num_classes))
for step, batch in enumerate(eval_loader):
    with torch.no_grad():
        print ("%d/%d" % (step+1, len(eval_loader)))

        image, label, _, name = batch
        # (image has shape: (batch_size, 3, h, w))
        # (label has shape: (batch_size, h, w))

        batch_size = image.size(0)
        h = image.size(2)
        w = image.size(3)

        p = torch.zeros(batch_size, num_classes, h, w).cuda() # (shape: (batch_size, num_classes, h, w))
        for model in models:
            logits_downsampled = model(Variable(image).cuda()) # (shape: (batch_size, num_classes, h/8, w/8))
            logits = F.upsample(input=logits_downsampled , size=(h, w), mode='bilinear', align_corners=True) # (shape: (batch_size, num_classes, h, w))
            p_value = F.softmax(logits, dim=1) # (shape: (batch_size, num_classes, h, w))
            p = p + p_value/M_float

        p_numpy = p.cpu().data.numpy().transpose(0, 2, 3, 1) # (array of shape: (batch_size, h, w, num_classes))

        seg_pred = np.argmax(p_numpy, axis=3).astype(np.uint8)
        m_seg_pred = np.ma.masked_array(seg_pred, mask=torch.eq(label, 255))
        np.ma.set_fill_value(m_seg_pred, 20)
        seg_pred = m_seg_pred

        seg_gt = label.numpy().astype(np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, num_classes)

        entropy = -np.sum(p_numpy*np.log(p_numpy), axis=3) # (shape: (batch_size, h, w))
        pred_label_imgs_raw = np.argmax(p_numpy, axis=3).astype(np.uint8)
        for i in range(image.size(0)):
            img = image[i].data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
            img = img + np.array([102.9801, 115.9465, 122.7717])
            img = img[:,:,::-1]
            cv2.imwrite(output_path + "/" + name[i] + "_img.png", img)

            label_img = label[i].data.cpu().numpy()
            label_img = label_img.astype(np.uint8)
            label_img_color = label_img_2_color(label_img)[:,:,::-1]
            overlayed_img = 0.30*img + 0.70*label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)
            cv2.imwrite(output_path + "/" + name[i] + "_label_overlayed.png", overlayed_img)

            pred_label_img = pred_label_imgs_raw[i]
            pred_label_img = pred_label_img.astype(np.uint8)
            pred_label_img_color = label_img_2_color(pred_label_img)[:,:,::-1]
            overlayed_img = 0.30*img + 0.70*pred_label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)
            cv2.imwrite(output_path + "/" + name[i] + "_pred_overlayed.png", overlayed_img)

            entropy_img = entropy[i]
            entropy_img = (entropy_img/max_entropy)*255
            entropy_img = entropy_img.astype(np.uint8)
            entropy_img = cv2.applyColorMap(entropy_img, cv2.COLORMAP_HOT)
            cv2.imwrite(output_path + "/" + name[i] + "_entropy.png", entropy_img)

            names.append(name[i])

        if (step+1)*batch_size > 30: # (create video of 30 examples)
            break

pos = confusion_matrix.sum(1)
res = confusion_matrix.sum(0)
tp = np.diag(confusion_matrix)

IU_array = (tp / np.maximum(1.0, pos + res - tp))
mean_IU = IU_array.mean()
print({'meanIU':mean_IU, 'IU_array':IU_array})

# (names contains "10832" etc.)

out = cv2.VideoWriter("%s/video.avi" % output_path, cv2.VideoWriter_fourcc(*"MJPG"), 1, (2*w, 2*h))
for step, name in enumerate(names):
    if step % 10 == 0:
        print ("step: %d/%d" % (step+1, len(names)))

    img = cv2.imread(output_path + "/" + name + "_img.png", -1)
    label_overlayed = cv2.imread(output_path + "/" + name + "_label_overlayed.png", -1)
    pred_overlayed = cv2.imread(output_path + "/" + name + "_pred_overlayed.png", -1)
    entropy = cv2.imread(output_path + "/" + name + "_entropy.png", -1)

    combined_img = np.zeros((2*h, 2*w, 3), dtype=np.uint8)

    combined_img[0:h, 0:w] = img
    combined_img[0:h, w:2*w] = label_overlayed
    combined_img[h:2*h, 0:w] = pred_overlayed
    combined_img[h:2*h, w:2*w] = entropy

    out.write(combined_img)
    out.write(combined_img)
    out.write(combined_img) # (write the same image 3 times to get 0.33 FPS)

out.release()
