# code-checked
# server-checked

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import os
from utils.criterion import CriterionCrossEntropy
from utils.parallel import DataParallelModel, DataParallelCriterion

from models.model_mcdropout import get_model
from dataset.cityscapes_cleaned10 import CitySegmentationTrain

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_id = "mcdropout"

learning_rate = 0.01
power = 0.9
crop_size = (512, 512)
num_classes = 19
restore_from = "/home/evaluating_bdl/segmentation/resnet101-imagenet.pth"
snapshot_dir_base = "/home/evaluating_bdl/segmentation/training_logs/%s" % model_id
data_dir = "/home/data/cityscapes"
data_list = "/home/evaluating_bdl/segmentation/dataset/list/cityscapes/train.lst"
batch_size = 8
random_mirror = True
random_scale = True
momentum = 0.9
weight_decay = 0.0005
save_pred_every = 5000
num_steps = 60000
ignore_label = 255

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

M = 8
for model_i in range(M):
    snapshot_dir = snapshot_dir_base + "_%d/" % model_i
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    deeplab = get_model(num_classes=num_classes)

    # load pretrained ResNet101 backbone:
    saved_state_dict = torch.load(restore_from)
    new_params = deeplab.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0]=='fc' and not  i_parts[0]=='last_linear' and not  i_parts[0]=='classifier':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    deeplab.load_state_dict(new_params)

    model = DataParallelModel(deeplab)
    model.train()
    model.float()
    model.cuda()

    criterion = CriterionCrossEntropy()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    train_dataset = CitySegmentationTrain(root=data_dir, list_path=data_list, max_iters=num_steps*batch_size, crop_size=crop_size)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': learning_rate }],
                lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer.zero_grad()

    train_losses = []
    batch_train_losses = []
    for i_iter, batch in enumerate(train_loader):
        images, labels, _, _ = batch
        images = Variable(images.cuda())
        labels = Variable(labels.long().cuda())

        preds = model(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        loss.backward()
        optimizer.step()

        print ("model %d/%d, iter %d/%d, loss: %g, lr: %g" % (model_i+1, M, i_iter, num_steps, loss.data.cpu().numpy(), lr))

        batch_train_losses.append(loss.data.cpu().numpy())
        if i_iter % 500 == 0:
            train_loss = np.mean(batch_train_losses)
            train_losses.append(train_loss)
            with open("%strain_losses.pkl" % snapshot_dir, "wb") as file:
                pickle.dump(train_losses, file)
            plt.figure(1)
            plt.plot(train_losses, "k^")
            plt.plot(train_losses, "k")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.title("train losses")
            plt.savefig("%strain_losses.png" % snapshot_dir)
            plt.close(1)
            batch_train_losses = []

        if i_iter >= num_steps-1:
            print("saving model...")
            torch.save(deeplab.state_dict(), snapshot_dir + "checkpoint_" + str(num_steps) + ".pth")
            break

        if i_iter % save_pred_every == 0:
            print("taking snapshot...")
            torch.save(deeplab.state_dict(), snapshot_dir + "checkpoint_" + str(i_iter) + ".pth")
