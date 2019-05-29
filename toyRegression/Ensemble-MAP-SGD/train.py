# code-checked
# server-checked

from datasets import ToyDataset # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# NOTE! change this to not overwrite all log data when you train the model:
model_id = "Ensemble-MAP-SGD_1_M1024"

num_epochs = 150
batch_size = 32
learning_rate = 0.01

power = 0.9

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

train_dataset = ToyDataset()
N = float(len(train_dataset))
print (N)

alpha = 1.0

num_train_batches = int(len(train_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)

num_steps = num_epochs*num_train_batches

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

M = 1024
for i in range(M):
    network = ToyNet(model_id + "_%d" % i, project_dir="/root/evaluating_bdl/toyRegression").cuda()

    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    epoch_losses_train = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("epoch: %d/%d" % (epoch+1, num_epochs))
        print ("network: %d/%d" % (i+1, M))

        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (x, y) in enumerate(train_loader):
            x = Variable(x).cuda().unsqueeze(1) # (shape: (batch_size, 1))
            y = Variable(y).cuda().unsqueeze(1) # (shape: (batch_size, 1))

            outputs = network(x)
            mean = outputs[0] # (shape: (batch_size, ))
            log_var = outputs[1] # (shape: (batch_size, )) (log(sigma^2))

            ####################################################################
            # compute the loss:
            ####################################################################
            loss_likelihood = torch.mean(torch.exp(-log_var)*torch.pow(y - mean, 2) + log_var)

            loss_prior = 0.0
            for param in network.parameters():
                if param.requires_grad:
                    loss_prior += (1.0/N)*(1.0/alpha)*torch.sum(torch.pow(param, 2))

            loss = loss_likelihood + loss_prior

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
            lr = adjust_learning_rate(optimizer, epoch*num_train_batches + step)

            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
        plt.close(1)

        print('learning_rate: {}'.format(lr))

        # save the model weights to disk:
        checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(network.state_dict(), checkpoint_path)
