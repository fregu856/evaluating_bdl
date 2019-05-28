# code-checked

import torch
from torch.optim.optimizer import Optimizer, required

class SGD(Optimizer):
    """
    Slightly modified to fit the SGHMC update equations.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                d_p.mul_((-1.0)*group['lr']) ###################################
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf ##############################################

                p.data.add_(d_p)

        return loss

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

import math

for i in range(10):
    # NOTE! change this to not overwrite all log data when you train the model:
    model_id = "SGHMC-64_%d" % (i + 1)

    L = 64
    print ("L: %d" % L)

    eta = 0.1

    num_epochs = L*150
    batch_size = 32
    learning_rate = 0.01

    power = 0.9

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))

    def adjust_learning_rate(optimizer, i_iter):
        lr = lr_poly(learning_rate, i_iter, num_steps, power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    loss_fn = nn.CrossEntropyLoss()

    train_dataset = ToyDataset()
    N = float(len(train_dataset))
    print (N)

    alpha = 1.0

    num_train_batches = int(len(train_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)

    num_steps = num_epochs*num_train_batches + 1

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    network = ToyNet(model_id, project_dir="/root/evaluating_bdl/toyClassification").cuda()

    optimizer = SGD(network.parameters(), momentum=(1.0-eta), lr=learning_rate)

    epoch_losses_train = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("epoch: %d/%d" % (epoch+1, num_epochs))
        print ("model: %d/%d" % (i+1, 10))

        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (x, y) in enumerate(train_loader):
            x = Variable(x).cuda() # (shape: (batch_size, 2))
            y = Variable(y).cuda() # (shape: (batch_size, ))

            logits = network(x) # (shape: (batch_size, num_classes)) (num_classes==2)

            ####################################################################
            # compute the loss:
            ####################################################################
            lr = adjust_learning_rate(optimizer, epoch*num_train_batches + step)

            loss_likelihood = loss_fn(logits, y)

            loss_prior = 0.0
            for param in network.parameters():
                if param.requires_grad:
                    loss_prior += (1.0/2.0)*(1.0/N)*(1.0/alpha)*torch.sum(torch.pow(param, 2))

            loss_noise = 0.0
            for param in network.parameters():
                if param.requires_grad:
                    loss_noise += (1.0/math.sqrt(N))*math.sqrt(2.0*eta/lr)*torch.sum(param*Variable(torch.normal(torch.zeros(param.size()), std=1.0).cuda()))

            loss = loss_likelihood + loss_prior + loss_noise

            loss_value = loss_likelihood.data.cpu().numpy()
            batch_losses.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
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

        print ("lr: %g" % lr)

        # save the model weights to disk:
        checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(network.state_dict(), checkpoint_path)
