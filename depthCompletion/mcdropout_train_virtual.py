# code-checked
# server-checked

import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from model_mcdropout import DepthCompletionNet

from datasets import DatasetVirtualKITTIAugmentation, DatasetVirtualKITTIVal
from criterion import MaskedL2Gauss, RMSE

import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_id = "mcdropout_virtual"

snapshot_dir_base = "/root/evaluating_bdl/depthCompletion/training_logs/%s" % model_id

virtualkitti_path = "/root/data/virtualkitti"

batch_size = 4
weight_decay = 0.0005
num_steps = 40000

val_batch_size = 4

save_pred_every = 1000

M = 16
for i in range(M):
    learning_rate = 1.0e-5

    snapshot_dir = snapshot_dir_base + "_%d/" % i
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    train_dataset = DatasetVirtualKITTIAugmentation(virtualkitti_path=virtualkitti_path, max_iters=num_steps*batch_size, crop_size=(352, 352))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = DatasetVirtualKITTIVal(virtualkitti_path=virtualkitti_path)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

    criterion = MaskedL2Gauss().cuda()
    rmse_criterion = RMSE().cuda()

    model = DepthCompletionNet().cuda()
    model = torch.nn.DataParallel(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad()

    train_losses = []
    batch_train_losses = []
    val_losses = []
    train_rmses = []
    batch_train_rmses = []
    val_rmses = []
    for i_iter, batch in enumerate(train_loader):
        imgs, sparses, targets, file_ids = batch
        imgs = Variable(imgs.cuda()) # (shape: (batch_size, h, w))
        sparses = Variable(sparses.cuda()) # (shape: (batch_size, h, w))
        targets = Variable(targets.cuda()) # (shape: (batch_size, h, w))

        means, log_vars = model(imgs, sparses) # (both of shape: (batch_size, 1, h, w))

        loss = criterion(means, log_vars, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rmse = rmse_criterion(means, targets)

        print ("%d/%d, loss: %g, RMSE: %g" % (i_iter, num_steps, loss.data.cpu().numpy(), rmse.data.cpu().numpy()))

        batch_train_losses.append(loss.data.cpu().numpy())
        batch_train_rmses.append(rmse.data.cpu().numpy())

        if (i_iter % 500 == 0) and (i_iter > 0):
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

            train_rmse = np.mean(batch_train_rmses)
            train_rmses.append(train_rmse)
            with open("%strain_rmses.pkl" % snapshot_dir, "wb") as file:
                pickle.dump(train_rmses, file)
            plt.figure(1)
            plt.plot(train_rmses, "k^")
            plt.plot(train_rmses, "k")
            plt.ylabel("rmse")
            plt.xlabel("epoch")
            plt.title("train rmses")
            plt.savefig("%strain_rmses.png" % snapshot_dir)
            plt.close(1)
            batch_train_rmses = []

        ########################################################################
        # evaluate on the val set:
        ########################################################################
        if (i_iter % 1000 == 0) and (i_iter > 0):
            model.eval()

            batch_val_losses = []
            batch_val_rmses = []
            for i_iter_val, batch in enumerate(val_loader):
                with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                    imgs, sparses, targets, file_ids = batch
                    imgs = Variable(imgs.cuda()) # (shape: (batch_size, h, w))
                    sparses = Variable(sparses.cuda()) # (shape: (batch_size, h, w))
                    targets = Variable(targets.cuda()) # (shape: (batch_size, h, w))

                    means, log_vars = model(imgs, sparses) # (both of shape: (batch_size, 1, h, w))

                    loss = criterion(means, log_vars, targets)
                    rmse = rmse_criterion(means, targets)

                    print ("     (%d/%d) val: %d/%d, loss: %g, RMSE: %g" % (i_iter, num_steps, i_iter_val, len(val_dataset)/val_batch_size, loss.data.cpu().numpy(), rmse.data.cpu().numpy()))

                    batch_val_losses.append(loss.data.cpu().numpy())
                    batch_val_rmses.append(rmse.data.cpu().numpy())

            val_loss = np.mean(batch_val_losses)
            print ("val_loss: %g" % val_loss)
            val_losses.append(val_loss)
            with open("%sval_losses.pkl" % snapshot_dir, "wb") as file:
                pickle.dump(val_losses, file)
            plt.figure(1)
            plt.plot(val_losses, "k^")
            plt.plot(val_losses, "k")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.title("val losses")
            plt.savefig("%sval_losses.png" % snapshot_dir)
            plt.close(1)

            val_rmse = np.mean(batch_val_rmses)
            print ("val_rmse: %g" % val_rmse)
            val_rmses.append(val_rmse)
            with open("%sval_rmses.pkl" % snapshot_dir, "wb") as file:
                pickle.dump(val_rmses, file)
            plt.figure(1)
            plt.plot(val_rmses, "k^")
            plt.plot(val_rmses, "k")
            plt.ylabel("rmse")
            plt.xlabel("epoch")
            plt.title("val rmses")
            plt.savefig("%sval_rmses.png" % snapshot_dir)
            plt.close(1)

            model.train()
        ########################################################################

        if i_iter >= num_steps-1:
            print("saving model...")
            torch.save(model.state_dict(), snapshot_dir + "checkpoint_" + str(num_steps) + ".pth")
            break

        if i_iter % save_pred_every == 0:
            print("taking snapshot...")
            torch.save(model.state_dict(), snapshot_dir + "checkpoint_" + str(i_iter) + ".pth")
