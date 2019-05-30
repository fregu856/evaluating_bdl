# code-checked
# server-checked

import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from model_mcdropout import DepthCompletionNet

from datasets import DatasetKITTIVal
from criterion import MaskedL2Gauss, RMSE

import numpy as np
import cv2
import random
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_id = "mcdropout_virtual"

model_is = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
print (len(model_is))

snapshot_dir = "/root/evaluating_bdl/depthCompletion/training_logs/%s_eval_ause" % model_id

kitti_depth_path = "/root/data/kitti_depth"

batch_size = 4

if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

colors = {}
colors[1] = "k"
colors[2] = "b"
colors[4] = "g"
colors[8] = "r"
colors[16] = "c"
colors[32] = "m"
colors[64] = "y"

run_colors = {}
run_colors[0] = "k"
run_colors[1] = "#E53935"
run_colors[2] = "#8E24AA"
run_colors[3] = "#3949AB"
run_colors[4] = "#1E88E5"
run_colors[5] = "#00ACC1"
run_colors[6] = "#00897B"
run_colors[7] = "#7CB342"
run_colors[8] = "#FDD835"
run_colors[9] = "#FB8C00"
run_colors[10] = "#D81B60"
run_colors[11] = "#5E35B1"
run_colors[12] = "#039BE5"
run_colors[13] = "#43A047"
run_colors[14] = "#C0CA33"
run_colors[15] = "#FFB300"
run_colors[16] = "#F4511E"
run_colors[17] = "#6D4C41"
run_colors[18] = "#546E7A"
run_colors[19] = "#827717"

num_model_is = len(model_is)
print (num_model_is)

M_values = [1, 2, 4, 8, 16, 32]
print (M_values)

# # # # # # # # # # # # # # # # # # debug START:
# M_values = [1, 2, 4]
# model_is = [0, 1]
# # # # # # # # # # # # # # # # # # debug END:

num_runs_per_M = 1

sparsification_error_values = {}
error_rmse_values = {}
sigma_rmse_values = {}
for model_i in model_is:
    sparsification_error_values[model_i] = {}
    error_rmse_values[model_i] = {}
    sigma_rmse_values[model_i] = {}

    for M in M_values:
        sparsification_error_values[model_i][M] = {}
        error_rmse_values[model_i][M] = {}
        sigma_rmse_values[model_i][M] = {}

auc_sparsification_error_values = {}
loss_values = {}
rmse_values = {}
for M in M_values:
    auc_sparsification_error_values[M] = []

    loss_values[M] = []
    rmse_values[M] = []

eval_dataset = DatasetKITTIVal(kitti_depth_path=kitti_depth_path)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

criterion = MaskedL2Gauss().cuda()
rmse_criterion = RMSE().cuda()

for model_i in model_is:
    print ("model_i: %d" % model_i)

    restore_from = "/root/evaluating_bdl/depthCompletion/trained_models/%s_%d/checkpoint_40000.pth" % (model_id, model_i)
    model = DepthCompletionNet().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(restore_from))
    model.eval()

    for M in M_values:
        M_float = float(M)
        print ("M: %d" % M)

        for run in range(num_runs_per_M):
            print ("run: %d" % run)

            batch_losses = []
            batch_rmses = []
            sigma_alea_values = np.array([])
            sigma_epi_values = np.array([])
            sigma_pred_values = np.array([])
            squared_error_values = np.array([])
            for i_iter, batch in enumerate(eval_loader):
                with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                    imgs, sparses, targets, file_ids = batch
                    imgs = Variable(imgs.cuda()) # (shape: (batch_size, h, w))
                    sparses = Variable(sparses.cuda()) # (shape: (batch_size, h, w))
                    targets = Variable(targets.cuda()) # (shape: (batch_size, h, w))

                    means = []
                    sigma_2_aleas = []
                    for i in range(M):
                        mean, log_var = model(imgs, sparses) # (both of shape: (batch_size, 1, h, w))

                        sigma_2_alea = torch.exp(log_var) # (sigma_alea^2) # (shape: (batch_size, 1, h, w))

                        means.append(mean)
                        sigma_2_aleas.append(sigma_2_alea)

                    mean = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w))
                    for value in means:
                        mean = mean + value/M_float

                    sigma_2_alea = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w)) (sigma_alea^2)
                    for value in sigma_2_aleas:
                        sigma_2_alea = sigma_2_alea + value/M_float

                    sigma_2_epi = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w)) (sigma_epi^2)
                    for value in means:
                        sigma_2_epi = sigma_2_epi + torch.pow(mean - value, 2)/M_float

                    sigma_2_pred = sigma_2_alea + sigma_2_epi # (sigma_pred^2)

                    loss = criterion(mean, torch.log(sigma_2_pred), targets)
                    rmse = rmse_criterion(mean, targets)

                    print('iter = {}/{} completed, loss = {}, rmse = {}'.format(i_iter, len(eval_dataset)/batch_size, loss.data.cpu().numpy(), rmse.data.cpu().numpy()))

                    batch_losses.append(loss.data.cpu().numpy())
                    batch_rmses.append(rmse.data.cpu().numpy())

                    sigma_alea = torch.sqrt(sigma_2_alea) # (shape: (batch_size, 1, h, w))
                    sigma_epi = torch.sqrt(sigma_2_epi) # (shape: (batch_size, 1, h, w))
                    sigma_pred = torch.sqrt(sigma_2_pred) # (shape: (batch_size, 1, h, w))

                    target = torch.unsqueeze(targets, 1) # (shape: (batch_size, 1, h, w))

                    valid_mask = (target > 0).detach() # (shape: (batch_size, 1, h, w))

                    mean = mean[valid_mask] # (shape: (num_valids, ))
                    sigma_alea = sigma_alea[valid_mask] # (shape: (num_valids, ))
                    sigma_epi = sigma_epi[valid_mask] # (shape: (num_valids, ))
                    sigma_pred = sigma_pred[valid_mask] # (shape: (num_valids, ))
                    target = target[valid_mask] # (shape: (num_valids, ))

                    squared_error = torch.pow(target - mean, 2) # (shape: (num_valids, ))

                    sigma_alea_values = np.concatenate((sigma_alea_values, sigma_alea.data.cpu().numpy()))
                    sigma_epi_values = np.concatenate((sigma_epi_values, sigma_epi.data.cpu().numpy()))
                    sigma_pred_values = np.concatenate((sigma_pred_values, sigma_pred.data.cpu().numpy()))
                    squared_error_values = np.concatenate((squared_error_values, squared_error.data.cpu().numpy()))

                    # # # # # # # # # # # # # # # # # # debug START:
                    # if i_iter > 0:
                    #     break
                    # # # # # # # # # # # # # # # # # # debug END:

            val_loss = np.mean(batch_losses)
            print ("val loss: %g" % val_loss)
            val_rmse = np.mean(batch_rmses)
            print ("val rmse: %g" % val_rmse)
            loss_values[M].append(val_loss)
            rmse_values[M].append(val_rmse)

            # (sigma_alea/epi/pred_values has shape: (num_predictions_with_GT, ))
            # (squared_error_values has shape: (num_predictions_with_GT, ))

            print (sigma_alea_values.shape)
            print (sigma_epi_values.shape)
            print (sigma_pred_values.shape)
            print (squared_error_values.shape)

            num_predictions_with_GT = squared_error_values.shape[0]

            rmse = np.sqrt(np.mean(squared_error_values))
            print (rmse)

            #sorted_inds_sigma_alea = np.argsort(sigma_alea_values) # (sigma_values[sorted_inds_sigma[0]]: SMALLEST element of sigma_values)
            #sorted_inds_sigma_epi = np.argsort(sigma_epi_values) # (sigma_values[sorted_inds_sigma[0]]: SMALLEST element of sigma_values)
            sorted_inds_sigma_pred = np.argsort(sigma_pred_values) # (sigma_values[sorted_inds_sigma[0]]: SMALLEST element of sigma_values)
            sorted_inds_error = np.argsort(squared_error_values)

            sigma_alea_rmses = []
            sigma_epi_rmses = []
            sigma_pred_rmses = []
            error_rmses = []
            fractions = list(np.arange(start=0.0, stop=1.0, step=0.01)) # ([0.0, 0.01, ..., 0.99], 100 elements)
            for step, fraction in enumerate(fractions):
                #print ("fraction: %d/%d" % (step+1, len(fractions)))

                #sigma_alea_rmse = np.sqrt(np.mean( squared_error_values[sorted_inds_sigma_alea[0:int((1.0-fraction)*num_predictions_with_GT)]] ))
                #sigma_alea_rmses.append(sigma_alea_rmse)

                #sigma_epi_rmse = np.sqrt(np.mean( squared_error_values[sorted_inds_sigma_epi[0:int((1.0-fraction)*num_predictions_with_GT)]] ))
                #sigma_epi_rmses.append(sigma_epi_rmse)

                sigma_pred_rmse = np.sqrt(np.mean( squared_error_values[sorted_inds_sigma_pred[0:int((1.0-fraction)*num_predictions_with_GT)]] ))
                sigma_pred_rmses.append(sigma_pred_rmse)

                error_rmse = np.sqrt(np.mean( squared_error_values[sorted_inds_error[0:int((1.0-fraction)*num_predictions_with_GT)]] ))
                error_rmses.append(error_rmse)

            error_rmses_normalized = error_rmses/error_rmses[0]
            #sigma_alea_rmses_normalized = sigma_alea_rmses/sigma_alea_rmses[0]
            #sigma_epi_rmses_normalized = sigma_epi_rmses/sigma_epi_rmses[0]
            sigma_pred_rmses_normalized = sigma_pred_rmses/sigma_pred_rmses[0]

            #sparsification_errors_alea = sigma_alea_rmses_normalized - error_rmses_normalized
            #sparsification_errors_epi = sigma_epi_rmses_normalized - error_rmses_normalized
            sparsification_errors_pred = sigma_pred_rmses_normalized - error_rmses_normalized

            #ause_alea = np.trapz(y=sparsification_errors_alea, x=fractions)
            #print ("Area Under the Sparsification Error curve (AUSE) - Alea: %g" % ause_alea)

            #ause_epi = np.trapz(y=sparsification_errors_epi, x=fractions)
            #print ("Area Under the Sparsification Error curve (AUSE) - Epi: %g" % ause_epi)

            ause_pred = np.trapz(y=sparsification_errors_pred, x=fractions)
            print ("Area Under the Sparsification Error curve (AUSE) - Pred: %g" % ause_pred)

            sparsification_error_values[model_i][M][run] = np.array(sparsification_errors_pred)
            error_rmse_values[model_i][M][run] = np.array(error_rmses_normalized)
            sigma_rmse_values[model_i][M][run] = np.array(sigma_pred_rmses_normalized)

            auc_sparsification_error_values[M].append(ause_pred)

            print ("#######################")

        print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

auc_sparsification_error_means = {}
auc_sparsification_error_stds = {}
loss_means = {}
loss_stds = {}
rmse_means = {}
rmse_stds = {}
for M in M_values:
    auc_sparsification_error_values_mean = 0.0
    for value in auc_sparsification_error_values[M]:
        auc_sparsification_error_values_mean += value/float(num_runs_per_M*num_model_is)

    auc_sparsification_error_values_var = 0.0
    for value in auc_sparsification_error_values[M]:
        auc_sparsification_error_values_var += ((value - auc_sparsification_error_values_mean)**2)/float(num_runs_per_M*num_model_is)

    auc_sparsification_error_values_std = np.sqrt(auc_sparsification_error_values_var)

    auc_sparsification_error_means[M] = auc_sparsification_error_values_mean
    auc_sparsification_error_stds[M] = auc_sparsification_error_values_std

    ###

    loss_values_mean = 0.0
    for value in loss_values[M]:
        loss_values_mean += value/float(num_runs_per_M*num_model_is)

    loss_values_var = 0.0
    for value in loss_values[M]:
        loss_values_var += ((value - loss_values_mean)**2)/float(num_runs_per_M*num_model_is)

    loss_values_std = np.sqrt(loss_values_var)

    loss_means[M] = loss_values_mean
    loss_stds[M] = loss_values_std

    ###

    rmse_values_mean = 0.0
    for value in rmse_values[M]:
        rmse_values_mean += value/float(num_runs_per_M*num_model_is)

    rmse_values_var = 0.0
    for value in rmse_values[M]:
        rmse_values_var += ((value - rmse_values_mean)**2)/float(num_runs_per_M*num_model_is)

    rmse_values_std = np.sqrt(rmse_values_var)

    rmse_means[M] = rmse_values_mean
    rmse_stds[M] = rmse_values_std

for M in M_values:
    print ("M = %d, Sparsification error (AUC) - mean: %g, std: %g" % (M, auc_sparsification_error_means[M], auc_sparsification_error_stds[M]))
    print ("M = %d, Loss - mean: %g, std: %g" % (M, loss_means[M], loss_stds[M]))
    print ("M = %d, RMSE - mean: %g, std: %g" % (M, rmse_means[M], rmse_stds[M]))
    print ("#####")

plt.figure(1)
for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(fractions, sparsification_error_values[model_i][M][run], color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(fractions, sparsification_error_values[model_i][M][run], color=colors[M], alpha=0.5)
plt.legend()
plt.ylabel("Sparsification error")
plt.xlabel("Fraction of removed pixels")
sparsification_error_ylim = plt.ylim()
plt.title("Sparsification error curve")
plt.savefig("%s/sparsification_error_curve.png" % snapshot_dir)
plt.close(1)

for M in M_values:
    plt.figure(1)
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(fractions, sparsification_error_values[model_i][M][run], color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(fractions, sparsification_error_values[model_i][M][run], color=colors[M], alpha=0.5)
    plt.legend()
    plt.ylabel("Sparsification error")
    plt.xlabel("Fraction of removed pixels")
    plt.ylim(sparsification_error_ylim)
    plt.title("Sparsification error curve")
    plt.savefig("%s/sparsification_error_curve_M%d.png" % (snapshot_dir, M))
    plt.close(1)

    plt.figure(1)
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            plt.plot(fractions, error_rmse_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run], linestyle="dotted")
            plt.plot(fractions,sigma_rmse_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run])
    plt.ylabel("RMSE (normalized)")
    plt.xlabel("Fraction of removed pixels")
    plt.ylim((-0.05, 1.05))
    plt.title("Sparsification plot - M=%d" % M)
    plt.savefig("%s/sparsification_plot_M%d.png" % (snapshot_dir, M))
    plt.close(1)

for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            plt.figure(1)
            plt.plot(fractions, error_rmse_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run], linestyle="dotted", label="Oracle")
            plt.plot(fractions, sigma_rmse_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run], label="Model")
            plt.legend()
            plt.ylabel("RMSE (normalized)")
            plt.xlabel("Fraction of removed pixels")
            plt.ylim((-0.05, 1.05))
            plt.title("Sparsification plot - M=%d, model_i=%d, %d" % (M, model_i, run))
            plt.savefig("%s/sparsification_plot_M%d_model_i%d_%d.png" % (snapshot_dir, M, model_i, run))
            plt.close(1)

with open("%s/auc_sparsification_error_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(auc_sparsification_error_values, file)

with open("%s/loss_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(loss_values, file)

with open("%s/rmse_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(rmse_values, file)

with open("%s/sparsification_error_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(sparsification_error_values, file)

with open("%s/error_rmse_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(error_rmse_values, file)

with open("%s/sigma_rmse_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(sigma_rmse_values, file)
