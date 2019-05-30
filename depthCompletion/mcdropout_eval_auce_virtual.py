# code-checked
# server-checked

import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from model_mcdropout import DepthCompletionNet

from datasets import DatasetVirtualKITTIVal
from criterion import MaskedL2Gauss, RMSE

import numpy as np
import cv2
import pickle

import scipy.stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import random

model_id = "mcdropout_virtual"

model_is = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
print (len(model_is))

snapshot_dir = "/root/evaluating_bdl/depthCompletion/training_logs/%s_eval_auce_virtual" % model_id

virtualkitti_path = "/root/data/virtualkitti"

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

num_model_is = len(model_is)
print (num_model_is)

M_values = [1, 2, 4, 8, 16, 32]
print (M_values)

# # # # # # # # # # # # # # # # # # debug START:
# M_values = [1, 2, 4]
# model_is = [0, 1]
# # # # # # # # # # # # # # # # # # debug END:

num_runs_per_M = 1

coverage_values = {}
avg_length_values = {}
coverage_error_values = {}
abs_coverage_error_values = {}
neg_coverage_error_values = {}
for model_i in model_is:
    coverage_values[model_i] = {}
    avg_length_values[model_i] = {}
    coverage_error_values[model_i] = {}
    abs_coverage_error_values[model_i] = {}
    neg_coverage_error_values[model_i] = {}

    for M in M_values:
        coverage_values[model_i][M] = {}
        avg_length_values[model_i][M] = {}
        coverage_error_values[model_i][M] = {}
        abs_coverage_error_values[model_i][M] = {}
        neg_coverage_error_values[model_i][M] = {}

auc_abs_error_values = {}
auc_neg_error_values = {}
auc_length_values = {}
loss_values = {}
rmse_values = {}
for M in M_values:
    auc_abs_error_values[M] = []
    auc_neg_error_values[M] = []
    auc_length_values[M] = []

    loss_values[M] = []
    rmse_values[M] = []

eval_dataset = DatasetVirtualKITTIVal(virtualkitti_path=virtualkitti_path)
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
            mean_values = np.array([])
            target_values = np.array([])
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

                    sigma_alea_values = np.concatenate((sigma_alea_values, sigma_alea.data.cpu().numpy()))
                    sigma_epi_values = np.concatenate((sigma_epi_values, sigma_epi.data.cpu().numpy()))
                    sigma_pred_values = np.concatenate((sigma_pred_values, sigma_pred.data.cpu().numpy()))
                    mean_values = np.concatenate((mean_values, mean.data.cpu().numpy()))
                    target_values = np.concatenate((target_values, target.data.cpu().numpy()))

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
            # (mean_values has shape: (num_predictions_with_GT, ))
            # (target_values has shape: (num_predictions_with_GT, ))

            print (sigma_alea_values.shape)
            print (sigma_epi_values.shape)
            print (sigma_pred_values.shape)
            print (mean_values.shape)
            print (target_values.shape)

            num_predictions_with_GT = float(target_values.shape[0])

            coverage_values_alea = []
            coverage_values_epi = []
            coverage_values_pred = []
            avg_length_values_alea = []
            avg_length_values_epi = []
            avg_length_values_pred = []
            alphas = list(np.arange(start=0.01, stop=1.0, step=0.01)) # ([0.01, 0.02, ..., 0.99], 99 elements)
            for step, alpha in enumerate(alphas):
                #print ("alpha: %d/%d" % (step+1, len(alphas)))

                lower_values_alea = mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_alea_values # (shape: (num_predictions_with_GT, ))
                upper_values_alea = mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_alea_values # (shape: (num_predictions_with_GT, ))

                coverage_alea = np.count_nonzero(np.logical_and(target_values >= lower_values_alea, target_values <= upper_values_alea))/num_predictions_with_GT
                coverage_values_alea.append(coverage_alea)

                avg_length_alea = np.mean(upper_values_alea - lower_values_alea)
                avg_length_values_alea.append(avg_length_alea)
                #
                lower_values_epi = mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_epi_values # (shape: (num_predictions_with_GT, ))
                upper_values_epi = mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_epi_values # (shape: (num_predictions_with_GT, ))

                coverage_epi = np.count_nonzero(np.logical_and(target_values >= lower_values_epi, target_values <= upper_values_epi))/num_predictions_with_GT
                coverage_values_epi.append(coverage_epi)

                avg_length_epi = np.mean(upper_values_epi - lower_values_epi)
                avg_length_values_epi.append(avg_length_epi)
                #
                lower_values_pred = mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_pred_values # (shape: (num_predictions_with_GT, ))
                upper_values_pred = mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_pred_values # (shape: (num_predictions_with_GT, ))

                coverage_pred = np.count_nonzero(np.logical_and(target_values >= lower_values_pred, target_values <= upper_values_pred))/num_predictions_with_GT
                coverage_values_pred.append(coverage_pred)

                avg_length_pred = np.mean(upper_values_pred - lower_values_pred)
                avg_length_values_pred.append(avg_length_pred)

            auc_length_alea = np.trapz(y=avg_length_values_alea, x=alphas)
            print ("AUC - Length - Alea: %g" % auc_length_alea)
            auc_length_epi = np.trapz(y=avg_length_values_epi, x=alphas)
            print ("AUC - Length - Epi: %g" % auc_length_epi)
            auc_length_pred = np.trapz(y=avg_length_values_pred, x=alphas)
            print ("AUC - Length - Pred: %g" % auc_length_pred)

            coverage_error_values_alea =  np.array(coverage_values_alea) - (1.0 - np.array(alphas))
            coverage_error_values_epi =  np.array(coverage_values_epi) - (1.0 - np.array(alphas))
            coverage_error_values_pred =  np.array(coverage_values_pred) - (1.0 - np.array(alphas))

            abs_coverage_error_values_alea = np.abs(coverage_error_values_alea)
            abs_coverage_error_values_epi = np.abs(coverage_error_values_epi)
            abs_coverage_error_values_pred = np.abs(coverage_error_values_pred)

            neg_coverage_error_values_alea = (np.abs(coverage_error_values_alea) - coverage_error_values_alea)/2.0
            neg_coverage_error_values_epi = (np.abs(coverage_error_values_epi) - coverage_error_values_epi)/2.0
            neg_coverage_error_values_pred = (np.abs(coverage_error_values_pred) - coverage_error_values_pred)/2.0

            auc_error_alea = np.trapz(y=abs_coverage_error_values_alea, x=alphas)
            print ("AUC - Empirical coverage absolute error - Alea: %g" % auc_error_alea)
            auc_error_epi = np.trapz(y=abs_coverage_error_values_epi, x=alphas)
            print ("AUC - Empirical coverage absolute error - Epi: %g" % auc_error_epi)
            auc_error_pred = np.trapz(y=abs_coverage_error_values_pred, x=alphas)
            print ("AUC - Empirical coverage absolute error - Pred: %g" % auc_error_pred)

            auc_neg_error_alea = np.trapz(y=neg_coverage_error_values_alea, x=alphas)
            print ("AUC - Empirical coverage negative error - Alea: %g" % auc_neg_error_alea)
            auc_neg_error_epi = np.trapz(y=neg_coverage_error_values_epi, x=alphas)
            print ("AUC - Empirical coverage negative error - Epi: %g" % auc_neg_error_epi)
            auc_neg_error_pred = np.trapz(y=neg_coverage_error_values_pred, x=alphas)
            print ("AUC - Empirical coverage negative error - Pred: %g" % auc_neg_error_pred)

            coverage_values[model_i][M][run] = np.array(coverage_values_pred)
            avg_length_values[model_i][M][run] = np.array(avg_length_values_pred)
            coverage_error_values[model_i][M][run] = np.array(coverage_error_values_pred)
            abs_coverage_error_values[model_i][M][run] = abs_coverage_error_values_pred
            neg_coverage_error_values[model_i][M][run] = neg_coverage_error_values_pred

            auc_abs_error_values[M].append(auc_error_pred)
            auc_length_values[M].append(auc_length_pred)
            auc_neg_error_values[M].append(auc_neg_error_pred)

            print ("#######################")

        print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

auc_abs_error_means = {}
auc_abs_error_stds = {}
auc_neg_error_means = {}
auc_neg_error_stds = {}
auc_length_means = {}
auc_length_stds = {}
loss_means = {}
loss_stds = {}
rmse_means = {}
rmse_stds = {}
for M in M_values:
    auc_abs_error_values_mean = 0.0
    for value in auc_abs_error_values[M]:
        auc_abs_error_values_mean += value/float(num_runs_per_M*num_model_is)

    auc_abs_error_values_var = 0.0
    for value in auc_abs_error_values[M]:
        auc_abs_error_values_var += ((value - auc_abs_error_values_mean)**2)/float(num_runs_per_M*num_model_is)

    auc_abs_error_values_std = np.sqrt(auc_abs_error_values_var)

    auc_abs_error_means[M] = auc_abs_error_values_mean
    auc_abs_error_stds[M] = auc_abs_error_values_std

    ###

    auc_neg_error_values_mean = 0.0
    for value in auc_neg_error_values[M]:
        auc_neg_error_values_mean += value/float(num_runs_per_M*num_model_is)

    auc_neg_error_values_var = 0.0
    for value in auc_neg_error_values[M]:
        auc_neg_error_values_var += ((value - auc_neg_error_values_mean)**2)/float(num_runs_per_M*num_model_is)

    auc_neg_error_values_std = np.sqrt(auc_neg_error_values_var)

    auc_neg_error_means[M] = auc_neg_error_values_mean
    auc_neg_error_stds[M] = auc_neg_error_values_std

    ###

    auc_length_values_mean = 0.0
    for value in auc_length_values[M]:
        auc_length_values_mean += value/float(num_runs_per_M*num_model_is)

    auc_length_values_var = 0.0
    for value in auc_length_values[M]:
        auc_length_values_var += ((value - auc_length_values_mean)**2)/float(num_runs_per_M*num_model_is)

    auc_length_values_std = np.sqrt(auc_length_values_var)

    auc_length_means[M] = auc_length_values_mean
    auc_length_stds[M] = auc_length_values_std

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
    print ("M = %d, Empirical coverage absolute error (AUC) - mean: %g, std: %g" % (M, auc_abs_error_means[M], auc_abs_error_stds[M]))
    print ("M = %d, Empirical coverage negative error (AUC) - mean: %g, std: %g" % (M, auc_neg_error_means[M], auc_neg_error_stds[M]))
    print ("M = %d, Average length (AUC) - mean: %g, std: %g" % (M, auc_length_means[M], auc_length_stds[M]))
    print ("M = %d, Loss - mean: %g, std: %g" % (M, loss_means[M], loss_stds[M]))
    print ("M = %d, RMSE - mean: %g, std: %g" % (M, rmse_means[M], rmse_stds[M]))
    print ("#####")

plt.figure(1)
plt.plot([0.0, 1.0], [0.0, 1.0], "k:", label="Perfect")
for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(coverage_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(coverage_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
plt.legend()
plt.ylabel("Empirical coverage")
plt.xlabel("p")
plt.title("Prediction intervals - Empirical coverage")
plt.savefig("%s/empirical_coverage.png" % snapshot_dir)
plt.close(1)

plt.figure(1)
for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(avg_length_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(avg_length_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
plt.legend()
plt.ylabel("Average interval length [m]")
plt.xlabel("p")
avg_length_ylim = plt.ylim()
plt.title("Prediction intervals - Average interval length")
plt.savefig("%s/length.png" % snapshot_dir)
plt.close(1)

plt.figure(1)
plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
plt.legend()
plt.ylabel("Empirical coverage error")
plt.xlabel("p")
coverage_error_ylim = plt.ylim()
plt.title("Prediction intervals - Empirical coverage error")
plt.savefig("%s/empirical_coverage_error.png" % snapshot_dir)
plt.close(1)

plt.figure(1)
plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(abs_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(abs_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
plt.legend()
plt.ylabel("Empirical coverage absolute error")
plt.xlabel("p")
abs_coverage_error_ylim = plt.ylim()
plt.title("Prediction intervals - Empirical coverage absolute error")
plt.savefig("%s/empirical_coverage_absolute_error.png" % snapshot_dir)
plt.close(1)

plt.figure(1)
plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(neg_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(neg_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
plt.legend()
plt.ylabel("Empirical coverage negative error")
plt.xlabel("p")
neg_coverage_error_ylim = plt.ylim()
plt.title("Prediction intervals - Empirical coverage negative error")
plt.savefig("%s/empirical_coverage_negative_error.png" % snapshot_dir)
plt.close(1)

for M in M_values:
    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 1.0], "k:", label="Perfect")
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(coverage_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(coverage_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
    plt.legend()
    plt.ylabel("Empirical coverage")
    plt.xlabel("p")
    plt.title("Prediction intervals - Empirical coverage")
    plt.savefig("%s/empirical_coverage_M%d.png" % (snapshot_dir, M))
    plt.close(1)

    plt.figure(1)
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(avg_length_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(avg_length_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
    plt.legend()
    plt.ylabel("Average interval length [m]")
    plt.xlabel("p")
    plt.ylim(avg_length_ylim)
    plt.title("Prediction intervals - Average interval length")
    plt.savefig("%s/length_M%d.png" % (snapshot_dir, M))
    plt.close(1)

    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
    plt.legend()
    plt.ylabel("Empirical coverage error")
    plt.xlabel("p")
    plt.ylim(coverage_error_ylim)
    plt.title("Prediction intervals - Empirical coverage error")
    plt.savefig("%s/empirical_coverage_error_M%d.png" % (snapshot_dir, M))
    plt.close(1)

    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(abs_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(abs_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
    plt.legend()
    plt.ylabel("Empirical coverage absolute error")
    plt.xlabel("p")
    plt.ylim(abs_coverage_error_ylim)
    plt.title("Prediction intervals - Empirical coverage absolute error")
    plt.savefig("%s/empirical_coverage_absolute_error_M%d.png" % (snapshot_dir, M))
    plt.close(1)

    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 0.0], "k:", label="Perfect")
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            if (model_i_step == 0) and (run == 0):
                plt.plot(alphas, np.flip(neg_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5, label="M = %d" % M)
            else:
                plt.plot(alphas, np.flip(neg_coverage_error_values[model_i][M][run], 0), color=colors[M], alpha=0.5)
    plt.legend()
    plt.ylabel("Empirical coverage negative error")
    plt.xlabel("p")
    plt.ylim(neg_coverage_error_ylim)
    plt.title("Prediction intervals - Empirical coverage negative error")
    plt.savefig("%s/empirical_coverage_negative_error_M%d.png" % (snapshot_dir, M))
    plt.close(1)

with open("%s/auc_abs_error_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(auc_abs_error_values, file)

with open("%s/auc_neg_error_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(auc_neg_error_values, file)

with open("%s/auc_length_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(auc_length_values, file)

with open("%s/loss_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(loss_values, file)

with open("%s/rmse_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(rmse_values, file)

with open("%s/coverage_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(coverage_values, file)

with open("%s/avg_length_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(avg_length_values, file)

with open("%s/coverage_error_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(coverage_error_values, file)

with open("%s/abs_coverage_error_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(abs_coverage_error_values, file)

with open("%s/neg_coverage_error_values.pkl" % snapshot_dir, "wb") as file:
    pickle.dump(neg_coverage_error_values, file)
