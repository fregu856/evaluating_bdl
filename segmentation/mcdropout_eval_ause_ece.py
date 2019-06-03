# code-checked
# server-checked

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data

import os
import numpy as np
import cv2
import random
import pickle

from datasets import DatasetCityscapesEval
from models.model_mcdropout import get_model

from utils.utils import get_confusion_matrix

model_id = "mcdropout_syn"
model_is = [0, 1, 2, 3, 4, 5, 6, 7]

num_model_is = len(model_is)
print (num_model_is)

output_path = "/home/evaluating_bdl/segmentation/training_logs/%s_eval_ause_ece" % model_id
if not os.path.exists(output_path):
    os.makedirs(output_path)

num_conf_intervals = 10 # (most papers seem to use 10, but maybe I should use 15? 20?)
conf_interval_size = 1.0/num_conf_intervals

data_dir = "/home/data/cityscapes"
data_list = "/home/evaluating_bdl/segmentation/lists/cityscapes/val.lst"
batch_size = 2
num_classes = 19

eval_dataset = DatasetCityscapesEval(root=data_dir, list_path=data_list)
eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

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
run_colors[20] = "#D81B60"
run_colors[21] = "#5E35B1"
run_colors[22] = "#039BE5"
run_colors[23] = "#43A047"
run_colors[24] = "#C0CA33"
run_colors[25] = "#FFB300"
run_colors[26] = "#F4511E"
run_colors[27] = "#6D4C41"
run_colors[28] = "#546E7A"
run_colors[29] = "#827717"
run_colors[30] = "#D81B60"
run_colors[31] = "#5E35B1"
run_colors[32] = "#039BE5"
run_colors[33] = "#43A047"
run_colors[34] = "#C0CA33"
run_colors[35] = "#FFB300"
run_colors[36] = "#F4511E"
run_colors[37] = "#6D4C41"
run_colors[38] = "#546E7A"
run_colors[39] = "#827717"

M_values = [1, 2, 4, 8, 16]
print (M_values)

# # # # # # # # # # # # # # # # # # debug START:
# M_values = [1, 2, 4]
# model_is = [0, 1]
# num_model_is = len(model_is)
# # # # # # # # # # # # # # # # # # debug END:

num_runs_per_M = 1

sparsification_error_values = {}
error_brier_score_values = {}
entropy_brier_score_values = {}
rel_diagrams = {}
for model_i in model_is:
    sparsification_error_values[model_i] = {}
    error_brier_score_values[model_i] = {}
    entropy_brier_score_values[model_i] = {}
    rel_diagrams[model_i] = {}

    for M in M_values:
        sparsification_error_values[model_i][M] = {}
        error_brier_score_values[model_i][M] = {}
        entropy_brier_score_values[model_i][M] = {}
        rel_diagrams[model_i][M] = {}

auc_sparsification_error_values = {}
mIoU_values = {}
ECE_values = {}
for M in M_values:
    auc_sparsification_error_values[M] = []
    mIoU_values[M] = []
    ECE_values[M] = []

for model_i in model_is:
    print ("model_i: %d" % model_i)

    restore_from = "/home/evaluating_bdl/segmentation/trained_models/%s_%d/checkpoint_60000.pth" % (model_id, model_i)
    deeplab = get_model(num_classes=num_classes)
    deeplab.load_state_dict(torch.load(restore_from))
    model = nn.DataParallel(deeplab)
    model.eval()
    model.cuda()

    for M in M_values:
        M_float = float(M)
        print ("M: %d" % M)

        for run in range(num_runs_per_M):
            print ("run: %d" % run)

            interval_2_num_preds = {}
            interval_2_num_correct_preds = {}
            interval_2_confs = {}
            interval_2_mean_conf = {}
            for i in range(num_conf_intervals):
                interval_2_num_preds[i] = 0
                interval_2_num_correct_preds[i] = 0
                interval_2_confs[i] = np.array([])

            confusion_matrix = np.zeros((num_classes, num_classes))
            entropy_values = np.array([])
            squared_error_values = np.array([])
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
                    for i in range(M):
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

                    # (label has shape: (batch_size, h, w))
                    # (p has shape: (batch_size, num_classes, h, w))

                    label = label.long().cuda()

                    p = torch.transpose(p, 1, 2) # (shape: (batch_size, h, num_classes, w))
                    p = torch.transpose(p, 2, 3) # (shape: (batch_size, h, w, num_classes))

                    nonignore_mask = (label != 255).detach() # (shape: (batch_size, h, w))

                    label = label[nonignore_mask] # (shape: (num_nonignores, ))
                    p = p[nonignore_mask, :] # (shape: (num_nonignores, num_classes))

                    label_onehot = torch.zeros(label.size(0), p.size(1)).cuda().scatter_(1, label.unsqueeze(1), 1.0) # (shape: (num_nonignores, num_classes))

                    squared_error = torch.sum(torch.pow(label_onehot - p, 2), dim=1) # (shape: (num_nonignores, ))

                    entropy = -torch.sum(p*torch.log(p), dim=1) # (shape: (num_nonignores, ))

                    entropy_values = np.concatenate((entropy_values, entropy.data.cpu().numpy()))
                    squared_error_values = np.concatenate((squared_error_values, squared_error.data.cpu().numpy()))


                    label = label.cpu().data.numpy().astype(np.uint8) # (shape: (num_nonignores, ))
                    p = p.cpu().data.numpy() # (shape: (num_nonignores, num_classes))

                    pred = np.argmax(p, axis=1).astype(np.uint8) # (shape: (num_nonignores, ))

                    conf = np.max(p, axis=1) # (shape: (num_nonignores, ))

                    for i in range(num_conf_intervals):
                        lower = i*conf_interval_size
                        upper = (i + 1)*conf_interval_size

                        confs_in_interval = conf[np.nonzero(np.logical_and(conf >= lower, conf < upper))] # (shape: (num_preds_in_interval, ))
                        num_preds_in_interval = confs_in_interval.shape[0]
                        num_correct_preds_in_interval = np.count_nonzero(np.logical_and(np.logical_and(conf >= lower, conf < upper), pred == label))

                        interval_2_num_preds[i] += num_preds_in_interval
                        interval_2_num_correct_preds[i] += num_correct_preds_in_interval
                        interval_2_confs[i] = np.concatenate((interval_2_confs[i], confs_in_interval))

                        # if num_preds_in_interval > 0:
                        #     accuracy = float(num_correct_preds_in_interval/num_preds_in_interval)
                        # else:
                        #     accuracy = -1.0
                        # print ("[%g, %g[" % (lower, upper))
                        # print (num_preds_in_interval)
                        # print (num_correct_preds_in_interval)
                        # print (np.mean(confs_in_interval))
                        # print (accuracy)
                        # print ("$$$$$$")

                    # # # # # # # # # # # # # # # # # # debug START:
                    # if step > -1:
                    #     break
                    # # # # # # # # # # # # # # # # # # debug END:

            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)

            IU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IU = IU_array.mean()
            print({'meanIU':mean_IU, 'IU_array':IU_array})
            mIoU_values[M].append(mean_IU)

            # (entropy_values has shape: (num_nonignores_predictions, ))
            # (squared_error_values has shape: (num_nonignores_predictions, ))
            print (entropy_values.shape)
            print (squared_error_values.shape)

            num_nonignores_predictions = squared_error_values.shape[0]
            print ("num_nonignores_predictions: %d" % num_nonignores_predictions)

            brier_score = np.mean(squared_error_values)
            print ("brier_score: %g" % brier_score)

            sorted_inds_entropy = np.argsort(entropy_values) # (entropy_values[sorted_inds_entropy[0]]: SMALLEST element of entropy_values)
            sorted_inds_error = np.argsort(squared_error_values)

            entropy_brier_scores = []
            error_brier_scores = []
            fractions = list(np.arange(start=0.0, stop=1.0, step=0.01)) # ([0.0, 0.01, ..., 0.99], 100 elements)
            for step, fraction in enumerate(fractions):
                if (step % 10) == 0:
                    print ("fraction: %d/%d" % (step+1, len(fractions)))

                entropy_brier_score = np.mean( squared_error_values[sorted_inds_entropy[0:int((1.0-fraction)*num_nonignores_predictions)]] )
                entropy_brier_scores.append(entropy_brier_score)

                error_brier_score = np.mean( squared_error_values[sorted_inds_error[0:int((1.0-fraction)*num_nonignores_predictions)]] )
                error_brier_scores.append(error_brier_score)

            error_brier_scores_normalized = error_brier_scores/error_brier_scores[0]
            entropy_brier_scores_normalized = entropy_brier_scores/error_brier_scores[0]

            sparsification_errors = entropy_brier_scores_normalized - error_brier_scores_normalized

            ause = np.trapz(y=sparsification_errors, x=fractions)
            print ("Area Under the Sparsification Error curve (AUSE): %g" % ause)

            sparsification_error_values[model_i][M][run] = np.array(sparsification_errors)
            error_brier_score_values[model_i][M][run] = np.array(error_brier_scores_normalized)
            entropy_brier_score_values[model_i][M][run] = np.array(entropy_brier_scores_normalized)

            auc_sparsification_error_values[M].append(ause)

            # num_preds_total = 0
            # for i in range(num_conf_intervals):
            #     if interval_2_num_preds[i] > 0:
            #         accuracy = float(interval_2_num_correct_preds[i]/interval_2_num_preds[i])
            #     else:
            #         accuracy = -1.0
            #     print (interval_2_num_preds[i])
            #     print (interval_2_num_correct_preds[i])
            #     print (interval_2_confs[i].shape)
            #     print (np.mean(interval_2_confs[i]))
            #     print (accuracy)
            #     print ("$$$$$$$$$$$$$$$$$$")
            #     num_preds_total += interval_2_num_preds[i]
            # print (num_preds_total)

            # (num_preds_total == num_nonignores_predictions)

            ECE = 0.0
            for i in range(num_conf_intervals):
                if interval_2_num_preds[i] > 0:
                    accuracy = float(float(interval_2_num_correct_preds[i])/float(interval_2_num_preds[i]))

                    conf = np.mean(interval_2_confs[i])
                    interval_2_mean_conf[i] = conf

                    ECE += float(float(interval_2_num_preds[i]/float(num_nonignores_predictions)))*np.abs(accuracy - conf)

                    # print (accuracy)
                    # print (conf)
                    # print (ECE)
                    # print ("%%%%%%%%%%%%%%%%%%%%%")

            print ("ECE: %g" % ECE)

            rel_diagram = {}
            rel_diagram["interval_2_num_preds"] = interval_2_num_preds
            rel_diagram["interval_2_num_correct_preds"] = interval_2_num_correct_preds
            rel_diagram["interval_2_mean_conf"] = interval_2_mean_conf

            rel_diagrams[model_i][M][run] = rel_diagram

            ECE_values[M].append(ECE)

            print ("#######################")

        print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

auc_sparsification_error_means = {}
auc_sparsification_error_stds = {}
mIoU_means = {}
mIoU_stds = {}
ECE_means = {}
ECE_stds = {}
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

    mIoU_values_mean = 0.0
    for value in mIoU_values[M]:
        mIoU_values_mean += value/float(num_runs_per_M*num_model_is)

    mIoU_values_var = 0.0
    for value in mIoU_values[M]:
        mIoU_values_var += ((value - mIoU_values_mean)**2)/float(num_runs_per_M*num_model_is)

    mIoU_values_std = np.sqrt(mIoU_values_var)

    mIoU_means[M] = mIoU_values_mean
    mIoU_stds[M] = mIoU_values_std

    ###

    ECE_values_mean = 0.0
    for value in ECE_values[M]:
        ECE_values_mean += value/float(num_runs_per_M*num_model_is)

    ECE_values_var = 0.0
    for value in ECE_values[M]:
        ECE_values_var += ((value - ECE_values_mean)**2)/float(num_runs_per_M*num_model_is)

    ECE_values_std = np.sqrt(ECE_values_var)

    ECE_means[M] = ECE_values_mean
    ECE_stds[M] = ECE_values_std

for M in M_values:
    print ("M = %d, AUSE - mean: %g, std: %g" % (M, auc_sparsification_error_means[M], auc_sparsification_error_stds[M]))
    print ("M = %d, mIoU - mean: %g, std: %g" % (M, mIoU_means[M], mIoU_stds[M]))
    print ("M = %d, ECE - mean: %g, std: %g" % (M, ECE_means[M], ECE_stds[M]))
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
plt.savefig("%s/sparsification_error_curve.png" % output_path)
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
    plt.savefig("%s/sparsification_error_curve_M%d.png" % (output_path, M))
    plt.close(1)

    plt.figure(1)
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            plt.plot(fractions, error_brier_score_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run], linestyle="dotted")
            plt.plot(fractions, entropy_brier_score_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run])
    plt.ylabel("Brier score (normalized)")
    plt.xlabel("Fraction of removed pixels")
    plt.ylim((-0.05, 1.05))
    plt.title("Sparsification plot - M=%d" % M)
    plt.savefig("%s/sparsification_plot_M%d.png" % (output_path, M))
    plt.close(1)

for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            plt.figure(1)
            plt.plot(fractions, error_brier_score_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run], linestyle="dotted", label="Oracle")
            plt.plot(fractions, entropy_brier_score_values[model_i][M][run], color=run_colors[model_i_step*num_runs_per_M + run], label="Model")
            plt.legend()
            plt.ylabel("RMSE (normalized)")
            plt.xlabel("Fraction of removed pixels")
            plt.ylim((-0.05, 1.05))
            plt.title("Sparsification plot - M=%d, model_i=%d, %d" % (M, model_i, run))
            plt.savefig("%s/sparsification_plot_M%d_model_i%d_%d.png" % (output_path, M, model_i, run))
            plt.close(1)

for M in M_values:
    plt.figure(1)
    plt.plot([0.0, 1.0], [0.0, 1.0], "k:")
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            rel_diagram = rel_diagrams[model_i][M][run]

            interval_2_num_preds = rel_diagram["interval_2_num_preds"]
            interval_2_num_correct_preds = rel_diagram["interval_2_num_correct_preds"]
            interval_2_mean_conf = rel_diagram["interval_2_mean_conf"]

            accuracies = []
            confs = []
            for i in range(num_conf_intervals):
                if interval_2_num_preds[i] > 0:
                    confs.append(interval_2_mean_conf[i])
                    accuracies.append(float(float(interval_2_num_correct_preds[i])/float(interval_2_num_preds[i])))

            if (model_i_step == 0) and (run == 0):
                plt.plot(np.array(confs), np.array(accuracies), color=colors[M], marker="o", alpha=0.5, label="M = %d" % M)
                plt.plot(np.array(confs), np.array(accuracies), color=colors[M], alpha=0.5)
            else:
                plt.plot(np.array(confs), np.array(accuracies), color=colors[M], marker="o", alpha=0.5)
                plt.plot(np.array(confs), np.array(accuracies), color=colors[M], alpha=0.5)
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.ylim((-0.05, 1.05))
    plt.xlim((-0.05, 1.05))
    plt.title("Reliability diagram - M=%d" % M)
    plt.savefig("%s/rel_diagram_M%d.png" % (output_path, M))
    plt.close(1)

for M in M_values:
    for model_i_step, model_i in enumerate(model_is):
        for run in range(num_runs_per_M):
            rel_diagram = rel_diagrams[model_i][M][run]

            interval_2_num_preds = rel_diagram["interval_2_num_preds"]
            interval_2_num_correct_preds = rel_diagram["interval_2_num_correct_preds"]
            interval_2_mean_conf = rel_diagram["interval_2_mean_conf"]

            num_total_preds = 0
            for i in range(num_conf_intervals):
                num_total_preds += interval_2_num_preds[i]

            accuracies = []
            confs = []
            hist_heights = []
            hist_positions = []
            for i in range(num_conf_intervals):
                hist_heights.append(float(float(interval_2_num_preds[i])/float(num_total_preds)))
                hist_positions.append(i*conf_interval_size + conf_interval_size/2.0)

                if interval_2_num_preds[i] > 0:
                    confs.append(interval_2_mean_conf[i])
                    accuracies.append(float(float(interval_2_num_correct_preds[i])/float(interval_2_num_preds[i])))

            plt.figure(1)
            plt.bar(np.array(hist_positions), np.array(hist_heights), width=conf_interval_size, color="black", alpha=0.15)
            plt.plot([0.0, 1.0], [0.0, 1.0], "k:")
            plt.plot(np.array(confs), np.array(accuracies), "ro")
            plt.plot(np.array(confs), np.array(accuracies), "r")
            plt.ylabel("Accuracy")
            plt.xlabel("Confidence")
            plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            plt.ylim((-0.05, 1.05))
            plt.xlim((-0.05, 1.05))
            plt.title("Reliability diagram - M=%d, model_i=%d, %d" % (M, model_i, run))
            plt.savefig("%s/rel_diagram_M%d_model_i%d_%d.png" % (output_path, M, model_i, run))
            plt.close(1)

with open("%s/auc_sparsification_error_values.pkl" % output_path, "wb") as file:
    pickle.dump(auc_sparsification_error_values, file)

with open("%s/mIoU_values.pkl" % output_path, "wb") as file:
    pickle.dump(mIoU_values, file)

with open("%s/sparsification_error_values.pkl" % output_path, "wb") as file:
    pickle.dump(sparsification_error_values, file)

with open("%s/error_brier_score_values.pkl" % output_path, "wb") as file:
    pickle.dump(error_brier_score_values, file)

with open("%s/entropy_brier_score_values.pkl" % output_path, "wb") as file:
    pickle.dump(entropy_brier_score_values, file)

with open("%s/ECE_values.pkl" % output_path, "wb") as file:
    pickle.dump(ECE_values, file)

with open("%s/rel_diagrams.pkl" % output_path, "wb") as file:
    pickle.dump(rel_diagrams, file)
