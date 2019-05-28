# code-checked
# server-checked

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
import matplotlib.cm as cm
import cv2

import numpy as np

L = 256
num_epochs = L*150

num_epochs_low = int(0.75*num_epochs)
print (num_epochs_low)

x_min = -6.0
x_max = 6.0
num_points = 60

M_values = [1, 4, 16, 64, 256]
for M in M_values:
    for iter in range(6):
        print (M)

        if M > 1:
            step_size = float(num_epochs - num_epochs_low)/float(M-1)
        else:
            step_size = 0
        print (step_size)

        networks = []
        for i in range(M):
            print (int(num_epochs - i*step_size))

            network = ToyNet("eval_SGLD-256_1-10", project_dir="/root/evaluating_bdl/toyClassification").cuda()
            network.load_state_dict(torch.load("/root/evaluating_bdl/toyClassification/training_logs/model_SGLD-256_%d/checkpoints/model_SGLD-256_%d_epoch_%d.pth" % (iter+1, iter+1, int(num_epochs - i*step_size))))
            networks.append(network)

        M_float = float(len(networks))
        print (M_float)

        for network in networks:
            network.eval()

        false_prob_values = np.zeros((num_points, num_points))
        x_values = np.linspace(x_min, x_max, num_points, dtype=np.float32)
        for x_1_i, x_1_value in enumerate(x_values):
            for x_2_i, x_2_value in enumerate(x_values):
                x = torch.from_numpy(np.array([x_1_value, x_2_value])).unsqueeze(0).cuda() # (shape: (1, 2))

                mean_prob_vector = np.zeros((2, ))
                for network in networks:
                    logits = network(x) # (shape: (1, num_classes)) (num_classes==2)
                    prob_vector = F.softmax(logits, dim=1) # (shape: (1, num_classes))

                    prob_vector = prob_vector.data.cpu().numpy()[0] # (shape: (2, ))

                    mean_prob_vector += prob_vector/M_float

                false_prob_values[x_2_i, x_1_i] = mean_prob_vector[0]

        plt.figure(1)
        x_1, x_2 = np.meshgrid(x_values, x_values)
        plt.pcolormesh(x_1, x_2, false_prob_values, cmap="RdBu", vmin=0, vmax=1)
        plt.colorbar()
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.savefig("%s/predictive_density_M=%d_%d.png" % (network.model_dir, M, iter+1))
        plt.close(1)

    print ("##################################################################")

# M = int(M)
#
# fc1_weight_samples = np.zeros((M, 1, 10, 2))
# fc1_bias_samples = np.zeros((M, 1, 10))
# fc2_weight_samples = np.zeros((M, 1, 10, 10))
# fc2_bias_samples = np.zeros((M, 1, 10))
# fc3_weight_samples = np.zeros((M, 1, 2, 10))
# fc3_bias_samples = np.zeros((M, 1, 2))
# for index, network in enumerate(networks):
#     for name, param in network.named_parameters():
#         if name == "fc1.weight":
#             fc1_weight_samples[index, 0, :] = param.data.cpu().numpy()
#         elif name == "fc1.bias":
#             fc1_bias_samples[index, 0, :] = param.data.cpu().numpy()
#         elif name == "fc2.weight":
#             fc2_weight_samples[index, 0, :] = param.data.cpu().numpy()
#         elif name == "fc2.bias":
#             fc2_bias_samples[index, 0, :] = param.data.cpu().numpy()
#         elif name == "fc3.weight":
#             fc3_weight_samples[index, 0, :] = param.data.cpu().numpy()
#         elif name == "fc3.bias":
#             fc3_bias_samples[index, 0, :] = param.data.cpu().numpy()
#         else:
#             raise Exception("Unknown network parameter!")
#
# import os
# if not os.path.exists("%s/param_distributions" % (network.model_dir)):
#     os.makedirs("%s/param_distributions" % (network.model_dir))
#
# # (fc1_weight_samples has shape: (M, 1, 10, 2))
# for param_index_i in range(10):
#     for param_index_j in range(2):
#         values = fc1_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
#         plt.figure(1)
#         plt.hist(np.array(values), bins=100)
#         plt.savefig("%s/param_distributions/fc1_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
#         plt.close(1)
#
# # (fc1_bias_samples has shape: (M, 1, 10))
# for param_index in range(10):
#     values = fc1_bias_samples[:, 0, param_index] # (shape: (M, ))
#     plt.figure(1)
#     plt.hist(np.array(values), bins=100)
#     plt.savefig("%s/param_distributions/fc1_bias_%d.png" % (network.model_dir, param_index))
#     plt.close(1)
#
# # (fc2_weight_samples has shape: (M, 1, 10, 10))
# for param_index_i in range(10):
#     for param_index_j in range(10):
#         values = fc2_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
#         plt.figure(1)
#         plt.hist(np.array(values), bins=100)
#         plt.savefig("%s/param_distributions/fc2_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
#         plt.close(1)
#
# # (fc2_bias_samples has shape: (M, 1, 10))
# for param_index in range(10):
#     values = fc2_bias_samples[:, 0, param_index] # (shape: (M, ))
#     plt.figure(1)
#     plt.hist(np.array(values), bins=100)
#     plt.savefig("%s/param_distributions/fc2_bias_%d.png" % (network.model_dir, param_index))
#     plt.close(1)
#
# # (fc3_weight_samples has shape: (M, 1, 2, 10))
# for param_index_i in range(2):
#     for param_index_j in range(10):
#         values = fc3_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
#         plt.figure(1)
#         plt.hist(np.array(values), bins=100)
#         plt.savefig("%s/param_distributions/fc3_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
#         plt.close(1)
#
# # (fc3_bias_samples has shape: (M, 1, 2))
# for param_index in range(2):
#     values = fc3_bias_samples[:, 0, param_index] # (shape: (M, ))
#     plt.figure(1)
#     plt.hist(np.array(values), bins=100)
#     plt.savefig("%s/param_distributions/fc3_bias_%d.png" % (network.model_dir, param_index))
#     plt.close(1)
