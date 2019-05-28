# code-checked
# server-checked

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import cv2

from model_pytorch import ToyNet

from model_pyro import det_net

batch_size = 32

x_min = -6.0
x_max = 6.0
num_points = 60

with open("%s/fc1_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc1_weight_samples = pickle.load(file) # (shape: (1000, 1, 10, 1))
    print (fc1_weight_samples.shape)
with open("%s/fc1_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc1_bias_samples = pickle.load(file) # (shape: (1000, 1, 10))
    print (fc1_bias_samples.shape)

with open("%s/fc2_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc2_weight_samples = pickle.load(file) # (shape: (1000, 1, 10, 10))
    print (fc2_weight_samples.shape)
with open("%s/fc2_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc2_bias_samples = pickle.load(file) # (shape: (1000, 1, 10))
    print (fc2_bias_samples.shape)

with open("%s/fc3_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc3_weight_samples = pickle.load(file) # (shape: (1000, 1, 1, 10))
    print (fc3_weight_samples.shape)
with open("%s/fc3_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc3_bias_samples = pickle.load(file) # (shape: (1000, 1, 1))
    print (fc3_bias_samples.shape)

num_samples = fc1_weight_samples.shape[0]
print ("num_samples: %d" % num_samples)

networks = []
for i in range(num_samples):
    fc1_weight = fc1_weight_samples[i, 0, :] # (shape: (10, 1))
    fc1_bias = fc1_bias_samples[i, 0, :] # (shape: (10, ))

    fc2_weight = fc2_weight_samples[i, 0, :] # (shape: (10, 10))
    fc2_bias = fc2_bias_samples[i, 0, :] # (shape: (10, ))

    fc3_weight = fc3_weight_samples[i, 0, :] # (shape: (1, 10))
    fc3_bias = fc3_bias_samples[i, 0, :] # (shape: (1, ))

    network = ToyNet("eval_HMC", project_dir="/root/evaluating_bdl/toyClassification").cuda()
    for name, param in network.named_parameters():
        if name == "fc1.weight":
            param.data = torch.from_numpy(fc1_weight).cuda()
        elif name == "fc1.bias":
            param.data = torch.from_numpy(fc1_bias).cuda()
        elif name == "fc2.weight":
            param.data = torch.from_numpy(fc2_weight).cuda()
        elif name == "fc2.bias":
            param.data = torch.from_numpy(fc2_bias).cuda()
        elif name == "fc3.weight":
            param.data = torch.from_numpy(fc3_weight).cuda()
        elif name == "fc3.bias":
            param.data = torch.from_numpy(fc3_bias).cuda()
        else:
            raise Exception("Unknown network parameter!")
    networks.append(network)

M = float(len(networks))
print (M)

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

            mean_prob_vector += prob_vector/M

        false_prob_values[x_2_i, x_1_i] = mean_prob_vector[0]

with open("%s/false_prob_values.pkl" % network.model_dir, "wb") as file:
    pickle.dump(false_prob_values, file)

# #####
# with open("/root/evaluating_bdl/toyClassification/HMC/false_prob_values.pkl", "rb") as file: # (needed for python3)
#     false_prob_values = pickle.load(file) # (shape: (60, 60))
# x_values = np.linspace(x_min, x_max, num_points, dtype=np.float32)
# network = ToyNet("eval_HMC", project_dir="/root/evaluating_bdl/toyClassification").cuda()
# #####

plt.figure(1)
x_1, x_2 = np.meshgrid(x_values, x_values)
plt.pcolormesh(x_1, x_2, false_prob_values, cmap="RdBu", vmin=0, vmax=1)
plt.colorbar()
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/predictive_density.png" % network.model_dir)
plt.savefig("%s/predictive_density.pdf" % network.model_dir, dpi=400)
plt.close(1)

with open("/root/evaluating_bdl/toyClassification/x.pkl", "rb") as file: # (needed for python3)
    x = pickle.load(file) # (shape: (2000, 2))

with open("/root/evaluating_bdl/toyClassification/y.pkl", "rb") as file: # (needed for python3)
    y = pickle.load(file) #  (shape: (2000, ))

x_1_train = []
x_2_train = []
y_train = []
for i in range(x.shape[0]):
    if x[i, 0] > 0:
        x_1_train.append(x[i, 0])
        x_2_train.append(x[i, 1])
        y_train.append(y[i])

y_train = np.array(y_train)
x_train = np.zeros((len(y_train), 2), dtype=np.float32)
x_train[:, 0] = np.array(x_1_train)
x_train[:, 1] = np.array(x_2_train)

fig_h = 5
fig_w = 1.09375*fig_h
cmap = matplotlib.cm.get_cmap("RdBu")
x_train_false = x_train[y_train == 0] # (shape: (num_false, 2))
x_train_true = x_train[y_train == 1] # (shape: (num_true, 2))
print ("num_false: %d" % x_train_false.shape[0])
print ("num_true: %d" % x_train_true.shape[0])
plt.figure(1, figsize=(fig_w, fig_h))
plt.plot(x_train_false[:, 0], x_train_false[:, 1], linestyle="None", marker="2", color=cmap(255))
plt.plot(x_train_true[:, 0], x_train_true[:, 1], linestyle="None", marker="2", color=cmap(0))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/training_data.png" % network.model_dir)
plt.savefig("%s/training_data.pdf" % network.model_dir, dpi=400)
plt.close(1)

x_values = np.linspace(x_min, x_max, 2, dtype=np.float32)
x_1, x_2 = np.meshgrid(x_values, x_values)
dist = np.sqrt(x_1**2 + x_2**2)
false_prob_values_GT = np.zeros(dist.shape)
plt.figure(1)
plt.pcolormesh(x_1, x_2, false_prob_values_GT, cmap="RdBu", vmin=0, vmax=1)
plt.colorbar()
circle = plt.Circle((0,0), 2.4, color=cmap(255))
plt.gcf().gca().add_artist(circle)
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/predictive_density_GT.png" % network.model_dir)
plt.savefig("%s/predictive_density_GT.pdf" % network.model_dir, dpi=400)
plt.close(1)

import os
if not os.path.exists("%s/param_distributions" % (network.model_dir)):
    os.makedirs("%s/param_distributions" % (network.model_dir))

# (fc1_weight_samples has shape: (M, 1, 10, 2))
for param_index_i in range(10):
    for param_index_j in range(2):
        values = fc1_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/param_distributions/fc1_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc1_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc1_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc2_weight_samples has shape: (M, 1, 10, 10))
for param_index_i in range(10):
    for param_index_j in range(10):
        values = fc2_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/param_distributions/fc2_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc2_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc2_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc2_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_weight_samples has shape: (M, 1, 2, 10))
for param_index_i in range(2):
    for param_index_j in range(10):
        values = fc3_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/param_distributions/fc3_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc3_bias_samples has shape: (M, 1, 2))
for param_index in range(2):
    values = fc3_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc3_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)
