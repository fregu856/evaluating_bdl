# code-checked

from datasets import ToyDatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
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

M = 2

batch_size = 32

max_logvar = 2.0

L = 64
num_epochs = L*150

num_epochs_low = int(0.75*num_epochs)
print (num_epochs_low)

step_size = float(num_epochs - num_epochs_low)/float(M-1)
print (step_size)

networks = []
for i in range(M):
    print (int(num_epochs - i*step_size))

    network = ToyNet("eval_SGLD-64_1", project_dir="/root/evaluating_bdl/toyRegression").cuda()
    network.load_state_dict(torch.load("/root/evaluating_bdl/toyRegression/training_logs/model_SGLD-64_1/checkpoints/model_SGLD-64_1_epoch_%d.pth" % int(num_epochs - i*step_size)))
    networks.append(network)

M = float(len(networks))
print (M)

val_dataset = ToyDatasetEval()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

for network in networks:
    network.eval()
x_values = []
final_mean_values = []
final_sigma_tot_values = []
final_sigma_epi_values = []
final_sigma_alea_values = []
for step, (x) in enumerate(val_loader):
    x = Variable(x).cuda().unsqueeze(1) # (shape: (batch_size, 1))

    means = []
    vars = []
    for network in networks:
        outputs = network(x)
        mean = outputs[0] # (shape: (batch_size, ))
        var = outputs[1] # (shape: (batch_size, )) (log(sigma^2))
        var = max_logvar - F.relu(max_logvar-var)

        means.append(mean)
        vars.append(var)

    for i in range(x.size(0)):
        x_value = x[i].data.cpu().numpy()[0]

        mean_values = []
        for mean in means:
            mean_value = mean[i].data.cpu().numpy()[0]
            mean_values.append(mean_value)

        sigma_alea_values = []
        for var in vars:
            sigma_alea_value = torch.exp(var[i]).data.cpu().numpy()[0]
            sigma_alea_values.append(sigma_alea_value)

        mean_value = 0.0
        for value in mean_values:
            mean_value += value/M

        sigma_epi_value = 0.0
        for value in mean_values:
            sigma_epi_value += ((mean_value - value)**2)/M

        sigma_alea_value = 0.0
        for value in sigma_alea_values:
            sigma_alea_value += value/M

        sigma_tot_value = sigma_epi_value + sigma_alea_value

        x_values.append(x_value)
        final_mean_values.append(mean_value)
        final_sigma_epi_values.append(sigma_epi_value)
        final_sigma_alea_values.append(sigma_alea_value)
        final_sigma_tot_values.append(sigma_tot_value)

plt.figure(1)
plt.plot(x_values, final_mean_values, "r")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_alea_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_alea_values)), color="C3", alpha=0.25)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("predicted vs true mean(x) with aleatoric uncertainty")
plt.savefig("%s/mu_alea_pred_true.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("true mean(x) with aleatoric uncertainty")
plt.savefig("%s/mu_alea_true.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, final_mean_values, "r")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_alea_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_alea_values)), color="C3", alpha=0.25)
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("predicted mean(x) with aleatoric uncertainty")
plt.savefig("%s/mu_alea_pred.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, np.sqrt(np.array(final_sigma_alea_values)), "r")
plt.plot(x_values, 0.15*(1.0/(1 + np.exp(-np.array(x_values)))), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.xlabel("x")
plt.title("predicted vs true aleatoric uncertainty")
plt.savefig("%s/alea_pred_true.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, np.sqrt(np.array(final_sigma_epi_values)), "r")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.xlabel("x")
plt.title("predicted epistemic uncertainty")
plt.savefig("%s/epi_pred.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, final_mean_values, "r")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_epi_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_epi_values)), color="C1", alpha=0.25)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("predicted vs true mean(x) with epistemic uncertainty")
plt.savefig("%s/mu_epi_pred_true.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, final_mean_values, "r")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_tot_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_tot_values)), color="C2", alpha=0.25)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.ylabel("mu(x)")
plt.xlabel("x")
plt.title("predicted vs true mean(x) with total uncertainty")
plt.savefig("%s/mu_tot_pred_true.png" % network.model_dir)
plt.close(1)

with open("/root/evaluating_bdl/toyRegression/HMC/x_values.pkl", "rb") as file: # (needed for python3)
    x_values_HMC = pickle.load(file) # (list of 1000 elements)

with open("/root/evaluating_bdl/toyRegression/HMC/final_mean_values.pkl", "rb") as file: # (needed for python3)
    mean_values_HMC = pickle.load(file) # (list of 1000 elements)

with open("/root/evaluating_bdl/toyRegression/HMC/final_sigma_tot_values.pkl", "rb") as file: # (needed for python3)
    sigma_squared_values_HMC = pickle.load(file) # (list of 1000 elements)

plt.figure(1)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.xlim([-6, 6])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/predictive_density_GT_.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values_HMC, mean_values_HMC, "r")
plt.fill_between(x_values_HMC, np.array(mean_values_HMC) - 2*np.sqrt(np.array(sigma_squared_values_HMC)), np.array(mean_values_HMC) + 2*np.sqrt(np.array(sigma_squared_values_HMC)), color="C3", alpha=0.25)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.xlim([-6, 6])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/predictive_density_HMC_.png" % network.model_dir)
plt.close(1)

plt.figure(1)
plt.plot(x_values, final_mean_values, "r")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_tot_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_tot_values)), color="C3", alpha=0.25)
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.xlim([-6, 6])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/predictive_density_.png" % network.model_dir)
plt.close(1)

with open("/root/evaluating_bdl/toyRegression/x.pkl", "rb") as file: # (needed for python3)
    x = pickle.load(file)

with open("/root/evaluating_bdl/toyRegression/y.pkl", "rb") as file: # (needed for python3)
    y = pickle.load(file)

plt.figure(1)
plt.plot(x, y, "2k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.xlim([-6, 6])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/training_data.png" % network.model_dir)
plt.close(1)

M = int(M)

fc1_mean_weight_samples = np.zeros((M, 1, 10, 1))
fc1_mean_bias_samples = np.zeros((M, 1, 10))
fc2_mean_weight_samples = np.zeros((M, 1, 10, 10))
fc2_mean_bias_samples = np.zeros((M, 1, 10))
fc3_mean_weight_samples = np.zeros((M, 1, 1, 10))
fc3_mean_bias_samples = np.zeros((M, 1, 1))

fc1_var_weight_samples = np.zeros((M, 1, 10, 1))
fc1_var_bias_samples = np.zeros((M, 1, 10))
fc2_var_weight_samples = np.zeros((M, 1, 10, 10))
fc2_var_bias_samples = np.zeros((M, 1, 10))
fc3_var_weight_samples = np.zeros((M, 1, 1, 10))
fc3_var_bias_samples = np.zeros((M, 1, 1))
for index, network in enumerate(networks):
    for name, param in network.named_parameters():
        if name == "fc1_mean.weight":
            fc1_mean_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc1_mean.bias":
            fc1_mean_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_mean.weight":
            fc2_mean_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_mean.bias":
            fc2_mean_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_mean.weight":
            fc3_mean_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_mean.bias":
            fc3_mean_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc1_var.weight":
            fc1_var_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc1_var.bias":
            fc1_var_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_var.weight":
            fc2_var_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_var.bias":
            fc2_var_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_var.weight":
            fc3_var_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_var.bias":
            fc3_var_bias_samples[index, 0, :] = param.data.cpu().numpy()
        else:
            raise Exception("Unknown network parameter!")

import os
if not os.path.exists("%s/param_distributions" % (network.model_dir)):
    os.makedirs("%s/param_distributions" % (network.model_dir))

# (fc1_mean_weight_samples has shape: (M, 1, 10, 1))
for param_index in range(10):
    values = fc1_mean_weight_samples[:, 0, param_index, 0] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_mean_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc1_mean_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc1_mean_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_mean_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc2_mean_weight_samples has shape: (M, 1, 10, 10))
for param_index_i in range(10):
    for param_index_j in range(10):
        values = fc2_mean_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/param_distributions/fc2_mean_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc2_mean_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc2_mean_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc2_mean_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_mean_weight_samples has shape: (M, 1, 1, 10))
for param_index in range(10):
    values = fc3_mean_weight_samples[:, 0, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc3_mean_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_mean_bias_samples has shape: (M, 1, 1))
values = fc3_mean_bias_samples[:, 0, 0] # (shape: (M, ))
plt.figure(1)
plt.hist(np.array(values), bins=100)
plt.savefig("%s/param_distributions/fc3_mean_bias.png" % (network.model_dir))
plt.close(1)


# (fc1_var_weight_samples has shape: (M, 1, 10, 1))
for param_index in range(10):
    values = fc1_var_weight_samples[:, 0, param_index, 0] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_var_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc1_var_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc1_var_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_var_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc2_var_weight_samples has shape: (M, 1, 10, 10))
for param_index_i in range(10):
    for param_index_j in range(10):
        values = fc2_var_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/param_distributions/fc2_var_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc2_var_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc2_var_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc2_var_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_var_weight_samples has shape: (M, 1, 1, 10))
for param_index in range(10):
    values = fc3_var_weight_samples[:, 0, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc3_var_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_var_bias_samples has shape: (M, 1, 1))
values = fc3_var_bias_samples[:, 0, 0] # (shape: (M, ))
plt.figure(1)
plt.hist(np.array(values), bins=100)
plt.savefig("%s/param_distributions/fc3_var_bias.png" % (network.model_dir))
plt.close(1)
