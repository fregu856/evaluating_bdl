# code-checked
# server-checked

from datasets import ToyDatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

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

from model_pytorch import ToyNet

from model_pyro import det_net

batch_size = 32

max_logvar = 2.0

with open("%s/fc1_mean_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc1_mean_weight_samples = pickle.load(file) # (shape: (1000, 1, 10, 1))
    print (fc1_mean_weight_samples.shape)
with open("%s/fc1_mean_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc1_mean_bias_samples = pickle.load(file) # (shape: (1000, 1, 10))
    print (fc1_mean_bias_samples.shape)

with open("%s/fc2_mean_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc2_mean_weight_samples = pickle.load(file) # (shape: (1000, 1, 10, 10))
    print (fc2_mean_weight_samples.shape)
with open("%s/fc2_mean_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc2_mean_bias_samples = pickle.load(file) # (shape: (1000, 1, 10))
    print (fc2_mean_bias_samples.shape)

with open("%s/fc3_mean_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc3_mean_weight_samples = pickle.load(file) # (shape: (1000, 1, 1, 10))
    print (fc3_mean_weight_samples.shape)
with open("%s/fc3_mean_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc3_mean_bias_samples = pickle.load(file) # (shape: (1000, 1, 1))
    print (fc3_mean_bias_samples.shape)

with open("%s/fc1_var_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc1_var_weight_samples = pickle.load(file) # (shape: (1000, 1, 10, 1))
    print (fc1_var_weight_samples.shape)
with open("%s/fc1_var_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc1_var_bias_samples = pickle.load(file) # (shape: (1000, 1, 10))
    print (fc1_var_bias_samples.shape)

with open("%s/fc2_var_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc2_var_weight_samples = pickle.load(file) # (shape: (1000, 1, 10, 10))
    print (fc2_var_weight_samples.shape)
with open("%s/fc2_var_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc2_var_bias_samples = pickle.load(file)  # (shape: (1000, 1, 10))
    print (fc2_var_bias_samples.shape)

with open("%s/fc3_var_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc3_var_weight_samples = pickle.load(file) # (shape: (1000, 1, 1, 10))
    print (fc3_var_weight_samples.shape)
with open("%s/fc3_var_bias_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    fc3_var_bias_samples = pickle.load(file) # (shape: (1000, 1, 1))
    print (fc3_var_bias_samples.shape)

num_samples = fc1_mean_weight_samples.shape[0]
print ("num_samples: %d" % num_samples)

networks = []
for i in range(num_samples):
    fc1_mean_weight = fc1_mean_weight_samples[i, 0, :] # (shape: (10, 1))
    fc1_mean_bias = fc1_mean_bias_samples[i, 0, :] # (shape: (10, ))

    fc2_mean_weight = fc2_mean_weight_samples[i, 0, :] # (shape: (10, 10))
    fc2_mean_bias = fc2_mean_bias_samples[i, 0, :] # (shape: (10, ))

    fc3_mean_weight = fc3_mean_weight_samples[i, 0, :] # (shape: (1, 10))
    fc3_mean_bias = fc3_mean_bias_samples[i, 0, :] # (shape: (1, ))

    fc1_var_weight = fc1_var_weight_samples[i, 0, :] # (shape: (10, 1))
    fc1_var_bias = fc1_var_bias_samples[i, 0, :] # (shape: (10, ))

    fc2_var_weight = fc2_var_weight_samples[i, 0, :] # (shape: (10, 10))
    fc2_var_bias = fc2_var_bias_samples[i, 0, :] # (shape: (10, ))

    fc3_var_weight = fc3_var_weight_samples[i, 0, :] # (shape: (1, 10))
    fc3_var_bias = fc3_var_bias_samples[i, 0, :] # (shape: (1, ))

    network = ToyNet("eval_HMC", project_dir="/root/evaluating_bdl/toyRegression").cuda()
    for name, param in network.named_parameters():
        if name == "fc1_mean.weight":
            param.data = torch.from_numpy(fc1_mean_weight).cuda()
        elif name == "fc1_mean.bias":
            param.data = torch.from_numpy(fc1_mean_bias).cuda()
        elif name == "fc2_mean.weight":
            param.data = torch.from_numpy(fc2_mean_weight).cuda()
        elif name == "fc2_mean.bias":
            param.data = torch.from_numpy(fc2_mean_bias).cuda()
        elif name == "fc3_mean.weight":
            param.data = torch.from_numpy(fc3_mean_weight).cuda()
        elif name == "fc3_mean.bias":
            param.data = torch.from_numpy(fc3_mean_bias).cuda()
        elif name == "fc1_var.weight":
            param.data = torch.from_numpy(fc1_var_weight).cuda()
        elif name == "fc1_var.bias":
            param.data = torch.from_numpy(fc1_var_bias).cuda()
        elif name == "fc2_var.weight":
            param.data = torch.from_numpy(fc2_var_weight).cuda()
        elif name == "fc2_var.bias":
            param.data = torch.from_numpy(fc2_var_bias).cuda()
        elif name == "fc3_var.weight":
            param.data = torch.from_numpy(fc3_var_weight).cuda()
        elif name == "fc3_var.bias":
            param.data = torch.from_numpy(fc3_var_bias).cuda()
        else:
            raise Exception("Unknown network parameter!")
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

with open("%s/x_values.pkl" % network.model_dir, "wb") as file:
    pickle.dump(x_values, file)

with open("%s/final_mean_values.pkl" % network.model_dir, "wb") as file:
    pickle.dump(final_mean_values, file)

with open("%s/final_sigma_epi_values.pkl" % network.model_dir, "wb") as file:
    pickle.dump(final_sigma_epi_values, file)

with open("%s/final_sigma_alea_values.pkl" % network.model_dir, "wb") as file:
    pickle.dump(final_sigma_alea_values, file)

with open("%s/final_sigma_tot_values.pkl" % network.model_dir, "wb") as file:
    pickle.dump(final_sigma_tot_values, file)

# #####
# network = ToyNet("eval_HMC", project_dir="/root/evaluating_bdl/toyRegression").cuda()
# with open("/root/evaluating_bdl/toyRegression/HMC/x_values.pkl", "rb") as file: # (needed for python3)
#     x_values = pickle.load(file) # (list of 1000 elements)
# with open("/root/evaluating_bdl/toyRegression/HMC/final_mean_values.pkl", "rb") as file: # (needed for python3)
#     final_mean_values = pickle.load(file) # (list of 1000 elements)
# with open("/root/evaluating_bdl/toyRegression/HMC/final_sigma_alea_values.pkl", "rb") as file: # (needed for python3)
#     final_sigma_alea_values = pickle.load(file) # (list of 1000 elements)
# with open("/root/evaluating_bdl/toyRegression/HMC/final_sigma_epi_values.pkl", "rb") as file: # (needed for python3)
#     final_sigma_epi_values = pickle.load(file) # (list of 1000 elements)
# with open("/root/evaluating_bdl/toyRegression/HMC/final_sigma_tot_values.pkl", "rb") as file: # (needed for python3)
#     final_sigma_tot_values = pickle.load(file) # (list of 1000 elements)
# #####

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

# (fc1_mean_weight_samples has shape: (1000, 1, 10, 1))
for param_index in range(10):
    values = fc1_mean_weight_samples[:, 0, param_index, 0] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc1_mean_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc1_mean_bias_samples has shape: (1000, 1, 10))
for param_index in range(10):
    values = fc1_mean_bias_samples[:, 0, param_index] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc1_mean_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc2_mean_weight_samples has shape: (1000, 1, 10, 10))
for param_index_i in range(10):
    for param_index_j in range(10):
        values = fc2_mean_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (1000, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/fc2_mean_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc2_mean_bias_samples has shape: (1000, 1, 10))
for param_index in range(10):
    values = fc2_mean_bias_samples[:, 0, param_index] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc2_mean_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_mean_weight_samples has shape: (1000, 1, 1, 10))
for param_index in range(10):
    values = fc3_mean_weight_samples[:, 0, 0, param_index] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc3_mean_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_mean_bias_samples has shape: (1000, 1, 1))
values = fc3_mean_bias_samples[:, 0, 0] # (shape: (1000, ))
plt.figure(1)
plt.hist(np.array(values), bins=100)
plt.savefig("%s/fc3_mean_bias.png" % (network.model_dir))
plt.close(1)


# (fc1_var_weight_samples has shape: (1000, 1, 10, 1))
for param_index in range(10):
    values = fc1_var_weight_samples[:, 0, param_index, 0] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc1_var_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc1_var_bias_samples has shape: (1000, 1, 10))
for param_index in range(10):
    values = fc1_var_bias_samples[:, 0, param_index] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc1_var_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc2_var_weight_samples has shape: (1000, 1, 10, 10))
for param_index_i in range(10):
    for param_index_j in range(10):
        values = fc2_var_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (1000, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/fc2_var_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc2_var_bias_samples has shape: (1000, 1, 10))
for param_index in range(10):
    values = fc2_var_bias_samples[:, 0, param_index] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc2_var_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_var_weight_samples has shape: (1000, 1, 1, 10))
for param_index in range(10):
    values = fc3_var_weight_samples[:, 0, 0, param_index] # (shape: (1000, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/fc3_var_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_var_bias_samples has shape: (1000, 1, 1))
values = fc3_var_bias_samples[:, 0, 0] # (shape: (1000, ))
plt.figure(1)
plt.hist(np.array(values), bins=100)
plt.savefig("%s/fc3_var_bias.png" % (network.model_dir))
plt.close(1)
