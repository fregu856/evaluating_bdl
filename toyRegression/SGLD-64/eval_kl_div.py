# code-checked
# server-checked

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

L = 64
num_epochs = L*150

num_epochs_low = int(0.75*num_epochs)
print (num_epochs_low)

batch_size = 32

max_logvar = 2.0

with open("/root/evaluating_bdl/toyRegression/HMC/x_values.pkl", "rb") as file: # (needed for python3)
    x_values_HMC = pickle.load(file) # (list of 1000 elements)

with open("/root/evaluating_bdl/toyRegression/HMC/final_mean_values.pkl", "rb") as file: # (needed for python3)
    mean_values_HMC = pickle.load(file) # (list of 1000 elements)

with open("/root/evaluating_bdl/toyRegression/HMC/final_sigma_tot_values.pkl", "rb") as file: # (needed for python3)
    sigma_squared_values_HMC = pickle.load(file) # (list of 1000 elements)

print (len(x_values_HMC))
print (len(mean_values_HMC))
print (len(sigma_squared_values_HMC))

for i in range(len(x_values_HMC)):
    if x_values_HMC[i] < -3:
        train_interval_lower_index = i+1
    elif x_values_HMC[i] < 3:
        train_interval_upper_index = i

print (x_values_HMC[train_interval_lower_index])
print (x_values_HMC[train_interval_lower_index-1])
print (x_values_HMC[train_interval_upper_index])
print (x_values_HMC[train_interval_upper_index+1])

print (train_interval_lower_index)
print (train_interval_upper_index)

num_points_HMC = float(len(x_values_HMC))
print (num_points_HMC)
print ("##############")

M_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
for M in M_values:
    print (M)

    step_size = float(num_epochs - num_epochs_low)/float(M-1)
    print (step_size)

    if (step_size < 1):
        break

    KL_p_HMC_q_total_values = []
    KL_p_HMC_q_train_values = []
    KL_p_HMC_q_train_GT_values = []
    for j in range(6):
        networks = []
        for i in range(M):
            #print (int(num_epochs - i*step_size))

            network = ToyNet("eval_SGLD-64", project_dir="/root/evaluating_bdl/toyRegression").cuda()
            network.load_state_dict(torch.load("/root/evaluating_bdl/toyRegression/training_logs/model_SGLD-64_%d/checkpoints/model_SGLD-64_%d_epoch_%d.pth" % (j+1, j+1, int(num_epochs - i*step_size))))
            networks.append(network)

        M_float = float(len(networks))
        print (M_float)

        val_dataset = ToyDatasetEval()

        num_val_batches = int(len(val_dataset)/batch_size)

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
                    mean_value += value/M_float

                sigma_epi_value = 0.0
                for value in mean_values:
                    sigma_epi_value += ((mean_value - value)**2)/M_float

                sigma_alea_value = 0.0
                for value in sigma_alea_values:
                    sigma_alea_value += value/M_float

                sigma_tot_value = sigma_epi_value + sigma_alea_value

                x_values.append(x_value)
                final_mean_values.append(mean_value)
                final_sigma_epi_values.append(sigma_epi_value)
                final_sigma_alea_values.append(sigma_alea_value)
                final_sigma_tot_values.append(sigma_tot_value)

        mean_values = final_mean_values
        sigma_squared_values = final_sigma_tot_values

        num_points = float(len(x_values))

        if num_points != num_points_HMC:
            raise Exception("Not the same number of evaluation points!")

        for i in range(len(x_values)):
            if x_values[i] != x_values_HMC[i]:
                raise Exception("Different evaluation points!")

        KL_p_HMC_q_total = np.mean(np.log(np.sqrt(np.array(sigma_squared_values))/np.sqrt(np.array(sigma_squared_values_HMC))) + (np.array(sigma_squared_values_HMC) + (np.array(mean_values_HMC) - np.array(mean_values))**2)/(2*np.array(sigma_squared_values)) - 1.0/2.0)
        KL_p_HMC_q_total_values.append(KL_p_HMC_q_total)
        #print ("KL_p_HMC_q_total: %g" % KL_p_HMC_q_total)

        KL_p_HMC_q_train = np.mean(np.log(np.sqrt(np.array(sigma_squared_values[train_interval_lower_index:train_interval_upper_index]))/np.sqrt(np.array(sigma_squared_values_HMC[train_interval_lower_index:train_interval_upper_index]))) + (np.array(sigma_squared_values_HMC[train_interval_lower_index:train_interval_upper_index]) + (np.array(mean_values_HMC[train_interval_lower_index:train_interval_upper_index]) - np.array(mean_values[train_interval_lower_index:train_interval_upper_index]))**2)/(2*np.array(sigma_squared_values[train_interval_lower_index:train_interval_upper_index])) - 1.0/2.0)
        KL_p_HMC_q_train_values.append(KL_p_HMC_q_train)
        #print ("KL_p_HMC_q_train: %g" % KL_p_HMC_q_train)

        KL_p_HMC_q_train_GT = np.mean(np.log(np.sqrt(np.array(sigma_squared_values[train_interval_lower_index:train_interval_upper_index]))/np.array(0.15*(1.0/(1 + np.exp(-np.array(x_values[train_interval_lower_index:train_interval_upper_index])))))) + (np.array(0.15*(1.0/(1 + np.exp(-np.array(x_values[train_interval_lower_index:train_interval_upper_index])))))**2 + (np.array(np.sin(np.array(x_values[train_interval_lower_index:train_interval_upper_index]))) - np.array(mean_values[train_interval_lower_index:train_interval_upper_index]))**2)/(2*np.array(sigma_squared_values[train_interval_lower_index:train_interval_upper_index])) - 1.0/2.0)
        KL_p_HMC_q_train_GT_values.append(KL_p_HMC_q_train_GT)
        #print ("KL_p_HMC_q_train_GT: %g" % KL_p_HMC_q_train_GT)

    print ("mean_total: %g" % np.mean(np.array(KL_p_HMC_q_total_values)))
    print ("std_total: %g" % np.std(np.array(KL_p_HMC_q_total_values)))
    print ("max_total: %g" % np.max(np.array(KL_p_HMC_q_total_values)))
    print ("min_total: %g" % np.min(np.array(KL_p_HMC_q_total_values)))

    print ("mean_train: %g" % np.mean(np.array(KL_p_HMC_q_train_values)))
    print ("std_train: %g" % np.std(np.array(KL_p_HMC_q_train_values)))
    print ("max_train: %g" % np.max(np.array(KL_p_HMC_q_train_values)))
    print ("min_train: %g" % np.min(np.array(KL_p_HMC_q_train_values)))

    print ("mean_train_GT: %g" % np.mean(np.array(KL_p_HMC_q_train_GT_values)))
    print ("std_train_GT: %g" % np.std(np.array(KL_p_HMC_q_train_GT_values)))
    print ("max_train_GT: %g" % np.max(np.array(KL_p_HMC_q_train_GT_values)))
    print ("min_train_GT: %g" % np.min(np.array(KL_p_HMC_q_train_GT_values)))

    print (M)

    print ("########################")
