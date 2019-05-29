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

batch_size = 32

M_values = [1, 4, 16, 64, 256]
for M in M_values:
    for iter in range(6):
        network_inds = list(np.random.randint(low=0, high=1024, size=(M, )))
        print (network_inds)

        networks = []
        for i in network_inds:
            network = ToyNet("eval_Ensemble-Adam_1_M1024", project_dir="/root/evaluating_bdl/toyRegression").cuda()
            network.load_state_dict(torch.load("/root/evaluating_bdl/toyRegression/training_logs/model_Ensemble-Adam_1_M1024_%d/checkpoints/model_Ensemble-Adam_1_M1024_epoch_150.pth" % i))
            networks.append(network)

        M_float = float(len(networks))
        print (M_float)

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

        max_sigma_alea_value = -1000
        for i in range(len(x_values)):
            if (x_values[i] < 3) and (x_values[i] > -3):
                 if final_sigma_alea_values[i] > max_sigma_alea_value:
                     max_sigma_alea_value = final_sigma_alea_values[i]

        print (max_sigma_alea_value)

        for i in range(len(x_values)):
            if final_sigma_alea_values[i] > max_sigma_alea_value:
                final_sigma_alea_values[i] = max_sigma_alea_value

            final_sigma_tot_values[i] = final_sigma_alea_values[i] + final_sigma_epi_values[i]

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
        plt.savefig("%s/predictive_density_M=%d_%d.png" % (network.model_dir, M, iter+1))
        plt.close(1)
