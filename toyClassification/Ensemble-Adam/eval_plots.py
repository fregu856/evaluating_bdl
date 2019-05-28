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

x_min = -6.0
x_max = 6.0
num_points = 60

M_values = [1, 4, 16, 64, 256]
for M in M_values:
    for iter in range(6):
        network_inds = list(np.random.randint(low=0, high=1024, size=(M, )))
        print (network_inds)

        networks = []
        for i in network_inds:
            network = ToyNet("eval_Ensemble-Adam_1_M1024", project_dir="/root/evaluating_bdl/toyClassification").cuda()
            network.load_state_dict(torch.load("/root/evaluating_bdl/toyClassification/training_logs/model_Ensemble-Adam_1_M1024_%d/checkpoints/model_Ensemble-Adam_1_M1024_epoch_150.pth" % i))
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
