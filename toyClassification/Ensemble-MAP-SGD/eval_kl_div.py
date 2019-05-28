# code-checked

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

x_min = -6.0
x_max = 6.0
num_points = 60

epsilon = 1.0e-30

with open("/root/evaluating_bdl/toyClassification/HMC/false_prob_values.pkl", "rb") as file: # (needed for python3)
    false_prob_values_HMC = pickle.load(file) # (shape: (60, 60))
print (false_prob_values_HMC.shape)
print (np.max(false_prob_values_HMC))
print (np.min(false_prob_values_HMC))

p_HMC = false_prob_values_HMC/np.sum(false_prob_values_HMC)

x_values = np.linspace(x_min, x_max, num_points, dtype=np.float32)

x_1_train_lower = 0 # (0)
x_1_train_upper = 0 # (3)
x_2_train_lower = 0 # (-3)
x_2_train_upper = 0 # (3)
for index, value in enumerate(x_values):
    if value < 0:
        x_1_train_lower = index+1

    if value < 3:
        x_1_train_upper = index
        x_2_train_upper = index

    if value < -3:
        x_2_train_lower = index+1

print (x_1_train_lower)
print (x_values[x_1_train_lower])
print (x_1_train_upper)
print (x_values[x_1_train_upper])
print (x_2_train_lower)
print (x_values[x_2_train_lower])
print (x_2_train_upper)
print (x_values[x_2_train_upper])

p_HMC_train = p_HMC[x_2_train_lower:x_2_train_upper, x_1_train_lower:x_1_train_upper] # (shape: (29, 14))
p_HMC_train = p_HMC_train/np.sum(p_HMC_train)

M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
for M in M_values:
    print (M)

    L = int(1024/M)
    if L > 20:
        L = 20
    print (L)

    KL_p_HMC_q_total_values = []
    KL_p_HMC_q_train_values = []
    for l in range(L):
        networks = []
        for i in range(M):
            if (l*M + i) % 100 == 0:
                print (l*M + i)

            network = ToyNet("eva_Ensemble-MAP-SGD_1_M1024", project_dir="/root/evaluating_bdl/toyClassification").cuda()
            network.load_state_dict(torch.load("/root/evaluating_bdl/toyClassification/training_logs/model_Ensemble-MAP-SGD_1_M1024_%d/checkpoints/model_Ensemble-MAP-SGD_1_M1024_epoch_150.pth" % (l*M + i)))
            networks.append(network)

        M_float = float(len(networks))
        # print (M_float)

        for network in networks:
            network.eval()

        false_prob_values = np.zeros((num_points, num_points))
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

        # print (false_prob_values.shape)
        # print (np.max(false_prob_values))
        # print (np.min(false_prob_values))

        q = false_prob_values/np.sum(false_prob_values)

        KL_p_HMC_q_total = np.sum(p_HMC*np.log(p_HMC/(q + epsilon) + epsilon))
        KL_p_HMC_q_total_values.append(KL_p_HMC_q_total)
        #print ("KL_p_HMC_q_total: %g" % KL_p_HMC_q_total)

        q_train = q[x_2_train_lower:x_2_train_upper, x_1_train_lower:x_1_train_upper]
        q_train = q_train/np.sum(q_train)

        KL_p_HMC_q_train = np.sum(p_HMC_train*np.log(p_HMC_train/(q_train + epsilon) + epsilon))
        KL_p_HMC_q_train_values.append(KL_p_HMC_q_train)
        #print ("KL_p_HMC_q_train: %g" % KL_p_HMC_q_train)

    print ("mean_total: %g" % np.mean(np.array(KL_p_HMC_q_total_values)))
    print ("std_total: %g" % np.std(np.array(KL_p_HMC_q_total_values)))
    print ("max_total: %g" % np.max(np.array(KL_p_HMC_q_total_values)))
    print ("min_total: %g" % np.min(np.array(KL_p_HMC_q_total_values)))
    print ("###")

    print ("mean_train: %g" % np.mean(np.array(KL_p_HMC_q_train_values)))
    print ("std_train: %g" % np.std(np.array(KL_p_HMC_q_train_values)))
    print ("max_train: %g" % np.max(np.array(KL_p_HMC_q_train_values)))
    print ("min_train: %g" % np.min(np.array(KL_p_HMC_q_train_values)))

    print (M)

    print ("########################")
