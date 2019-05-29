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

network = ToyNet("eval_Deterministic", project_dir="/root/evaluating_bdl/toyRegression").cuda()
network.load_state_dict(torch.load("/root/evaluating_bdl/toyRegression/training_logs/model_Deterministic/checkpoints/model_Deterministic_epoch_150.pth"))

val_dataset = ToyDatasetEval()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

network.eval()

x_values = []
final_mean_values = []
final_sigma_tot_values = []
final_sigma_epi_values = []
final_sigma_alea_values = []
for step, (x) in enumerate(val_loader):
    x = Variable(x).cuda().unsqueeze(1) # (shape: (batch_size, 1))

    mean = network(x) # (shape: (batch_size, ))

    for i in range(x.size(0)):
        x_value = x[i].data.cpu().numpy()[0]

        mean_value = mean[i].data.cpu().numpy()[0]

        x_values.append(x_value)
        final_mean_values.append(mean_value)

plt.figure(1)
plt.plot(x_values, final_mean_values, "r")
plt.plot(x_values, np.sin(np.array(x_values)), "k")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25)
plt.xlim([-6, 6])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig("%s/predictive_density.png" % network.model_dir)
plt.savefig("%s/predictive_density.pdf" % network.model_dir, dpi=400)
plt.close(1)
