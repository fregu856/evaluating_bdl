# code-checked
# server-checked

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import os

class ToyNet(nn.Module):
    def __init__(self, model_id, project_dir):
        super(ToyNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        input_dim = 1
        hidden_dim = 10
        output_dim = 1

        self.fc1_mean = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, output_dim)

        self.fc1_var = nn.Linear(input_dim, hidden_dim)
        self.fc2_var = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (x has shape (batch_size, input_dim))

        mean = F.relu(self.fc1_mean(x)) # (shape: (batch_size, hidden_dim))
        mean = F.relu(self.fc2_mean(mean)) # (shape: (batch_size, hidden_dim))
        mean = self.fc3_mean(mean) # (shape: batch_size, output_dim))

        var = F.relu(self.fc1_var(x)) # (shape: (batch_size, hidden_dim))
        var = F.relu(self.fc2_var(var)) # (shape: (batch_size, hidden_dim))
        var = self.fc3_var(var) # (shape: batch_size, output_dim))

        return (mean, var)

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
