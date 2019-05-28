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

        input_dim = 2
        hidden_dim = 10
        num_classes = 2

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # (x has shape (batch_size, input_dim))

        out = F.relu(self.fc1(x)) # (shape: (batch_size, hidden_dim))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, hidden_dim))
        out = self.fc3(out) # (shape: batch_size, num_classes))

        return out

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
