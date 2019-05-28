# code-checked

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

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

        for i in range(x_train.shape[0]):
            example = {}
            example["x"] = x_train[i]
            example["y"] = y_train[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples

#_ = ToyDataset()
