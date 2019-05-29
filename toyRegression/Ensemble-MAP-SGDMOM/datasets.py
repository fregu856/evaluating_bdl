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

        with open("/root/evaluating_bdl/toyRegression/x.pkl", "rb") as file: # (needed for python3)
            x = pickle.load(file)

        with open("/root/evaluating_bdl/toyRegression/y.pkl", "rb") as file: # (needed for python3)
            y = pickle.load(file)

        plt.figure(1)
        plt.plot(x, y, "k.")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.savefig("/root/evaluating_bdl/toyRegression/Ensemble-MAP-SGDMOM/training_data.png")
        plt.close(1)

        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i]
            example["y"] = y[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples

class ToyDatasetEval(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

        x = np.linspace(-7, 7, 1000, dtype=np.float32)

        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]

        return (x)

    def __len__(self):
        return self.num_examples

# _ = ToyDataset()
