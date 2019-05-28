import pyro
import pyro.distributions

import torch
import torch.nn.functional as F

from model_pytorch import ToyNet

det_net = ToyNet("ToyClass-HMC", project_dir="/root/bnns").cuda()

def model(x, y):
    fc1_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc1.weight), scale=torch.ones_like(det_net.fc1.weight))
    fc1_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc1.bias), scale=torch.ones_like(det_net.fc1.bias))

    fc2_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc2.weight), scale=torch.ones_like(det_net.fc2.weight))
    fc2_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc2.bias), scale=torch.ones_like(det_net.fc2.bias))

    fc3_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc3.weight), scale=torch.ones_like(det_net.fc3.weight))
    fc3_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc3.bias), scale=torch.ones_like(det_net.fc3.bias))

    priors = {"fc1.weight": fc1_weight_prior, "fc1.bias": fc1_bias_prior,
              "fc2.weight": fc2_weight_prior, "fc2.bias": fc2_bias_prior,
              "fc3.weight": fc3_weight_prior, "fc3.bias": fc3_bias_prior}

    lifted_module = pyro.random_module("module", det_net, priors)

    sampled_reg_model = lifted_module()

    logits = sampled_reg_model(x)

    return pyro.sample("obs", pyro.distributions.Categorical(logits=logits), obs=y)
