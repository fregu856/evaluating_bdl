# code-checked
# server-checked

import pyro
import pyro.distributions

import torch

from model_pytorch import ToyNet

det_net = ToyNet("HMC", project_dir="/root/evaluating_bdl/toyRegression").cuda()

def model(x, y):
    fc1_mean_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc1_mean.weight), scale=torch.ones_like(det_net.fc1_mean.weight))
    fc1_mean_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc1_mean.bias), scale=torch.ones_like(det_net.fc1_mean.bias))

    fc2_mean_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc2_mean.weight), scale=torch.ones_like(det_net.fc2_mean.weight))
    fc2_mean_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc2_mean.bias), scale=torch.ones_like(det_net.fc2_mean.bias))

    fc3_mean_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc3_mean.weight), scale=torch.ones_like(det_net.fc3_mean.weight))
    fc3_mean_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc3_mean.bias), scale=torch.ones_like(det_net.fc3_mean.bias))

    fc1_var_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc1_var.weight), scale=torch.ones_like(det_net.fc1_var.weight))
    fc1_var_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc1_var.bias), scale=torch.ones_like(det_net.fc1_var.bias))

    fc2_var_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc2_var.weight), scale=torch.ones_like(det_net.fc2_var.weight))
    fc2_var_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc2_var.bias), scale=torch.ones_like(det_net.fc2_var.bias))

    fc3_var_weight_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc3_var.weight), scale=torch.ones_like(det_net.fc3_var.weight))
    fc3_var_bias_prior = pyro.distributions.Normal(loc=torch.zeros_like(det_net.fc3_var.bias), scale=torch.ones_like(det_net.fc3_var.bias))

    priors = {"fc1_mean.weight": fc1_mean_weight_prior, "fc1_mean.bias": fc1_mean_bias_prior,
              "fc2_mean.weight": fc2_mean_weight_prior, "fc2_mean.bias": fc2_mean_bias_prior,
              "fc3_mean.weight": fc3_mean_weight_prior, "fc3_mean.bias": fc3_mean_bias_prior,
              "fc1_var.weight": fc1_var_weight_prior, "fc1_var.bias": fc1_var_bias_prior,
              "fc2_var.weight": fc2_var_weight_prior, "fc2_var.bias": fc2_var_bias_prior,
              "fc3_var.weight": fc3_var_weight_prior, "fc3_var.bias": fc3_var_bias_prior}

    lifted_module = pyro.random_module("module", det_net, priors)

    sampled_reg_model = lifted_module()

    mu, log_sigma_2 = sampled_reg_model(x)

    sigma = torch.sqrt(torch.exp(log_sigma_2))

    return pyro.sample("obs", pyro.distributions.Normal(mu, sigma), obs=y)
