# code-checked
# server-checked

from datasets import ToyDataset # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

import pyro
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc import MCMC, NUTS

from model_pyro import model, det_net

import pickle

pyro.enable_validation(True)
pyro.set_rng_seed(0)

train_dataset = ToyDataset()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset))

for step, (x, y) in enumerate(train_loader):
    x = Variable(x).cuda() # (shape: (batch_size, 2))
    y = Variable(y).cuda() # (shape: (batch_size, ))

    nuts_kernel = NUTS(model, jit_compile=False,)
    posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=1000, num_chains=1).run(x, y) # num_samples=1000, warmup_steps=1000

    fc1_weight_samples = EmpiricalMarginal(posterior, sites=["module$$$fc1.weight"])._get_samples_and_weights()[0].cpu().numpy() # (shape: (num_samples, 1, shape1, shape2))
    fc1_bias_samples = EmpiricalMarginal(posterior, sites=["module$$$fc1.bias"])._get_samples_and_weights()[0].cpu().numpy() # (shape: (num_samples, 1, shape1, shape2))

    fc2_weight_samples = EmpiricalMarginal(posterior, sites=["module$$$fc2.weight"])._get_samples_and_weights()[0].cpu().numpy() # (shape: (num_samples, 1, shape1, shape2))
    fc2_bias_samples = EmpiricalMarginal(posterior, sites=["module$$$fc2.bias"])._get_samples_and_weights()[0].cpu().numpy() # (shape: (num_samples, 1, shape1, shape2))

    fc3_weight_samples = EmpiricalMarginal(posterior, sites=["module$$$fc3.weight"])._get_samples_and_weights()[0].cpu().numpy() # (shape: (num_samples, 1, shape1, shape2))
    fc3_bias_samples = EmpiricalMarginal(posterior, sites=["module$$$fc3.bias"])._get_samples_and_weights()[0].cpu().numpy() # (shape: (num_samples, 1, shape1, shape2))

    print ("fc1_weight_samples.shape:")
    print (fc1_weight_samples.shape)
    print ("fc1_bias_samples.shape:")
    print (fc1_bias_samples.shape)
    print ("###")
    print ("fc2_weight_samples.shape:")
    print (fc2_weight_samples.shape)
    print ("fc2_bias_samples.shape:")
    print (fc2_bias_samples.shape)
    print ("###")
    print ("fc3_weight_samples.shape:")
    print (fc3_weight_samples.shape)
    print ("fc3_bias_samples.shape:")
    print (fc3_bias_samples.shape)

    with open("%s/fc1_weight_samples.pkl" % det_net.model_dir, "wb") as file:
        pickle.dump(fc1_weight_samples, file)
    with open("%s/fc1_bias_samples.pkl" % det_net.model_dir, "wb") as file:
        pickle.dump(fc1_bias_samples, file)

    with open("%s/fc2_weight_samples.pkl" % det_net.model_dir, "wb") as file:
        pickle.dump(fc2_weight_samples, file)
    with open("%s/fc2_bias_samples.pkl" % det_net.model_dir, "wb") as file:
        pickle.dump(fc2_bias_samples, file)

    with open("%s/fc3_weight_samples.pkl" % det_net.model_dir, "wb") as file:
        pickle.dump(fc3_weight_samples, file)
    with open("%s/fc3_bias_samples.pkl" % det_net.model_dir, "wb") as file:
        pickle.dump(fc3_bias_samples, file)

    # with open("%s/fc1_weight_samples.pkl" % det_net.model_dir, "rb") as file: # (needed for python3)
    #     test = pickle.load(file)
    #     print (test)
    #     print (test.shape)
