# code-checked

import torch
import torch.nn as nn

class MaskedL2(nn.Module):
    def __init__(self):
        super(MaskedL2, self).__init__()

        print ("MaskedL2")

    def forward(self, preds, targets):
        # (preds has shape: (batch_size, 1, h, w))
        # (targets has shape: (batch_size, h, w))

        targets = torch.unsqueeze(targets, 1) # (shape: (batch_size, 1, h, w))

        valid_mask = (targets > 0).detach()

        diffs = targets - preds
        diffs = diffs[valid_mask]

        loss = torch.mean(torch.pow(diffs, 2))

        return loss

class MaskedL2Gauss(nn.Module):
    def __init__(self):
        super(MaskedL2Gauss, self).__init__()

        print ("MaskedL2Gauss")

    def forward(self, means, log_vars, targets):
        # (means has shape: (batch_size, 1, h, w))
        # (log_vars has shape: (batch_size, 1, h, w))
        # (targets has shape: (batch_size, h, w))

        targets = torch.unsqueeze(targets, 1) # (shape: (batch_size, 1, h, w))

        valid_mask = (targets > 0).detach()

        targets = targets[valid_mask]
        means = means[valid_mask]
        log_vars = log_vars[valid_mask]

        loss = torch.mean(torch.exp(-log_vars)*torch.pow(targets - means, 2) + log_vars)

        return loss

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, preds, targets):
        # (preds has shape: (batch_size, 1, h, w))
        # (targets has shape: (batch_size, h, w))

        targets = torch.unsqueeze(targets, 1) # (shape: (batch_size, 1, h, w))

        valid_mask = targets > 0.1

        # convert to mm:
        targets = 1000*targets
        preds = 1000*preds

        targets = targets[valid_mask]
        preds = preds[valid_mask]

        rmse = torch.sqrt(torch.mean(torch.pow(targets - preds, 2)))

        return rmse

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, preds, targets):
        # (preds has shape: (batch_size, 1, h, w))
        # (targets has shape: (batch_size, h, w))

        targets = torch.unsqueeze(targets, 1) # (shape: (batch_size, 1, h, w))

        valid_mask = targets > 0.1

        # convert to mm:
        targets = 1000*targets
        preds = 1000*preds

        targets = targets[valid_mask]
        preds = preds[valid_mask]

        mae = torch.mean(torch.abs(targets - preds))

        return mae
