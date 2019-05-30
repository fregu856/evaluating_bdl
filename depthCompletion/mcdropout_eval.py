# code-checked
# server-checked

import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from model_mcdropout import DepthCompletionNet

from datasets import DatasetKITTIVal
from criterion import MaskedL2Gauss, RMSE

import numpy as np
import cv2

model_id = "mcdropout_virtual_0"
M = 4

snapshot_dir = "/root/evaluating_bdl/depthCompletion/training_logs/%s_eval" % model_id

kitti_depth_path = "/root/data/kitti_depth"

batch_size = 4

if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

restore_from = "/root/evaluating_bdl/depthCompletion/trained_models/%s/checkpoint_40000.pth" % model_id
model = DepthCompletionNet().cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(restore_from))
model.eval()

M_float = float(M)
print (M_float)

eval_dataset = DatasetKITTIVal(kitti_depth_path=kitti_depth_path)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

criterion = MaskedL2Gauss().cuda()
rmse_criterion = RMSE().cuda()

batch_losses = []
batch_rmses = []
for i_iter, batch in enumerate(eval_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs, sparses, targets, file_ids = batch
        imgs = Variable(imgs.cuda()) # (shape: (batch_size, h, w))
        sparses = Variable(sparses.cuda()) # (shape: (batch_size, h, w))
        targets = Variable(targets.cuda()) # (shape: (batch_size, h, w))

        means = []
        sigma_2_aleas = []
        for i in range(M):
            mean, log_var = model(imgs, sparses) # (both of shape: (batch_size, 1, h, w))

            sigma_2_alea = torch.exp(log_var) # (sigma_alea^2) # (shape: (batch_size, 1, h, w))

            means.append(mean)
            sigma_2_aleas.append(sigma_2_alea)

        mean = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w))
        for value in means:
            mean = mean + value/M_float

        sigma_2_alea = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w)) (sigma_alea^2)
        for value in sigma_2_aleas:
            sigma_2_alea = sigma_2_alea + value/M_float

        sigma_2_epi = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w)) (sigma_epi^2)
        for value in means:
            sigma_2_epi = sigma_2_epi + torch.pow(mean - value, 2)/M_float

        sigma_2_pred = sigma_2_alea + sigma_2_epi # (sigma_pred^2)

        loss = criterion(mean, torch.log(sigma_2_pred), targets)
        rmse = rmse_criterion(mean, targets)

        print('iter = {}/{} completed, loss = {}, rmse = {}'.format(i_iter, len(eval_dataset)/batch_size, loss.data.cpu().numpy(), rmse.data.cpu().numpy()))

        batch_losses.append(loss.data.cpu().numpy())
        batch_rmses.append(rmse.data.cpu().numpy())

        ########################################################################
        # visualization:
        ########################################################################
        mean = mean.data.cpu().numpy() # (shape: (batch_size, 1, h, w))
        sigma_2_alea = sigma_2_alea.data.cpu().numpy() # (shape: (batch_size, 1, h, w))
        sigma_2_epi = sigma_2_epi.data.cpu().numpy() # (shape: (batch_size, 1, h, w))
        sigma_2_pred = sigma_2_pred.data.cpu().numpy() # (shape: (batch_size, 1, h, w))
        targets = targets.data.cpu().numpy() # (shape: (batch_size, h, w))
        imgs = imgs.data.cpu().numpy() # (shape: (batch_size, h, w))
        sparses = sparses.data.cpu().numpy() # (shape: (batch_size, h, w))

        for i in range(mean.shape[0]):
            if i == 0:
                file_id = file_ids[i] # (file_id == "2011_09_26_drive_0036_sync_image_0000000752_image_02.png" (e.g.))
                file_id = file_id.split(".png")[0]  # (file_id == "2011_09_26_drive_0036_sync_image_0000000752_image_02"

                pred = mean[i] # (shape: (1, h, w))
                pred = pred.squeeze(0) # (shape: (h, w))

                sigma_2_alea_ = sigma_2_alea[i] # (shape: (1, h, w))
                sigma_2_alea_ = sigma_2_alea_.squeeze(0) # (shape: (h, w))
                sigma_alea = np.sqrt(sigma_2_alea_)

                sigma_2_epi_ = sigma_2_epi[i] # (shape: (1, h, w))
                sigma_2_epi_ = sigma_2_epi_.squeeze(0) # (shape: (h, w))
                sigma_epi = np.sqrt(sigma_2_epi_)

                sigma_2_pred_ = sigma_2_pred[i] # (shape: (1, h, w))
                sigma_2_pred_ = sigma_2_pred_.squeeze(0) # (shape: (h, w))
                sigma_pred = np.sqrt(sigma_2_pred_)

                img = imgs[i] # (shape: (h, w))
                img = img.astype(np.uint8)

                max_distance = 65.0

                target = targets[i] # (shape: (h, w))
                target[target > max_distance] = max_distance
                target = (target/max_distance)*255
                target = target.astype(np.uint8)

                sparse = sparses[i] # (shape: (h, w))
                sparse[sparse > max_distance] = max_distance
                sparse = (sparse/max_distance)*255
                sparse = sparse.astype(np.uint8)

                pred[pred > max_distance] = max_distance
                pred = (pred/max_distance)*255
                pred = pred.astype(np.uint8)

                sparse_color = cv2.applyColorMap(sparse, cv2.COLORMAP_SUMMER)
                sparse_color[sparse == 0] = 0

                target_color = cv2.applyColorMap(target, cv2.COLORMAP_SUMMER)
                target_color[target == 0] = 0

                pred_color = cv2.applyColorMap(pred, cv2.COLORMAP_SUMMER)

                max_interval_length = 75.0 # (corresponds to the maximum length of a 95% conf interval)
                max_sigma = max_interval_length/(2.0*1.96)

                sigma_alea[sigma_alea > max_sigma] = max_sigma
                sigma_alea = (sigma_alea/max_sigma)*255
                sigma_alea = sigma_alea.astype(np.uint8)
                sigma_alea_color = cv2.applyColorMap(sigma_alea, cv2.COLORMAP_HOT)

                sigma_epi[sigma_epi > max_sigma] = max_sigma
                sigma_epi = (sigma_epi/max_sigma)*255
                sigma_epi = sigma_epi.astype(np.uint8)
                sigma_epi_color = cv2.applyColorMap(sigma_epi, cv2.COLORMAP_HOT)

                sigma_pred[sigma_pred > max_sigma] = max_sigma
                sigma_pred = (sigma_pred/max_sigma)*255
                sigma_pred = sigma_pred.astype(np.uint8)
                sigma_pred_color = cv2.applyColorMap(sigma_pred, cv2.COLORMAP_HOT)

                cv2.imwrite(snapshot_dir + "/" + file_id + "_img.png", img)
                cv2.imwrite(snapshot_dir + "/" + file_id + "_sparse_color.png", sparse_color)
                cv2.imwrite(snapshot_dir + "/" + file_id + "_target_color.png", target_color)
                cv2.imwrite(snapshot_dir + "/" + file_id + "_pred_color.png", pred_color)
                cv2.imwrite(snapshot_dir + "/" + file_id + "_sigma_alea_color.png", sigma_alea_color)
                cv2.imwrite(snapshot_dir + "/" + file_id + "_sigma_epi_color.png", sigma_epi_color)
                cv2.imwrite(snapshot_dir + "/" + file_id + "_sigma_pred_color.png", sigma_pred_color)

        # # # # # # # # # # # # # # # # # # debug START:
        # if i_iter > 0:
        #     break
        # # # # # # # # # # # # # # # # # # debug END:

val_loss = np.mean(batch_losses)
print ("val loss: %g" % val_loss)
val_rmse = np.mean(batch_rmses)
print ("val rmse: %g" % val_rmse)
