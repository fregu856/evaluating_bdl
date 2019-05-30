# code-checked
# sever-checked

import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from model import DepthCompletionNet

from datasets import DatasetKITTIValSeq
from criterion import MaskedL2Gauss, RMSE

import numpy as np
import cv2

model_id = "ensembling_virtual"
model_is = [0, 1, 2, 3, 4, 5, 6, 7]
print (model_is)

snapshot_dir = "/root/evaluating_bdl/depthCompletion/training_logs/%s_%s_eval_seq" % (model_id, str(model_is))

kitti_depth_path = "/root/data/kitti_depth"
kitti_raw_path = "/root/data/kitti_raw"

batch_size = 4

models = []
for i in model_is:
    restore_from = "/root/evaluating_bdl/depthCompletion/trained_models/%s_%d/checkpoint_40000.pth" % (model_id, i)
    model = DepthCompletionNet().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(restore_from))
    model.eval()
    models.append(model)

M = float(len(models))
print (M)

criterion = MaskedL2Gauss().cuda()
rmse_criterion = RMSE().cuda()

val_sequences = ["2011_09_26_drive_0002", "2011_09_26_drive_0005", "2011_09_26_drive_0013", "2011_09_26_drive_0020", "2011_09_26_drive_0023", "2011_09_26_drive_0036", "2011_09_26_drive_0079", "2011_09_26_drive_0095", "2011_09_26_drive_0113", "2011_09_28_drive_0037", "2011_09_29_drive_0026", "2011_09_30_drive_0016", "2011_10_03_drive_0047"]
for step, seq in enumerate(val_sequences):
    print ("##################################################################")
    print ("seq: %d/%d, %s" % (step+1, len(val_sequences), seq))

    snapshot_dir_seq = snapshot_dir + "/" + seq

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(snapshot_dir_seq):
        os.makedirs(snapshot_dir_seq)

    eval_dataset = DatasetKITTIValSeq(kitti_depth_path=kitti_depth_path, kitti_raw_path=kitti_raw_path, seq=seq)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    batch_losses = []
    batch_rmses = []
    for i_iter, batch in enumerate(eval_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs, sparses, targets, file_ids, imgs_color = batch
            imgs = Variable(imgs.cuda()) # (shape: (batch_size, h, w))
            sparses = Variable(sparses.cuda()) # (shape: (batch_size, h, w))
            targets = Variable(targets.cuda()) # (shape: (batch_size, h, w))

            means = []
            sigma_2_aleas = []
            for model in models:
                mean, log_var = model(imgs, sparses) # (both of shape: (batch_size, 1, h, w))

                sigma_2_alea = torch.exp(log_var) # (sigma_alea^2) # (shape: (batch_size, 1, h, w))

                means.append(mean)
                sigma_2_aleas.append(sigma_2_alea)

            mean = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w))
            for value in means:
                mean = mean + value/M

            sigma_2_alea = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w)) (sigma_alea^2)
            for value in sigma_2_aleas:
                sigma_2_alea = sigma_2_alea + value/M

            sigma_2_epi = torch.zeros(means[0].size()).cuda() # (shape: (batch_size, 1, h, w)) (sigma_epi^2)
            for value in means:
                sigma_2_epi = sigma_2_epi + torch.pow(mean - value, 2)/M

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

            imgs_color = imgs_color.numpy()

            for i in range(mean.shape[0]):
                file_id = file_ids[i] # (file_id == "0000000005.png" (e.g.))
                file_id = file_id.split(".png")[0]  # (file_id == "0000000005")

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

                img_color = imgs_color[i]
                img_color = img_color.astype(np.uint8)

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

                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_img.png", img)
                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_img_color.png", img_color)
                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_sparse_color.png", sparse_color)
                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_target_color.png", target_color)
                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_pred_color.png", pred_color)
                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_sigma_alea_color.png", sigma_alea_color)
                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_sigma_epi_color.png", sigma_epi_color)
                cv2.imwrite(snapshot_dir_seq + "/" + file_id + "_sigma_pred_color.png", sigma_pred_color)

    val_loss = np.mean(batch_losses)
    print ("val loss: %g" % val_loss)
    val_rmse = np.mean(batch_rmses)
    print ("val rmse: %g" % val_rmse)

    img_h = 352
    img_w = 1216

    colorbar_w = 30
    colorbar_row = np.linspace(start=255.0, stop=0.0, num=img_h) # (shape: (img_h, ))
    colorbar = np.zeros((colorbar_w, img_h)) # (shape: (colorbar_w, img_h)
    colorbar = colorbar + colorbar_row
    colorbar = colorbar.T # (shape: (img_h, colorbar_w)
    colorbar = colorbar.astype(np.uint8)
    colorbar_SUMMER = cv2.applyColorMap(colorbar, cv2.COLORMAP_SUMMER) # (shape: (img_h, colorbar_w, 3)
    colorbar_HOT = cv2.applyColorMap(colorbar, cv2.COLORMAP_HOT) # (shape: (img_h, colorbar_w, 3)

    ids = eval_dataset.ids # (contains e.g. "0000000005.png" and so on)
    ids_sorted = sorted(ids)

    out = cv2.VideoWriter("%s/%s.avi" % (snapshot_dir_seq, seq), cv2.VideoWriter_fourcc(*"MJPG"), 12, (2*(img_w + colorbar_w), 4*img_h))
    for step, id in enumerate(ids_sorted):
        if step % 10 == 0:
            print ("step: %d/%d" % (step+1, len(ids)))

        # (id == "0000000005.png" e.g.)
        id = id.split(".png")[0]  # (id == "0000000005")

        img_color = cv2.imread(snapshot_dir_seq + "/" + id + "_img_color.png", -1) # (shape: (img_h, img_w, 3))
        sparse_color = cv2.imread(snapshot_dir_seq + "/" + id + "_sparse_color.png", -1) # (shape: (img_h, img_w, 3))
        target_color = cv2.imread(snapshot_dir_seq + "/" + id + "_target_color.png", -1) # (shape: (img_h, img_w, 3))
        pred_color = cv2.imread(snapshot_dir_seq + "/" + id + "_pred_color.png", -1) # (shape: (img_h, img_w, 3))
        sigma_alea_color = cv2.imread(snapshot_dir_seq + "/" + id + "_sigma_alea_color.png", -1) # (shape: (img_h, img_w, 3))
        sigma_epi_color = cv2.imread(snapshot_dir_seq + "/" + id + "_sigma_epi_color.png", -1) # (shape: (img_h, img_w, 3))
        sigma_pred_color = cv2.imread(snapshot_dir_seq + "/" + id + "_sigma_pred_color.png", -1) # (shape: (img_h, img_w, 3))

        combined_img = np.zeros((4*img_h, 2*(img_w + colorbar_w), 3), dtype=np.uint8)
        #
        combined_img[0:img_h, 0:img_w] = img_color
        combined_img[0:img_h, (img_w + colorbar_w):(2*img_w + colorbar_w)] = sparse_color
        combined_img[0:img_h, (2*img_w + colorbar_w):(2*img_w + 2*colorbar_w)] = colorbar_SUMMER
        #
        combined_img[img_h:(2*img_h), 0:img_w] = target_color
        combined_img[img_h:(2*img_h), img_w:(img_w + colorbar_w)] = colorbar_SUMMER
        combined_img[img_h:(2*img_h), (img_w + colorbar_w):(2*img_w + colorbar_w)] = pred_color
        combined_img[img_h:(2*img_h), (2*img_w + colorbar_w):(2*img_w + 2*colorbar_w)] = colorbar_SUMMER
        #
        combined_img[(2*img_h):(3*img_h), int(img_w+colorbar_w - (img_w+colorbar_w)/2):int(img_w+colorbar_w - (img_w+colorbar_w)/2 + img_w)] = sigma_pred_color
        combined_img[(2*img_h):(3*img_h), int(img_w+colorbar_w - (img_w+colorbar_w)/2 + img_w):int(img_w+colorbar_w - (img_w+colorbar_w)/2 + img_w + colorbar_w)] = colorbar_HOT
        #
        combined_img[(3*img_h):(4*img_h), 0:img_w] = sigma_alea_color
        combined_img[(3*img_h):(4*img_h), img_w:(img_w + colorbar_w)] = colorbar_HOT
        combined_img[(3*img_h):(4*img_h), (img_w + colorbar_w):(2*img_w + colorbar_w)] = sigma_epi_color
        combined_img[(3*img_h):(4*img_h), (2*img_w + colorbar_w):(2*img_w + 2*colorbar_w)] = colorbar_HOT

        out.write(combined_img)

    out.release()
