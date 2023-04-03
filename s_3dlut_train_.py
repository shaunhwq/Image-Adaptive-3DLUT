import os
import math
import itertools
import time
import datetime
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets

import models_x
import s_transforms
from datasets import get_distortion_dataset


# Note: Copy the transforms file to this repo first.


# ==== TRAINING SETTINGS ====
train_name = "0001"
# Dataloader settings
dataset_root = ""
if dataset_root == "":
    sys.exit("Set path to dataset root please")
batch_size = 1
dataloader_num_workers = 16
# Epochs related
num_epochs = 400
checkpoint_interval = 10
evaluate_interval = 5
# Loss Settings
loss_params = dict(lambda_smooth=1e-4, lambda_monotonicity=10.0)
# Optimizer Settings
adam_params = dict(lr=1e-4, betas=(0.9, 0.999))

distortion_transforms = s_transforms.RandomChoice([
    s_transforms.RandomFog(p=1.0, alpha=0.5, beta_min=0.005, beta_max=0.05),
    s_transforms.RandomDistortContrast(p=1.0),
    s_transforms.RandomAdjustPowerLawContrast(p=1.0, l_bound=0.5, h_bound=0.9),
])
train_transforms = s_transforms.Compose([
    # s_transforms.ShortsideResize(min_size=256),  # Shouldn't need to resize the image because of classifier
    # s_transforms.RandomRotate(p=0.5, angle=20),
    # s_transforms.RandomGaussianNoise(p=0.5, mean=0, std=0.1),
    # s_transforms.RandomImageCompression(p=0.5, low=50, high=100),
    # s_transforms.RandomHorizontalFlip(p=0.5),
    s_transforms.CvtColor(cv2.COLOR_BGR2RGB),
    s_transforms.ToTensor(),
    # s_transforms.Normalize(mean=[0.485, 0.406, 0.456], std=[0.229, 0.225, 0.224]),
])
test_transforms = s_transforms.Compose([
    # s_transforms.ShortsideResize(min_size=256),  # Shouldn't need to resize the image because of classifier
    s_transforms.CvtColor(cv2.COLOR_BGR2RGB),
    s_transforms.ToTensor(),
    # s_transforms.Normalize(mean=[0.485, 0.406, 0.456], std=[0.229, 0.225, 0.224])
])

# ==================================

# Folder creation
output_dir = os.path.join(os.getcwd(), "run", train_name)
checkpoints_folder = os.path.join(output_dir, "checkpoints")
best_weight_folder = os.path.join(output_dir, "best")
os.makedirs(checkpoints_folder, exist_ok=True)
os.makedirs(best_weight_folder, exist_ok=True)

# Retrieve dataloader
train_dataloader, test_dataloader = get_distortion_dataset(
    dataset_root=dataset_root,
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    distortion_transform=distortion_transforms,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
)

# Tensor type
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_pixelwise = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()  # just to monitor

# Model Initialization
LUT0 = models_x.Generator3DLUT_identity()
LUT1 = models_x.Generator3DLUT_zero()
LUT2 = models_x.Generator3DLUT_zero()
classifier = models_x.Classifier()
TV3 = models_x.TV_3D()
trilinear_ = models_x.TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

# Initialize weights
classifier.apply(models_x.weights_init_normal_classifier)
torch.nn.init.constant_(classifier.model[16].bias.data, 1.0)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()), **adam_params)


def generator_train(img):
    pred = classifier(img).squeeze()
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A0 = LUT0(img)
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)

    weights_norm = torch.mean(pred ** 2)

    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        combine_A[b,:,:,:] = pred[b,0] * gen_A0[b,:,:,:] + pred[b,1] * gen_A1[b,:,:,:] + pred[b,2] * gen_A2[b,:,:,:]

    return combine_A, weights_norm


def generator_eval(img):
    pred = classifier(img).squeeze()
    
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT
   
    weights_norm = torch.mean(pred ** 2)

    combine_A = img.new(img.size())
    combine_A = trilinear_(LUT,img)[1]

    return combine_A, weights_norm


prev_time = time.time()
train_scores = {"mse": [], "l1": [], "compound": [], "batch_no":[]}
test_scores = {"mse": [], "l1": [], "compound": [], "batch_no": []}

lowest_loss = float("inf")
for epoch in range(num_epochs):
    classifier.train()
    LUT0.train()
    LUT1.train()
    LUT2.train()

    train_mse_sum, train_l1_sum, train_compound_sum = 0.0, 0.0, 0.0
    for train_batch_idx, (train_tensors, train_target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Train", leave=False):
        train_tensor = train_tensors[0]
        train_tensor = Variable(train_tensor.type(Tensor))
        train_target = Variable(train_target.type(Tensor))

        optimizer_G.zero_grad()

        train_pred, weights_norm = generator_train(train_tensor)

        # Loss computation
        mse = criterion_pixelwise(train_pred, train_target)
        tv0, mn0 = TV3(LUT0)
        tv1, mn1 = TV3(LUT1)
        tv2, mn2 = TV3(LUT2)
        tv_cons = tv0 + tv1 + tv2
        mn_cons = mn0 + mn1 + mn2
        loss = mse + loss_params["lambda_smooth"] * (weights_norm + tv_cons) + loss_params["lambda_monotonicity"] * mn_cons

        loss.backward()
        optimizer_G.step()

        l1_loss_value = l1_loss(train_pred, train_target)
        
        train_mse_sum += mse.item()
        train_l1_sum += l1_loss_value.item()
        train_compound_sum += loss.item()

    # Training loss report
    train_scores["batch_no"].append(epoch)
    train_scores["l1"].append(train_l1_sum / len(train_dataloader))
    train_scores["mse"].append(train_mse_sum / len(train_dataloader))
    train_scores["compound"].append(train_compound_sum / len(train_dataloader))
    train_scores_formatted = [round(train_scores[name][-1], 5) for name in ["l1", "mse", "compound"]]
    print("[TRN] Epoch: {} | L1: {} | MSE: {} | Compound: {} |".format(epoch, *train_scores_formatted))

    # Test
    if epoch % evaluate_interval == 0:
        LUT0.eval()
        LUT1.eval()
        LUT2.eval()
        classifier.eval()
        tst_l1_sum, tst_mse_sum, tst_compound_sum = 0.0, 0.0, 0.0
        for test_batch_idx, (test_tensors, test_target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Test", leave=False):
            test_tensor = test_tensors[0]
            test_tensor = Variable(test_tensor.type(Tensor))
            test_target = Variable(test_target.type(Tensor))

            with torch.no_grad():
                test_pred, weights_norm = generator_eval(test_tensor)
            
            tst_mse = criterion_pixelwise(test_pred, test_target)
            tst_tv0, tst_mn0 = TV3(LUT0)
            tst_tv1, tst_mn1 = TV3(LUT1)
            tst_tv2, tst_mn2 = TV3(LUT2)
            tst_tv_cons = tst_tv0 + tst_tv1 + tst_tv2
            tst_mn_cons = tst_mn0 + tst_mn1 + tst_mn2
            tst_loss = tst_mse + loss_params["lambda_smooth"] * (weights_norm + tst_tv_cons) + loss_params["lambda_monotonicity"] * tst_mn_cons

            tst_l1_loss_value = l1_loss(test_pred, test_target)

            tst_l1_sum += tst_l1_loss_value.item()
            tst_mse_sum += tst_mse.item()
            tst_compound_sum += tst_loss.item()

        # Test report
        test_scores["batch_no"].append(epoch)
        test_scores["l1"].append(tst_l1_sum / len(test_dataloader))
        test_scores["mse"].append(tst_mse_sum / len(test_dataloader))
        test_scores["compound"].append(tst_compound_sum / len(test_dataloader))

        test_scores_formatted = [round(test_scores[name][-1], 5) for name in ["l1", "mse", "compound"]]
        print("[TST] Epoch: {} | L1: {} | MSE: {} | Compound: {} |".format(epoch, *test_scores_formatted))

        # Save weights if test loss is lesser than lowest
        if test_scores["compound"][-1] < lowest_loss:
            print(f"Saving weights... Lowest loss: {round(test_scores['compound'][-1], 4)}")
            lowest_loss = test_scores['compound'][-1]
            LUTs = {"0": LUT0.state_dict(),"1": LUT1.state_dict(),"2": LUT2.state_dict()}
            torch.save(LUTs, os.path.join(best_weight_folder, f"best_LUTs.pth"))
            torch.save(classifier.state_dict(), os.path.join(best_weight_folder, f"best_classifier.pth"))

    # Save model if checkpoint
    if epoch % checkpoint_interval == 0:
        print("Model Checkpoint.")
        LUTs = {"0": LUT0.state_dict(),"1": LUT1.state_dict(),"2": LUT2.state_dict()}
        torch.save(LUTs, os.path.join(checkpoints_folder, f"ckpt_{epoch}_LUTs.pth"))
        torch.save(classifier.state_dict(), os.path.join(checkpoints_folder, f"ckpt_{epoch}_classifier.pth"))

    # Visualize training curves
    for key in [key for key in train_scores.keys() if key != "batch_no"]:
        plt.figure()
        plt.plot(train_scores["batch_no"], train_scores[key], label="Train")
        plt.plot(test_scores["batch_no"], test_scores[key], label="Test")
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{key}.png"))
        plt.close("all")

    # Save records
    pd.DataFrame(train_scores).to_csv(os.path.join(output_dir, f"train_scores.csv"))
    pd.DataFrame(test_scores).to_csv(os.path.join(output_dir, f"test_scores.csv"))

    # === Logging ===
    # Determine approximate time left
    epochs_left = num_epochs - epoch
    time_left = datetime.timedelta(seconds=epochs_left * (time.time() - prev_time))
    prev_time = time.time()