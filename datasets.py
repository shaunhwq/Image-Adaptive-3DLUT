import glob
import random
import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision_x_functional as TF_x


class ImageDataset_sRGB(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK", combined=True):
        self.mode = mode
        self.unpaird_data = unpaird_data

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","JPG/480p",set1_input_files[i][:-1] + ".jpg"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","JPG/480p",set2_input_files[i][:-1] + ".jpg"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","JPG/480p",test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))

        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            #img_input = TF.resized_crop(img_input, i, j, h, w, (320,320))
            #img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (320,320))

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.8,1.2)
            img_input = TF.adjust_brightness(img_input,a)

            a = np.random.uniform(0.8,1.2)
            img_input = TF.adjust_saturation(img_input,a)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_XYZ(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK", combined=True):
        self.mode = mode

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))

        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

class ImageDataset_sRGB_unpaired(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK"):
        self.mode = mode
        self.unpaird_data = unpaird_data

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","JPG/480p",set1_input_files[i][:-1] + ".jpg"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","JPG/480p",set2_input_files[i][:-1] + ".jpg"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","JPG/480p",test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            #if np.random.random() > 0.5:
            #    img_input = TF.vflip(img_input)
            #    img_exptC = TF.vflip(img_exptC)
            #    img2 = TF.vflip(img2)

            a = np.random.uniform(0.6,1.4)
            img_input = TF.adjust_brightness(img_input,a)

            a = np.random.uniform(0.8,1.2)
            img_input = TF.adjust_saturation(img_input,a)


        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_XYZ_unpaired(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK"):
        self.mode = mode
        self.unpaird_data = unpaird_data

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_HDRplus(Dataset):
    def __init__(self, root, mode="train", combined=True):
        self.mode = mode

        file = open(os.path.join(root,'train.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"middle_480p",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"output_480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"middle_480p",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"output_480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio)
            crop_w = round(W*ratio)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h,crop_w,img_input.shape())
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

            #a = np.random.uniform(0.8,1.2)
            #img_input = TF_x.adjust_saturation(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

class ImageDataset_HDRplus_unpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        file = open(os.path.join(root,'train.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"middle_480p",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"output_480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"middle_480p",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"output_480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"middle_480p",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"output_480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":
            ratio = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio)
            crop_w = round(W*ratio)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            a = np.random.uniform(0.8,1.2)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class DistortionDataset(Dataset):
    def __init__(self, root, split, distortion_transforms, transform=None):
        super().__init__()
        self.transform = transform
        self.distortion_transforms = distortion_transforms

        train_percentage = 0.8

        self.image_paths = [os.path.join(root, image_path) for image_path in os.listdir(root) if image_path[0] != "."]

        # Split into train & test sets
        if split == "train":
            self.input_images = self.image_paths[:int(train_percentage * len(self.image_paths))]
        elif split == "test":
            self.input_images = self.image_paths[int(train_percentage * len(self.image_paths)):]
        else:
            raise NotImplementedError(f"split: {split} is not implemented")

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        source_path = self.input_images[idx]
        source_img = cv2.imread(source_path)
        inputs, target = [source_img], source_img
        inputs, target = self.distortion_transforms(inputs, target)

        if self.transform is not None:
            inputs, target = self.transform(inputs, target)

        return inputs, target


def get_distortion_dataset(dataset_root: str,
                           batch_size: int,
                           num_workers: int,
                           distortion_transform,
                           train_transforms: transforms.Compose = None,
                           test_transforms: transforms.Compose = None,
                           ):
    train_loader = torch.utils.data.DataLoader(
        dataset=DistortionDataset(
            root=dataset_root,
            split="train",
            distortion_transforms=distortion_transform,
            transform=train_transforms
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=DistortionDataset(
            root=dataset_root,
            split="test",
            distortion_transforms=distortion_transform,
            transform=test_transforms
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader