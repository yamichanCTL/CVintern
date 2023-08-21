import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import torch

torch.manual_seed(0)  # 随机数生成器的种子设置为0
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

train_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

np.random.shuffle(train_path)
np.random.shuffle(test_path)

DATA_CACHE = {}


class XunFeiDataset(Dataset):
    def __init__(self, img_path, transform=None):# transform数据增强的操作集
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img

        # 随机选择一些通道
        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)


import albumentations as A

train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
                  A.Compose([
                      A.RandomRotate90(),
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      # A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=True, num_workers=0, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[-10:],
                  A.Compose([
                      A.RandomCrop(120, 120),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
                  A.Compose([
                      A.RandomCrop(128, 128),
                      A.HorizontalFlip(p=0.5),
                      # A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)


class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()

        model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


model = XunFeiNet()
model = model.to('cuda')
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), 0.001)


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)   # casting to long
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"lossitem:{loss.item()}")

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate(val_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.type(torch.LongTensor)  # casting to long
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)


for num in range(2):
    print(f"epoch:{num}")
    train_loss = train(train_loader, model, criterion, optimizer)
    val_acc = validate(val_loader, model, criterion)
    train_acc = validate(train_loader, model, criterion)

    print(f"loss:{train_loss},train_acc{train_acc},val_acc{val_acc}")


def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)


pred = None
for _ in range(10):
    if pred is None:
        pred = predict(test_loader, model, criterion)
    else:
        pred += predict(test_loader, model, criterion)

submit = pd.DataFrame(
    {
        'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],
        'label': pred.argmax(1)
    })
submit['label'] = submit['label'].map({1: 'NC', 0: 'MCI'})
submit = submit.sort_values(by='uuid')
submit.to_csv('submit2.csv', index=None)