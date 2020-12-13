import os
import numpy as np
import re
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import gc

import PIL
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

import torchvision.models as models
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torch.optim import Adam
from tqdm.notebook import tqdm as tqdm
from ipywidgets import IntProgress

import json

class Dataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        if img is None:
            print('fuck')
            print(self.x[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, self.y[idx]
   

def load_nail_csv(folder='./ML_hw2/學生的training_data/'):
    path = []
    label = []
    # 此時我需要的輸出格式為：
    # path: 照片路徑 : ./ML_hw2/學生的training_data/A/id
    # label: 標籤  : [1,0,0]
    slice_csv = re.sub('學生的', "" ,folder.split('/')[-2] ) #提取training_data或test_data
    csv_path = f'{folder}{slice_csv}.csv'
    with open(csv_path, 'r', encoding='utf8') as f:        
        f.readline()
        for line in f:
            clean_line = line.replace('\n', '').replace('\ufeff', '').split(',')
            # [id, light, ground_truth, grade]
            curr_img_path = f'{folder}{clean_line[1]}/{clean_line[0]}'
            if not os.path.isfile(curr_img_path):
                print('catch')
                continue
            """
            img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(curr_img_path)
                continue
            """
            
            path.append(curr_img_path)
            label.append(int(clean_line[3])-1)
    print('data size: ')
    print(len(path), len(label))
    print(path[:10])
    print(label[:10])
    count = np.zeros(3)
    for check in label:
      count[int(check)] += 1
    print('(1)：'+str(count[0])+' '+str(count[0]/len(label)))
    print('(2)：'+str(count[1])+' '+str(count[1]/len(label)))
    print('(3)：'+str(count[2])+' '+str(count[2]/len(label)))

    print()
    return path, label

def dataloader_prepare(folder='./ML_hw2/學生的training_data/', batch_size=8):
    """
    aug:
        folder:讀入資料之位置
    """
    transform_flip = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(p = 1),
        torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform_rotation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomRotation((10,15), resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    path, label = load_nail_csv(folder)
    
    l = []
    l.append(Dataset(path, label, transform_flip))
    l.append(Dataset(path, label, transform))
    l.append(Dataset(path, label, transform_rotation))
    augment_dataset = torch.utils.data.ConcatDataset(l)
    #augment_dataset = Dataset(path, label, transform_flip) + Dataset(path, label, transform) + Dataset(path, label, transform_rotation)
    print(f'augment data len : {len(augment_dataset)}')
    print(f'augment data type : {augment_dataset}')
    #切分70%當作訓練集、30%當作驗證集
    train_size = int(0.7 * len(augment_dataset))
    valid_size = len(augment_dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(augment_dataset, [train_size, valid_size])
    print(train_data)
    print(f'augment data len : {len(train_data)}')
    print(f'augment data type : {train_data}')
    
    print(valid_data)
    print(f'augment data len : {len(valid_data)}')
    print(f'augment data type : {valid_data}')
    
    train_dataloader = DataLoader( train_data , batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader( valid_data , batch_size=batch_size, shuffle=True)
    
    
    print(train_data)
    print(f'augment data len : {len(train_dataloader)}')
    print(f'augment data type : {train_dataloader}')
    
    print(valid_data)
    print(f'augment data len : {len(valid_dataloader)}')
    print(f'augment data type : {valid_dataloader}')
    
    return train_dataloader, valid_dataloader

class CNN_Model(nn.Module):
    #列出需要哪些層
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.CNN = nn.Sequential(                       #(3,224,224)
            nn.Conv2d(3, 32, kernel_size=5, stride=1),  #(16,220,220)
            nn.ReLU(inplace=True),
            # Max pool 1
            nn.MaxPool2d(kernel_size=2),                #(16,110,110)
            # Convolution 2
            nn.Conv2d(32,32, kernel_size=5, stride=1),  #(8,106,106)
            nn.ReLU(inplace=True),
            # Max pool 2
            nn.MaxPool2d(kernel_size=2)                 #(8,53,53)
            # Fully connected 1 ,#input_shape=(8*53*53)
        )
        
        self.main = nn.Sequential(
            nn.Linear(in_features=32*53*53, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=3),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        out = self.CNN(input)
        out = out.view(out.size(0), -1) 
        return self.main(out)

def train(model,n_epochs,train_loader,valid_loader,optimizer,criterion,batch_size):
    train_acc_his,valid_acc_his=[],[]
    train_losses_his,valid_losses_his=[],[]
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('1')
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss,valid_loss = 0.0,0.0
        train_losses,valid_losses=[],[]
        train_correct,val_correct,train_total,val_total=0,0,0,0
        train_pred,train_target=torch.zeros(batch_size,1),torch.zeros(batch_size,1)
        val_pred,val_target=torch.zeros(batch_size,1),torch.zeros(batch_size,1)
        count=0
        count2=0
        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
            else:
                print('1')
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item()*data.size(0))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if count==0:
                train_pred=pred
                train_target=target.data.view_as(pred)
                count=count+1
            else:
                train_pred=torch.cat((train_pred,pred), 0)
                train_target=torch.cat((train_target,target.data.view_as(pred)), 0)
        train_pred=train_pred.cpu().view(-1).numpy().tolist()
        train_target=train_target.cpu().view(-1).numpy().tolist()
######################
        # validate the model #
        ######################
        model.eval()
        for data, target in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
            else:
                print('1')
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss =criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            val_total += data.size(0)
            valid_losses.append(loss.item()*data.size(0))
            if count2==0:
                val_pred=pred
                val_target=target.data.view_as(pred)
                count2=count+1
            else:
                val_pred=torch.cat((val_pred,pred), 0)
                val_target=torch.cat((val_target,target.data.view_as(pred)), 0)
        val_pred=val_pred.cpu().view(-1).numpy().tolist()
        val_target=val_target.cpu().view(-1).numpy().tolist()
        
        # calculate average losses
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        
        # calculate average accuracy
        train_acc=train_correct/train_total
        valid_acc=val_correct/val_total
        train_acc_his.append(train_acc)
        valid_acc_his.append(valid_acc)
        train_losses_his.append(train_loss)
        valid_losses_his.append(valid_loss)
    # print training/validation statistics 
        print(f'\tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
        print(f'\tTraining Accuracy: {train_acc:.6f} \tValidation Accuracy: {valid_acc:.6f}')
    return train_acc_his,valid_acc_his,train_losses_his,valid_losses_his,model

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)
model1=CNN_Model()
n_epochs = 10
LR = 0.001
batch_size = 32
train_dataloader, valid_dataloader = dataloader_prepare(folder='./ML_hw2/學生的training_data/',batch_size = 32)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR)
criterion = CrossEntropyLoss()
train_acc_his,valid_acc_his,train_losses_his,valid_losses_his,model1=train(model1,n_epochs,train_dataloader,valid_dataloader,optimizer1,criterion, batch_size)
"""
import json
from io import StringIO
import numpy as np

with open('report.json', 'a') as add_file:
    dict_data = {'model':model1,'n_epochs':n_epochs,'LR':LR,'batch_size'=batch_size,'train_acc': train_acc_his,'valid_acc':valid_acc_his}
    add_file = json.dump(dict_data)
"""
