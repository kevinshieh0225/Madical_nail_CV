import os
import numpy as np
import re
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import gc

#import h5py

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
            print('Not found img : ', self.x[idx])
            print(self.x[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, self.y[idx]
   

def load_nail_csv(folder='./ML_hw2/學生的training_data/'):
    """
    intro:
        先存入原圖位置，壓縮照片後再存入./ML_hw2/學生的training_data/resize/
        回傳 path , label
    aug:
        folder = 讀入資料之目的資料夾
        batch_size = batch_size
    output:
        path: 照片路徑 : ./ML_hw2/resize/id
        label: 標籤  : [1,0,0]
    """
    path = []
    label = []
    slice_csv = re.sub('學生的', "" ,folder.split('/')[-2] ) #提取training_data或test_data
    csv_path = f'{folder}{slice_csv}.csv'
    resize_folder = f'{folder}resize/'
    if not os.path.isdir(resize_folder):
        os.makedirs(resize_folder)
    with open(csv_path, 'r', encoding='utf8') as f:        
        f.readline()
        for line in tqdm(f):
            clean_line = line.replace('\n', '').replace('\ufeff', '').split(',')
            # [id, light, ground_truth, grade]
            curr_img_path = f'{folder}{clean_line[1]}/{clean_line[0]}'
            new_img_path = f'{resize_folder}{clean_line[0]}'
            if not os.path.isfile(curr_img_path):
                print(f'No file for path : {curr_img_path}')
                continue
            if not os.path.isfile(new_img_path):
                img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_img_path, img)
            path.append(new_img_path)
            label.append(int(clean_line[3])-1)

    print('data size: ')
    print(len(path), len(label))
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
    intro:
        使用load_nail_csv準備照片
        Dataset轉為dataset型式
        切 train, validation set , DataLoader存入
    aug:
        folder = 讀入資料之目的資料夾
        batch_size = batch_size
    output:
        train_dataloader, valid_dataloader
    """

    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(p = 0.5),
        torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    path, label = load_nail_csv(folder)

    augment_dataset = Dataset(path, label, transform)
    #切分70%當作訓練集、30%當作驗證集
    train_size = int(0.7 * len(augment_dataset))
    valid_size = len(augment_dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(augment_dataset, [train_size, valid_size])
    
    train_dataloader = DataLoader( train_data , batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader( valid_data , batch_size=batch_size, shuffle=True)
    
    return train_dataloader, valid_dataloader


def train(model,name,n_epochs,train_loader,valid_loader,optimizer,criterion,batch_size,patience):
    """
    intro:
        每次epoch都 train the model , validate the model
        並計算Early Stopping
        印出 train_loss , train_acc , val_loss , val_acc
        回傳 model
    aug:
        model,n_epochs,train_loader,valid_loader,optimizer,criterion,batch_size
    output:
        model
    """
    if os.path.isfile(f'./hw2_classification_weight_{name}.pt'):
        print(f'keep finetune{name}')
        model = torch.load(f'./hw2_classification_weight_{name}.pt')
    
    best_train_loss = 100
    best_train_acc = 0
    best_val_loss = 100
    best_val_acc = 0
    
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('no gpu use')
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss,valid_loss = 0.0,0.0
        train_losses,valid_losses=[],[]
        train_correct,val_correct,train_total,val_total=0,0,0,0

        print('running epoch: {}'.format(epoch))
        #############################################################################################################
        #                                              train the model                                              #
        #############################################################################################################
        model.train()
        for data, target in train_loader:
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
        
        #############################################################################################################
        #                                            validate the model                                             #
        #############################################################################################################
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
                
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss =criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            val_total += data.size(0)
            valid_losses.append(loss.item()*data.size(0))
        
        # calculate average losses
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        
        #############################################################################################################
        #                                        calculate average accuracy                                         #
        #############################################################################################################
        train_acc=train_correct/train_total
        valid_acc=val_correct/val_total
        print(f'\tTraining Loss: {train_loss:.3f} \tValidation Loss: {valid_loss:.3f}')
        print(f'\tTraining Accuracy: {train_acc:.3f} \tValidation Accuracy: {valid_acc:.3f}')
        
        #############################################################################################################
        #                                              Early Stopping                                               #
        #############################################################################################################
        if valid_loss > best_val_loss:
            trigger_times += 1
            print(f'trigger times: {trigger_times}\n')
            if trigger_times > patience:
                print(f'Early stopping at trigger times: {trigger_times}')
                print(f'\tLeast Training Loss: {best_train_loss:.4f} \nLeast Validation Loss: {best_val_loss:.4f}')
                print(f'\tBest Training Accuracy: {best_train_acc:.4f} \nBest Validation Accuracy: {best_val_acc:.4f}')
                break
        else:
            trigger_times = 0
            torch.save(model, f'./hw2_classification_weight_{name}.pt')
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_val_loss = valid_loss
            best_val_acc = valid_acc
            
    model = torch.load(f'./hw2_classification_weight_{name}.pt')    
    return model

def testing_result(model): 
    """
    intro:
        讀取'./ML_hw2/學生的testing_data/'
        並將照片處理，拿原本模型預測後輸出文件
    aug:
        model,n_epochs,train_loader,valid_loader,optimizer,criterion,batch_size
    output:
        model
    """
    testing_path = []
    testing_write = []
    folder = './ML_hw2/學生的testing_data/'
    slice_csv = 'testing_data'#提取testing_data
    csv_path = f'{folder}{slice_csv}.csv'
    resize_folder = f'{folder}resize/'
    if not os.path.isdir(resize_folder):
        os.makedirs(resize_folder)
    with open(csv_path, 'r', encoding='utf8') as f:   
        testing_write.append(f.readline())
        for line in f:
            clean_line = line.replace('\n', '').replace('\ufeff', '').split(',')
            # [id, light, ground_truth, grade]
            testing_write.append(clean_line)
            curr_img_path = f'{folder}{slice_csv}/{clean_line[0]}'
            new_img_path = f'{resize_folder}{clean_line[0]}'
            if not os.path.isfile(curr_img_path):
                print(curr_img_path)
                print('catch')
                continue
            if not os.path.isfile(new_img_path):
                img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_img_path, img)
            testing_path.append(new_img_path)
    print('data size: ')
    print(f'testing數量：{len(testing_path)}')
    print(testing_write)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    model.eval()
    pred_label=[]
    for path in tqdm(testing_path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = transform(img).cuda()
        img = img.unsqueeze(0)
        with torch.no_grad(): 
            output=model(img)
        pred = output.data.max(dim = 1, keepdim = True)[1]
        pred_label.append(int(pred))
        print(f'{path} / {int(pred)}')
    with open('HW2_E24056954.csv', 'w', encoding='utf8') as wp:
        wp.write(testing_write[0])
        for pred_label_,testing_write_ in zip(pred_label,testing_write[1:]):
            wp.write(f'{testing_write_[0]},{testing_write_[1]},,{pred_label_+1}\n')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

######################################################## Try alexnet finetune#####################################################

# model_ft_alexnet = models.alexnet(pretrained=True)
# num_ftrs = model_ft_alexnet.classifier[6].in_features
# model_ft_alexnet.classifier[6] = nn.Linear(num_ftrs,3)
# model_ft_alexnet=model_ft_alexnet.to(device)# 放入裝置
# print(model_ft_alexnet) # 列印新模型

# n_epochs = 100
# batch_size = 32
# train_dataloader, valid_dataloader = dataloader_prepare(folder='./ML_hw2/學生的training_data/',batch_size = batch_size)
# optimizer = torch.optim.Adam([
#     {'params':model_ft_alexnet.parameters()}
# ], lr=0.00001)
# criterion = nn.CrossEntropyLoss()
# patience = 3

# model_ft_alexnet=train(model_ft_alexnet,'model_ft_wide_resnet50_2',n_epochs,train_dataloader,valid_dataloader,optimizer,criterion, batch_size , patience)

#################################################### Try wide_resnet50_2 finetune #############################################

# model_ft_wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# num_ftrs = model_ft_wide_resnet50_2.fc.in_features
# model_ft_wide_resnet50_2.fc = nn.Linear(num_ftrs,3)
# model_ft_wide_resnet50_2=model_ft_wide_resnet50_2.to(device)# 放入裝置
# print(model_ft_wide_resnet50_2) # 列印新模型

# n_epochs = 100
# batch_size = 32
# train_dataloader, valid_dataloader = dataloader_prepare(folder='./ML_hw2/學生的training_data/',batch_size = batch_size)
# optimizer = torch.optim.Adam([
#     {'params':model_ft_wide_resnet50_2.parameters()}
# ], lr=0.0001)
# criterion = nn.CrossEntropyLoss()
# patience = 3

# model_ft_wide_resnet50_2=train(model_ft_wide_resnet50_2,'model_ft_wide_resnet50_2',n_epochs,train_dataloader,valid_dataloader,optimizer,criterion, batch_size , patience)

######################################################## Try vgg16 finetune ####################################################

# model_ft_vgg19 = models.vgg19(pretrained=True)
# num_ftrs = model_ft_vgg19.classifier[6].in_features
# model_ft_vgg19.classifier[6] = nn.Linear(num_ftrs,3)
# model_ft_vgg19=model_ft_vgg19.to(device)# 放入裝置
# print(model_ft_vgg19) # 列印新模型

# n_epochs = 100
# batch_size = 32
# train_dataloader, valid_dataloader = dataloader_prepare(folder='./ML_hw2/學生的training_data/',batch_size = batch_size)
# optimizer = torch.optim.Adam([
#     {'params':model_ft_vgg19.parameters()}
# ], lr=0.0001)
# criterion = nn.CrossEntropyLoss()
# patience = 3

# model_ft_vgg19=train(model_ft_vgg19,'model_ft_vgg19',n_epochs,train_dataloader,valid_dataloader,optimizer,criterion, batch_size , patience)

# torch.save(model_ft, "./hw2_classification")
# model_ft = torch.load('./hw2_classification')

# Print testing data result

name = 'model_ft_wide_resnet50_2'
model_ft_wide_resnet50_2 = torch.load(f'./hw2_classification_weight_{name}.pt')  
print(model_ft_wide_resnet50_2)
testing_result(model_ft_wide_resnet50_2)