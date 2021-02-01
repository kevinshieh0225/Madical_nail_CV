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
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module, Flatten
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
        label: 標籤  : int x={0,1,2} array
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
            #將未處理照片存入新資料夾位置：./ML_hw2/resize/
            if not os.path.isfile(new_img_path):
                img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_img_path, img)
            path.append(new_img_path)
            label.append(float(clean_line[2]))

    print('data size: ')
    print(len(path), len(label))
    print(path[:3])
    print(label[:3])
    print(type(path),type(label))
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

def train(model,name,n_epochs,train_loader,valid_loader,optimizer,criterion_train,criterion_test,batch_size,patience):
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
    print(f'Start to run {name}')
#     best_train_loss = 100
#     best_train_acc = 0
#     best_val_loss = 100
#     best_val_acc = 0
#     best_F1 = 0
#     last_epoch = 0

    history = {
        'train_loss':[],
        'valid_loss':[],
    }
    
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('no gpu use')
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss,valid_loss = 0.0,0.0
        train_losses,valid_losses=[],[]

        print(f'running epoch: {epoch}/{n_epochs}')
        #############################################################################################################
        #                                              train the model                                              #
        #############################################################################################################
        model.train()
        for num, (data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
            else:
                print('1')
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion_train(output.flatten(), target.float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item())
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            if num%10 == 0 :
                print(f'train stage：{num}/{len(train_loader)}', end='\r')
        #############################################################################################################
        #                                            validate the model                                             #
        #############################################################################################################
        model.eval()
        for num, (data, target) in enumerate(valid_loader):
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():#train_on_gpu
                data, target = data.cuda(), target.cuda()
                
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion_test(output.flatten(), target.float())
            # update validation loss
            valid_losses.append(loss.item())
                        
            if num%10 == 0 :
                print(f'Valid stage：{num}/{len(valid_loader)}', end='\r')
        #############################################################################################################
        #                                     print train/val/cmt epoch result                                      #
        #############################################################################################################
        # calculate average losses
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        print(f'Training Loss: {train_loss:.3f} \tValidation Loss: {valid_loss:.3f}\n')
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        #############################################################################################################
        #                                              Early Stopping                                               #
        #############################################################################################################
#         if best_F1 >= F1:
#             trigger_times += 1
#             print(f'trigger times: {trigger_times}\n')
#             if trigger_times > patience:
#                 print(f'Early stopping at trigger times: {trigger_times}')
#                 print(f'Least Training Loss: {best_train_loss:.4f} \nLeast Validation Loss: {best_val_loss:.4f}')
#                 print(f'Best Training Accuracy: {best_train_acc:.4f} \nBest Validation Accuracy: {best_val_acc:.4f}')
#                 print(f'Best f1-score: {best_F1:.4f}')
#                 last_epoch = epoch
#                 break
#         else:
#             save_json = {
#                 'accuracy':[train_acc],
#                 'val_accuracy':[valid_acc],
#                 'loss':[train_loss],
#                 'val_loss':[valid_loss],
#                 'precision':[p],
#                 'recall':[r],
#                 'f1':[F1]
#             }
                
#             trigger_times = 0
#             torch.save(model, f'./result/{name}/model.pt')
#             best_train_loss = train_loss
#             best_train_acc = train_acc
#             best_val_loss = valid_loss
#             best_val_acc = valid_acc
#             best_F1 = F1
    
        #############################################################################################################
        #                                                Draw picture                                               #
        #############################################################################################################
    if not os.path.isdir(f'./result/{name}/'):
        os.makedirs(f'./result/{name}/')
    with open(f'./result/{name}/result.json', 'w') as json_file:
        json.dump(history, json_file)
    
    x = np.arange(1,n_epochs+1,1)
    train_loss = history['train_loss']
    valid_loss = history['valid_loss']
    
    fig = plt.figure(figsize=(12,4))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.subplot(1,1,1)
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, train_loss, label='training')
    plt.plot(x, valid_loss, label='validation')
    plt.legend(loc='upper right')
    
    #save result
    plt.savefig(f'./result/{name}/plot.png')
#     torch.save(model, f'./result/{name}/model.pt')
    return model

def testing_result(model , folder = './ML_hw2/學生的testing_data/', csv_path = './HW2_E24056954.csv'): 
    """
    intro:
        讀取'./ML_hw2/學生的testing_data/'
        並將照片處理，拿原本模型預測後輸出文件
    aug:
        model
    result:
        ./HW2_E24056954.csv
    """
    #############################################################################################################
    #                                   loading and resize testing picture                                      #
    #############################################################################################################
    testing_path = []
    testing_write = []
    slice_csv = 'testing_data'#提取testing_data
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
    
    #############################################################################################################
    #                                   use hypothesis model predict testing set                                #
    #############################################################################################################
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    model.eval()
    pred_regression=[]
    for path in testing_path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = transform(img).cuda()
        img = img.unsqueeze(0)
        with torch.no_grad(): 
            output=model(img)
        output = float(output.squeeze(0)[0])
        pred_regression.append(output)
        print(f'{path} / {output}')
    #############################################################################################################
    #                                             output require csv                                            #
    #############################################################################################################
    with open('HW3_E24056954.csv', 'w', encoding='utf8') as wp:
        wp.write(testing_write[0])
        for pred_regression_,testing_write_ in zip(pred_regression,testing_write[1:]):
            wp.write(f'{testing_write_[0]},{testing_write_[1]},{pred_regression_},{testing_write_[3]}\n')

def testing_result_v2(model , folder = './testing_data_part2/' , csv_path = './testing_data_part2.csv'): 
    """
    intro:
        讀取'./ML_hw2/學生的testing_data/'
        並將照片處理，拿原本模型預測後輸出文件
    aug:
        model
    result:
        ./testing_data_part2.csv
    """
    #############################################################################################################
    #                                   loading and resize testing picture                                      #
    #############################################################################################################
    testing_path = []
    testing_write = []
    resize_folder = f'{folder}resize/'
    if not os.path.isdir(resize_folder):
        os.makedirs(resize_folder)
    with open(csv_path, 'r', encoding='utf8') as f:   
        testing_write.append(f.readline())
        for line in f:
            clean_line = line.replace('\n', '').split(',')
            # [id, light, ground_truth, grade]
            testing_write.append(clean_line)
            curr_img_path = f'{folder}/{clean_line[0]}'
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
    
    #############################################################################################################
    #                                   use hypothesis model predict testing set                                #
    #############################################################################################################
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    model.eval()
    pred_regression=[]
    for path in testing_path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = transform(img).cuda()
        img = img.unsqueeze(0)
        with torch.no_grad(): 
            output=model(img)
        output = float(output.squeeze(0)[0])
        pred_regression.append(output)
    #############################################################################################################
    #                                             output require csv                                            #
    #############################################################################################################
    with open('HW3_E24056954.csv', 'w', encoding='utf8') as wp:
        wp.write(testing_write[0])
        for pred_regression_,testing_write_ in zip(pred_regression,testing_write[1:]):
            wp.write(f'{testing_write_[0]},{pred_regression_}\n')


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)
    
    # Try wide_resnet50_2 finetune
    for i in range(10):
#         name = f'HW3_model_ft_wide_resnet50_2_trainloss_MSELoss_{i}'
#         model_ft = models.wide_resnet50_2(pretrained=True)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs,1)
    #     # Try vgg19 finetune
        name = f'HW3_model_ft_vgg19_{i}'
        model_ft = models.vgg19(pretrained=True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,1)

        if not os.path.isfile(f'./result/{name}/model.pt'):
            model_ft=model_ft.to(device)# 放入裝置
            model_ft = model_ft.float()  
            n_epochs = 10
            batch_size = 32
            train_dataloader, valid_dataloader = dataloader_prepare(folder='./ML_hw2/學生的training_data/',batch_size = batch_size)
            optimizer = torch.optim.Adam([
                {'params':model_ft.parameters()}
            ], lr=0.0001)
            criterion_train = nn.MSELoss()
            criterion_test = nn.MSELoss()
            patience = 3

            model_ft = train(model_ft,
                  name,
                  n_epochs,
                  train_dataloader,
                  valid_dataloader,
                  optimizer,
                  criterion_train,
                  criterion_test,
                  batch_size,
                  patience)
        else :
            model_ft = torch.load(f'./result/{name}/model.pt')
#     # Print testing data result
#     testing_result_v2(model_ft , folder = './testing_data_part2/' , csv_path = './testing_data_part2.csv')