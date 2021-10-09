import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from lib.ImageFolderWithPaths import ImageFolderWithPaths
import torch.nn.functional as F
import cv2
#from gradcam.utils import visualize_cam 
#from gradcam import GradCAM, GradCAMpp



dataset_dir = Path("./datasets")

# train params setting const
batch_size=64
shuffle=True
num_workers=4
train_epochs=10
eval_threshold_ratio = 0.5
eval_threshold_epoch = 10

# model params setting const
# https://pytorch.org/vision/stable/models.html
model_ft=models.resnet50(pretrained=True)
train_criterion=nn.CrossEntropyLoss()

# optimizer params
# 論文によって最適値が決められているので，そのあたりを設定してもらえれば無難
# train_optimizer=optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.5)
train_optimizer=optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
train_scheduler=optim.lr_scheduler.StepLR(train_optimizer, step_size=7, gamma=0.1)

eval_error_dict_num = dict()
eval_correct_dict_num = dict()
train_error_dict_num = dict()
train_correct_dict_num = dict()

eval_ans_dict_epoch = dict()
train_ans_dict_epoch = dict()

def load_dataset():
  data_transforms = {
    "train": transforms.Compose(
      [
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ]
    ),
    "eval": transforms.Compose(
      [ 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.RandomHorizontalFlip(0.8),
        #transforms.RandomVerticalFlip(0.8),
        #transforms.RandomRotation(degrees=10),
      ]
    )
  }
  return {
    x: ImageFolderWithPaths(dataset_dir / x, data_transforms[x])
    for x in ["train", "eval"]
  }

def make_dataloader(img_datasets):
  return {
    x: data.DataLoader(img_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    for x in ["train", "eval"]
  }

def get_devices(gpu_id=-1):
  if gpu_id >= 0 and torch.cuda.is_available():
    return torch.device("cuda", gpu_id)
  else:
    return torch.device("cpu")

def gen_transform():
  return transforms.Compose(
      [ 
        transforms.Resize(224,224),
        transforms.ToTensor(),
        '''
        transforms.RandomRotation(degrees=180),
        transforms.TenCrop(150),
        transforms.RandomResizedCrop(150, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(0.8),
        transforms.RandomVerticalFlip(0.8),
        transforms.RandomRotation(degrees=10),
        '''
      ]
    )

def get_features(model):
  features = []
  for _, val in model._modules.items():
    features.append(val)
  return features


''''
def toHeatMap(feature):
  feature_vec = feature.grad.view(512, 7*7)
  alpha = torch.mean(feature_vec, axis=1)
  feature = feature.squeeze(0)
  L = F.relu(torch.sum(feature*alpha.view(-1,1,1),0)) 
  L = L.detach().numpy()
  L_min = np.min(L) 
  L_max = np.max(L - L_min) 
  L = (L - L_min)/L_max
  L = cv2.resize(L, (224, 224))
  L = (L*255).reshape(-1)
  return L.reshape(224, 224, 3)
'''


# 指定epoch数分の学習を行う
def do_train(model, criterion, optimizer, scheduler, dataloaders, device, epochs):
  train_history = []
  for epoch in range(epochs):
    info, model = train_on_epoch(
      model, criterion, optimizer, scheduler, dataloaders, device
    )
    info["epoch"] = epoch + 1
    train_history.append(info)
    print(
      f"epoch {info['epoch']:<2} "
      f"[train] loss: {info['train_loss']:.6f}, accuracy: {info['train_accuracy']:.0%} "
      f"[eval] loss: {info['eval_loss']:.6f}, accuracy: {info['eval_accuracy']:.0%}"
    )
  history = pd.DataFrame(train_history)
  return history, model

# do_trainから呼ばれる 1epoch分の学習を行う
def train_on_epoch(model, criterion, optimizer, scheduler, dataloaders, device):
  info = {}
  for phase in ["train", "eval"]:
    if phase == "train":
      model.train()
    else:
      model.eval()
    total_loss = 0
    total_correct = 0
    for inputs, labels, paths in dataloaders[phase]:
      inputs, labels = inputs.to(device), labels.to(device)

      with torch.set_grad_enabled(phase == "train"):
        # 順伝搬
        outputs = model(inputs)

        # モデルの出力から最も値の大きいクラスを予測結果とする
        preds = outputs.argmax(dim=1)

        # lossの計算
        loss = criterion(outputs, labels)

        # 逆伝搬 + 重みの更新
        if phase == "train":
          optimizer.zero_grad()
          loss.backward()

          optimizer.step()
        
      total_loss += float(loss)
      total_correct += int((preds == labels).sum())

      with torch.set_grad_enabled(phase == "eval"):
        preds = outputs.argmax(dim=1)
      for i in range(len(preds)):
        correct = torch.equal(preds[i], labels[i])
        if not correct and ("eval" in paths[i]) and not("not_adhesion" in paths[i]):
          if paths[i] not in eval_ans_dict_epoch:
            eval_ans_dict_epoch[paths[i]] = [False]
          else:
            eval_ans_dict_epoch[paths[i]].append(False)
          # print(paths[i])
          if paths[i] not in eval_error_dict_num:
            eval_error_dict_num[paths[i]] = 1
          else:
            eval_error_dict_num[paths[i]] += 1
        elif correct and ("eval" in paths[i]) and not("not_adhesion" in paths[i]):
          if paths[i] not in eval_ans_dict_epoch:
            eval_ans_dict_epoch[paths[i]] = [True]
          else:
            eval_ans_dict_epoch[paths[i]].append(True)
          if paths[i] not in eval_correct_dict_num:
            eval_correct_dict_num[paths[i]] = 1
          else:
            eval_correct_dict_num[paths[i]] += 1
        
        if not correct and ("train" in paths[i]):
          # print(paths[i])
          if paths[i] not in train_ans_dict_epoch:
            train_ans_dict_epoch[paths[i]] = [False]
          else:
            train_ans_dict_epoch[paths[i]].append(False)
          if paths[i] not in train_error_dict_num:
            train_error_dict_num[paths[i]] = 1
          else:
            train_error_dict_num[paths[i]] += 1
        elif correct and ("train" in paths[i]):
          if paths[i] not in train_ans_dict_epoch:
            train_ans_dict_epoch[paths[i]] = [True]
          else:
            train_ans_dict_epoch[paths[i]].append(True)
          if paths[i] not in train_correct_dict_num:
            train_correct_dict_num[paths[i]] = 1
          else:
            train_correct_dict_num[paths[i]] += 1
    
    if phase == "train":
      scheduler.step()
    
    # 損失関数の値の平均 + 精度を計算
    info[f"{phase}_loss"] = total_loss / len(dataloaders[phase].dataset)
    info[f"{phase}_accuracy"] = total_correct / len(dataloaders[phase].dataset)

  return info, model

def plot_history(history):
  fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))
  # 損失の推移
  ax1.set_title("Loss")
  ax1.plot(history["epoch"], history["train_loss"], label="train")
  ax1.plot(history["epoch"], history["eval_loss"], label="eval")
  ax1.set_xlabel("Epoch")
  ax1.legend()

  # 精度の推移
  ax2.set_title("Accuracy")
  ax2.plot(history["epoch"], history["train_accuracy"], label="train")
  ax2.plot(history["epoch"], history["eval_accuracy"], label="eval")
  ax2.set_xlabel("Epoch")
  ax2.legend()

  # 写真
  fig.savefig("history.png")

def mkdir(path):
  if not os.path.exists(path):
    os.mkdir(path)
  
def dist_num(base, mp):
  for key in mp:
    if mp[key] >= train_epochs * eval_threshold_ratio:
      file_name = os.path.basename(key)
      shutil.copy(key, "{}/{}".format(base, file_name))

def dist_epoch(base, mp, ans):
  for key in mp:
    if mp[key][eval_threshold_epoch - 1] == ans:
      file_name = os.path.basename(key)
      shutil.copy(key, "{}/{}".format(base, file_name))

def rmDir():
  dirs = [
    "./train_correct_images",
    "./train_error_images",
    "./eval_correct_images",
    "./eval_error_images",
  ]
  for d in dirs:
    if os.path.exists(d):
      shutil.rmtree(d)

if __name__ == '__main__':
  rmDir()
  img_datasets = load_dataset()
  class_names = img_datasets["train"].classes
  print("train model to classify {}".format(class_names))
  data_loaders = make_dataloader(img_datasets=img_datasets)
  gpu_device = get_devices(gpu_id=0)

  # set up model
  model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
  model_ft = model_ft.to(gpu_device)
  
  # do train
  history, model = do_train(
    model=model_ft,
    criterion=train_criterion,
    scheduler=train_scheduler,
    optimizer=train_optimizer,
    dataloaders=data_loaders,
    device=gpu_device,
    epochs=train_epochs
  )
  
  # plot history
  plot_history(history=history)

  # model save
  torch.save(model.state_dict(), "./model.pth")
  eval_error_dir = "./eval_error_images"
  eval_correct_dir = "./eval_correct_images"
  train_error_dir = "./train_error_images"
  train_correct_dir = "./train_correct_images"

  mkdir(eval_error_dir)
  mkdir(eval_correct_dir)
  
  mkdir(train_error_dir)
  mkdir(train_correct_dir)
  
  dist_epoch(eval_error_dir, eval_ans_dict_epoch, False)
  dist_epoch(eval_correct_dir, eval_ans_dict_epoch, True)

  dist_epoch(train_error_dir, train_ans_dict_epoch, False)
  dist_epoch(train_correct_dir, train_ans_dict_epoch, True)

  

  '''
  dist_num(eval_error_dir, eval_error_dict_num)
  dist_num(eval_correct_dir, eval_correct_dict_num)

  dist_num(train_error_dir, train_error_dict_num)
  dist_num(train_correct_dir, train_correct_dict_num)
  '''