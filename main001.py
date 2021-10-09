from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

dataset_dir = Path("./datasets")

# train params setting const
batch_size=4
shuffle=True
num_workers=4
train_epochs=100

# model params setting const
# https://pytorch.org/vision/stable/models.html
model_ft=models.resnet50(pretrained=True)
train_criterion=nn.CrossEntropyLoss()

# optimizer params
# 論文によって最適値が決められているので，そのあたりを設定してもらえれば無難
train_optimizer=optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
train_scheduler=optim.lr_scheduler.StepLR(train_optimizer, step_size=7, gamma=0.1)



def load_dataset():
  data_transforms = {
    "train": transforms.Compose(
      [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]
    ),
    "eval": transforms.Compose(
      [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ]
    )
  }
  return {
    x: datasets.ImageFolder(dataset_dir / x, data_transforms[x])
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
      f"[test] loss: {info['eval_loss']:.6f}, accuracy: {info['eval_accuracy']:.0%}"
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
    for inputs, labels in dataloaders[phase]:
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

  fig.savefig("history.png")

if __name__ == '__main__':
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