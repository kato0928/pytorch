# Basic Modules
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2

# PyTorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable

# from gradcam.utils import visualize_cam
# from gradcam import GradCAM, GradCAMp

def get_devices(gpu_id=-1):
  if gpu_id >= 0 and torch.cuda.is_available():
    return torch.device("cuda", gpu_id)
  else:
    return torch.device("cpu")

device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
'''
model = models.densenet161(pretrained=True)
model.fc = nn.Linear(2048,5)
model = torch.nn.DataParallel(model).to(device)
model.eval()
'''
gpu_device = get_devices()
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(gpu_device)
model.load_state_dict(torch.load('./model.pth'))
model.fc = nn.Linear(2048,5)
model = torch.nn.DataParallel(model).to(device)
model.eval()

# Grad-CAM
grad_cam = GradCam(model)

images = []
# あるラベルの検証用データセットを呼び出してる想定
for path in glob.glob("./datasets/train/adhesion/*"):
    img = Image.open(path)
    torch_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(img).unsqueeze(0).to(device)
    # normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)
    feature_img = grad_cam(torch_img).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    feature_image.save("out.png")

# grid_image = make_grid(images, nrow=5)

# 結果の表示
# transforms.ToPILImage()(grid_image)