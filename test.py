from PIL import Image
import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

import numpy as np

mask=np.load("mask.npy")
cls=np.load("cls.npy")

print(mask.shape)
print(cls.shape)
for i in range(mask.shape[0]): 
    print(cls[0][i].argmax())
    plt.imshow(mask[i]) 
    plt.show()


