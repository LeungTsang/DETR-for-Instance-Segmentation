from PIL import Image
import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

torch.set_grad_enabled(False)
import mmcv
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import auto_fp16
from scipy.optimize import linear_sum_assignment

from detr_solo import detr_solo
import torchvision.transforms as T


model = detr_solo(num_classes=5)
state_dict = torch.load('detr_solo4.pth')
model.load_state_dict(state_dict)
model.train()
#model.load_state_dict(state_dict)
"""
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
"""
CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'N/A']

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pad(img, divisor):
    pad_h = int(np.ceil(img.shape[1] / divisor)) * divisor - img.shape[1]
    pad_w = int(np.ceil(img.shape[2] / divisor)) * divisor - img.shape[2]
    padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
    return padded_img

#print(type(model))

#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#im = Image.open(requests.get(url, stream=True).raw)

# standard PyTorch mean-std input image normalization
#transform = T.Compose([
#    T.Resize(800),
#    T.ToTensor(),
#    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#])

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im)
    img = pad(img,32).unsqueeze(0)
    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    print("det")
    img = np.load("img.npy")
    img = torch.from_numpy(img)
    outputs = model(img)



    #print(outputs)
    # keep only predictions with 0.7+ confidence
    print(outputs['pred_cls'])
    probas = outputs['pred_cls'].softmax(-1)[0, :, :]
    print(probas)
    pred_cls = outputs["pred_cls"][0]
    print(pred_cls.argmax(1).t())
    #keep = probas.max(-1).values > 0.7
    mask = outputs['pred_mask']#[keep]
    print(mask.shape)
    print(probas.shape)
    for i in range(mask.shape[0]): 
        print(probas[i])
        cl = probas[i].argmax()
        print(probas[i][cl])
        print(CLASSES[cl])
        plt.imshow(mask[i]>0) 
        plt.show()


im = Image.open('./img/000000391895.jpg')
detect(im, model, transform)


