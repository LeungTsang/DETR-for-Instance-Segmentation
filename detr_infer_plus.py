import argparse
import datetime
import time

import numpy as np
import torch
from torch import nn
import coco
from torch.utils.data import DataLoader
from detr_solo import *
import util.misc as utils

import torchvision.transforms as T

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pad(img, divisor):
    print(img.shape)
    pad_h = int(np.ceil(img.shape[1] / divisor)) * divisor - img.shape[1]
    pad_w = int(np.ceil(img.shape[2] / divisor)) * divisor - img.shape[2]
    padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
    return padded_img

def get_args_parser():
    parser = argparse.ArgumentParser('DETR_SOLO', add_help=False)

    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')

    return parser

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

CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'N/A', 'N/A']

def main(arg):

    device = torch.device(args.device)
    model = detr_solo(num_classes=18)
    model.to(device)
    state_dict = state_dict = torch.load('detr_solo_cat2000.pth')
    model.load_state_dict(state_dict)
    model.eval()

    img = Image.open('./img/000000000977.jpg')
    img = pad(transform(img),32)
    img = img.to(device).unsqueeze(0)
    output = model(img)
    pred_mask = output["pred_mask"]
    probas = output['pred_cls'].softmax(-1)[0, :, :]
    print(pred_mask.shape)
    for i in range(10):
        print(CLASSES[probas[i].argmax()])
        plt.imshow(pred_mask[i].detach().cpu()) 
        plt.show()
        #loss_dict = criterion(output, target)
        #weight_dict = criterion.weight_dict
        #losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR SOLO', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)

