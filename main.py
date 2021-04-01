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

def get_args_parser():
    parser = argparse.ArgumentParser('DETR_SOLO', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--loss_cls_w', default=1, type=float)
    parser.add_argument('--loss_mask_w', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    return parser

def main(arg):

    device = torch.device(args.device)
    model = detr_solo(num_classes=5)
    model.to(device)
    state_dict = state_dict = torch.load('detr_solo.pth')
    model.load_state_dict(state_dict)
    model.train()

    weight_dict = {'loss_cls': args.loss_cls_w, 'loss_mask': args.loss_mask_w}
    criterion = SetCriterion(num_classes=5, weight_dict=weight_dict, eos_coef=args.eos_coef)
    criterion.train()

    dataset_train = coco.build_dataset(image_set='train', args=args)
    dataset_val = coco.build_dataset(image_set='val', args=args)

    data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    print("Start training")
    start_time = time.time()
    for epoch in range(0, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader_train, 0):
            img, target = data
            #print(type(img))
            img = img.to(device)
            target = {k: v.to(device) for k, v in target.items()}

            optimizer.zero_grad()
            #print(img.shape)
            output = model(img)
            cls = output['pred_cls'].cpu().detach().numpy()
            mask = output['pred_mask'].cpu().detach().numpy()
            loss_dict = criterion(output, target)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            

            losses.backward()
            #optimizer.zero_grad()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            #optimizer.step()
            running_loss += losses.item()
            if i%100 == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss ))
                running_loss = 0.0
                print(loss_dict)
                #PATH = './detr_solo'+str(i)+'.pth'
                #torch.save(model.state_dict(), PATH)
            optimizer.step()
        PATH = './detr_solo'+str(epoch)+'.pth'
        torch.save(model.state_dict(), PATH)

    print("Finish training")

    PATH = './detr_solo.pth'
    torch.save(model.state_dict(), PATH)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR SOLO', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)

