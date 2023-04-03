#! /usr/bin/env python3

from __future__ import division
import json
import os
import argparse
from PIL import Image
import numpy as np
import tqdm
import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from models import MeasureNet
from utils_a.utils import to_cpu, mkdirs, worker_seed_set, load_classes, print_environment_info, provide_determinism, load_checkpoint_model
from itertools import chain
from utils_a.dataloader import ListDataset
from utils_a.augmentations import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS
from utils_a.parse_config import parse_data_config
from utils_a.loss import MeasureLoss
import cv2
import threading
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
from torch.autograd import Variable
# from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):

    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=False,
        transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the My model.")
    parser.add_argument("-d", "--data", type=str, default="config/config.cfg", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true',default=False, help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--model_weights", type=str, default="checkpoints/model6_2fc_best_ckpt_53.pth", help="Path to model weights for training")
    parser.add_argument("--pretrained_weights", type=str, default="pretrained_weights/resnet50.pth", help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument("--n_anchors", type=int, default=3, help="The number of anchors.")
    # parser.add_argument("--anchors", type=list, default=[[105, 105], [85, 85], [65, 65], [50, 50], [40, 40], [30, 30], [17, 17], [12, 12], [7, 7]], help="The selected anchors.")
    parser.add_argument("--anchors", type=list, default=[[128, 128], [48, 48], [16, 16]], help="The selected anchors.")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    # Create output directories if missing
    os.makedirs("output_gugu", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    valid_path = data_config["valid"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = MeasureNet(back_bone=data_config["back_bone"],
                       pretrained=data_config["pretrained"],
                       anchors=3,
                       num_classes=20)

    model = model.to(device)
    if args.model_weights:
        pth = args.model_weights
        model = load_checkpoint_model(model, pth, device)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        batch_size=1,
        img_size=416,
        n_cpu=4)

    print("vali:",len(validation_dataloader))

    model.eval()

    iimg = []
    results = []
    for imgpath, imgs, targets, htm in tqdm.tqdm(validation_dataloader, desc="Validating"):
        # print(imgpath, imgs.shape, targets)
        # imgs = Variable(imgs.type(Tensor), requires_grad=False)
        imgs = imgs.to(device)
        targets = targets.to(device)
        item = {
            "image_id": '',
            "category_id": 1,
            "keypoints": [],  # x1,y1,score1,x2,y2,score2
            "score": 0,
            "measure_point": []
        }
        item["image_id"] = imgpath[0]
        iimg.append(np.array(imgs[0].permute(1,2,0)))

        with torch.no_grad():
            phtm = model(imgs)  # [1, 10647, 25]
            aa = np.array(phtm.cpu())
            maxa = np.max(aa)
            coorda = np.where(aa == maxa)
            coorda = np.squeeze(coorda)

            item["measure_point"] = coorda.tolist()
            results.append(item)
    json.dump(results, open('myresult.json', 'w'))
    # print("sect_err =", np.mean(res))

