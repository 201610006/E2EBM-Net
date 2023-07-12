#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from models2 import MeasureNet
from utils_a.utils import to_cpu, mkdirs, worker_seed_set, load_classes, print_environment_info, provide_determinism, load_checkpoint_model

from utils_a.dataloader2 import ListDataset
from utils_a.augmentations import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS
from utils_a.parse_config import parse_data_config
from utils_a.loss import MeasureLoss

import threading

from terminaltables import AsciiTable
from torch.autograd import Variable
# from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.
    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

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

def _neg_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    af = 0.95
    #pos_inds = targets.eq(1).float()
    pos_inds = targets.gt(af).float()
    neg_inds = targets.lt(af).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0
    for pred in preds: 

        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                                   2) * neg_weights * neg_inds
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss 
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)

def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the My model.")
    parser.add_argument("-d", "--data", type=str, default="config/config.cfg", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true',default=False, help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--model_weights", type=str, default="checkpoints2/best_ckpt_28.pth", help="Path to model weights for training")
    parser.add_argument("--pretrained_weights", type=str, default="pretrained_weights/resnet34.pth", help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument("--n_anchors", type=int, default=3, help="The number of anchors.")
    parser.add_argument("--anchors", type=list, default=[[[105, 105], [85, 85], [65, 65]], [[50, 50], [40, 40], [30, 30]], [[17, 17], [12, 12], [7, 7]]], help="The selected anchors.")
    # parser.add_argument("--anchors", type=list, default=[[128, 128], [48, 48], [16, 16]], help="The selected anchors.")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    # logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output_gugu", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = MeasureNet(back_bone=data_config["back_bone"],
                       pretrained=data_config["pretrained"],
                       anchors=3,
                       num_classes=20)

    #if args.model_weights:
    #    pth = args.model_weights
    #    model = load_checkpoint_model(model, pth, device)
    model = model.to(device)
    # model = nn.DataParallel(model)
    criterion2 = nn.MSELoss()
    criterion = MeasureLoss(n_classes=data_config["classes"], n_anchors=args.n_anchors, anchors=args.anchors, device=device)
    dataloader = _create_data_loader(
        train_path,
        data_config["batchsize"],
        data_config["img_size"],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        data_config["batchsize"],
        data_config["img_size"],
        args.n_cpu)

    print("train:",len(dataloader),"vali:",len(validation_dataloader))
    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (data_config['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=data_config['learning_rate'],
            weight_decay=data_config['decay'],
        )
    elif (data_config['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=data_config['learning_rate'],
            weight_decay=data_config['decay'],
            momentum=data_config['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    mkdirs("mylogs")
    save_dir = os.path.join("mylogs", datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_.logs"
    fout_log = open(save_dir, 'w')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, patience=20)

    min_loss = 1.1
    for epoch in range(1, args.epochs + 1):
        print("\n---- Training Model ----")

        model.train()  # Set model to training mode
        batch_train = []
        for batch_i, (_, imgs, targets, htmap) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            imgs = imgs.to(device)
            htmap = htmap.to(device)

            outputs, htm = model(imgs)
            dloss, loss_components = criterion(outputs, targets)
            htl = _neg_loss(htm, htmap)
            # htl = criterion2(htm, htmap)

            loss = dloss + 1e-2*htl
            loss.backward()

            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

            batch_train.append([float(loss_components[0]), float(loss_components[1]), float(loss_components[2]),
                                float(loss_components[3]), float(loss_components[4]), float(loss_components[5]),
        train_b = torch.tensor(batch_train)
        lr = optimizer.param_groups[0]['lr']
        train_log = "train_log: %02d Cos_loss:%0.4e, Object_loss:%0.4e,Class_loss:%0.4e,Box_loss:%0.4e, XY_loss:%0.4e, Loss:%0.4e, HTLoss:%0.4e, lr:%.3e\n" % (
            epoch, train_b[:, 0].mean().item(), train_b[:, 1].mean().item(), train_b[:, 2].mean().item(),
            train_b[:, 3].mean().item(), train_b[:, 4].mean().item(), train_b[:, 5].mean().item(), train_b[:, 6].mean().item(), lr)
        print(train_log)
        fout_log.write(train_log)
        fout_log.flush()
        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            model.eval()  # Set model to evaluation mode
            vloss, ii = 0, 0
            batch_val = []
            for _, imgs, targets, htmap in tqdm.tqdm(validation_dataloader, desc="Validating"):

                # imgs = Variable(imgs.type(Tensor), requires_grad=False)
                imgs = imgs.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    outputs, htm = model(imgs)  # [1, 10647, 25]
                    val_loss, val_loss_components = criterion(outputs, targets)
                    vloss += val_loss
                    ii += 1
                    batch_val.append([float(val_loss_components[0]), float(val_loss_components[1]), float(val_loss_components[2]),
                                        float(val_loss_components[3]), float(val_loss_components[4]),
                                        float(val_loss_components[5])])

            
            val_b = torch.tensor(batch_val)
            vloss = val_b[:, 1].mean().item()#
            #vloss = vloss / ii

            scheduler.step(vloss)
            validation_log = "Val_log: %02d Cos_loss:%0.4e, Object_loss:%0.4e,Class_loss:%0.4e,Box_loss:%0.4e, XY_loss:%0.4e, Loss:%0.4e\n" % (
                epoch, val_b[:, 0].mean().item(), val_b[:, 1].mean().item(), val_b[:, 2].mean().item(),
                val_b[:, 3].mean().item(), val_b[:, 4].mean().item(), val_b[:, 5].mean().item())
            if vloss < min_loss:
                min_loss = vloss
                checkpoint_path = f"checkpoints2/best_ckpt_{epoch}.pth"
                print(f"---- Saving checkpoint to: '{checkpoint_path}' loss:'{min_loss}'----")
                torch.save(model.state_dict(), checkpoint_path)

            print(validation_log)
            fout_log.write(validation_log)
            fout_log.flush()
    fout_log.close()


if __name__ == "__main__":

    run()
