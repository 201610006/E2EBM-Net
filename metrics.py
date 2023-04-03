#! /usr/bin/env python3

from __future__ import division

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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def make_grid(nx=26, ny=26):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])#, indexing='ij')
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def make_anchor_grid(idx):
    anchors = [[[105, 105], [85, 85], [65, 65]], [[50, 50], [40, 40], [30, 30]], [[17, 17], [12, 12], [7, 7]]]
    anchors = torch.tensor(list(chain(*anchors[idx]))).float().view(-1, 2)
    anchors = anchors.clone().view(1, -1, 1, 1, 2)
    return anchors

def find_two_points(predictions, target, cls_s=3, cls_e=4):
    img_size = 416
    pdist = nn.PairwiseDistance(p=2)
    res = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        b, a, w, h, cont = prediction.shape # [b 3 h w 25]
        stride = img_size // w
        grid = make_grid(nx=w, ny=h)
        prediction[..., 0:2] = (prediction[..., 0:2].sigmoid() + grid) * stride  # xy
        prediction[..., 4:] = prediction[..., 4:].sigmoid() # 置信度和类别
        prediction = prediction.contiguous().view(b, -1, cont)  # =[1, 10647, 25] [2, 10647, 25] batch box value
        conf, j = prediction[:, :, 5:].max(2, keepdim=True)
        y = torch.cat((prediction[:, :, 4:5], (j).float(), prediction[:, :, :2]), 2)

        for k in range(b):
            p_s = y[k][y[k][:,1]==cls_s]
            if len(p_s)>0:
                p_s = p_s[p_s[:, 1].argsort(descending=True)][0]
            else:
                p_s = torch.zeros(y[k].shape)[0]

            p_e = y[k][y[k][:,1]==cls_e]
            if len(p_e)>0:
                p_e = p_e[p_e[:, 1].argsort(descending=True)][0]
            else:
                p_e = torch.zeros(y[k].shape)[0]
            t = target[target[:,0]==k]
            d1 = pdist(p_s[2:4],p_e[2:4])
            d2 = pdist(t[0][2:4],t[1][2:4])
            res.append(abs(d1-d2))
    return sum(res)/len(res)

def point_filtering_by_thres(predictions, conf_thres=0.2):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
         1、去掉置信度低的点 < conf_thres
         2、找到成对的类别点，如类别：1和2， 2和3，从高置信度高到底排序
    """
    img_size = 416
    pds = []
    for i in range(3):
        prediction = predictions[i]
        b, a, w, h, cont = prediction.shape # [b 3 h w 25]
        stride = img_size // w
        grid = make_grid(nx=w, ny=h)
        anchor_grid = make_anchor_grid(i)
        prediction[..., 0:2] = (prediction[..., 0:2].sigmoid() + grid) * stride  # xy
        prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * anchor_grid  # wh
        prediction[..., 4:] = prediction[..., 4:].sigmoid()
        prediction = prediction.contiguous().view(b, -1, cont)  # =[1, 10647, 25] [2, 10647, 25] batch box value
        pds.append(prediction)
    pds = torch.cat(pds, 1)
    print("=====", pds.shape)
    max_res = 300

    # multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    multi_label = False

    output = [torch.zeros((0, 6), device="cpu")] * pds.shape[0]
    # print('prediction==', prediction.shape)
    for xi, x in enumerate(pds):  # image index, image inference
        # print('x1==', x.shape)
        x = x[x[..., 4] > conf_thres]  # confidence [10647, 25] 第一次筛选, 置信度低不管类别对不对都去掉
        # If none remain process next image
        if not x.shape[0]:
            continue
        # print('x2==', x.shape)
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf [16, 20]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2) []
        box = xywh2xyxy(x[:, :4])
        # box = x[:, :4]
        # print("x2===",x.shape)
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # print((x[:, 5:] > conf_thres*0.01).nonzero(as_tuple=False).T)
            i, j = (x[:, 5:] > conf_thres*0.1).nonzero(as_tuple=False).T #第二次筛选，类别概率*置信度，nonzero得到非零元素位置索引，选出多个位置类别
            # print(i.shape, j.shape) # [212940]
            # [212940, 4] [212940, 1] [212940, 1] -> [212940, 6]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)  #box(x,y,x,y) 类别概率*置信度(0,1) 类别索引[0-19]

        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True) #[10647, 1] [10647, 1] 类别概率*置信度中选出1位最大的
            # print(conf.shape, j.shape)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres*0.01]
        # print('x3==', x.shape)
        x = x[x[:, 4].argsort(descending=True)]

        if x.shape[0] > max_res:
            x = x[:max_res]
        if x.shape[0] < 2:
            continue

        output[xi] = to_cpu(x)
        # print('x4==', x[:10])
    return output

def box_ii(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1)
    area2 = box_area(box2)
    inter = (torch.min(box1[2:4], box2[2:4]) -
             torch.max(box1[:2], box2[:2])).clamp(0).prod(0)
    io = inter / area1
    io2 = inter / area2
    # return max(io, io2)
    return io

def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def rescale_boxes_ag(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    boxes[:, 0] =(boxes[:, 0])/orig_w * unpad_w  + pad_x // 2# ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] =(boxes[:, 1])/orig_h * unpad_h+ pad_y // 2
    boxes[:, 2] =(boxes[:, 2])/orig_w * unpad_w+ pad_x // 2
    boxes[:, 3] =(boxes[:, 3])/orig_h * unpad_h+ pad_y // 2

    # Rescale bounding boxes to dimension of original image
    # boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    # boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    # boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    # boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def adj_point(tpoint, points_list):
    cxy, cw = 0, 0
    for ttp in points_list:
        if box_ii(ttp[:4], tpoint[:4]) > 0.9:
            cxy += ttp[4] * (ttp[:2] + ttp[2:4]) / 2
            cw += ttp[4]
    cxy = cxy / cw - (tpoint[:2] + tpoint[2:4]) / 2
    tpoint[:2] = tpoint[:2] + cxy
    tpoint[2:4] = tpoint[2:4] + cxy
    return tpoint

def liantong2(img, p0, pi): #图像 置信度最大的点 需要匹配的点
    '''特点：通常找不到的点就是边界比较模糊的点.
    规则视具体任务而定，羊水是垂直连通域中均匀的黑色不能有其它器官或组织隔断；
    股骨连通域是有一条明亮的连通路径，非连通路径则测量起点或终点不连续
    '''
    #把图像网格化
    h,w = img.shape[:2] #1008,1280
    # 构建一个稍微大一点的框框检测，防止测量对象是弯曲的超出测量框框
    band_height, band_width = int(h/30), int(w/30)
    point_circle_h, point_circle_w = int(h/30), int(w/30)
    alph = 1.25

    p0_center = ((p0[:2] + p0[2:4]) / 2).int()
    pi_center = ((pi[:2] + pi[2:4]) / 2).int()


    st_gx, st_gy = torch.min(p0_center,pi_center)
    ed_gx, ed_gy = torch.max(p0_center,pi_center)

    if ed_gy>h or ed_gx>w:
        return False

    if ed_gy - st_gy < h/10:
        # band_height = int(h/20)
        alph = 1
    if ed_gx - st_gx < w/10:
        # band_width = int(w/20)
        alph = 1

    newimg = np.zeros((img.shape[:2]))
    newimg[st_gy-band_height:ed_gy+band_height, st_gx-band_width:ed_gx+band_width] = img[st_gy-band_height:ed_gy+band_height, st_gx-band_width:ed_gx+band_width]
    thr = img[st_gy:ed_gy, st_gx:ed_gx].mean()
    newimg = newimg - alph*thr

    # kernel = np.ones((5, 5), np.float32) / 25
    # newimg = cv2.filter2D(newimg, -1, kernel)
    newimg = cv2.blur(newimg, (5, 5))

    newimg = np.where(newimg > 0, 255, 0).astype(np.uint8)

    # 弥补点位
    newimg[p0_center[1]-point_circle_h:p0_center[1]+point_circle_h, p0_center[0]-point_circle_w:p0_center[0]+point_circle_w] = 255
    newimg[pi_center[1] - point_circle_h:pi_center[1] + point_circle_h,pi_center[0] - point_circle_w:pi_center[0] + point_circle_w] = 255

    retval, labels = cv2.connectedComponents(newimg)

    # print('vv==',labels[p0_center[1],p0_center[0]],  labels[pi_center[1],pi_center[0]])
    if labels[p0_center[1],p0_center[0]] ==  labels[pi_center[1],pi_center[0]]:
        return True
    else:
        return False

def pick_another_point(img, tpoint, points_list):
    out_put = torch.zeros(tpoint.shape)
    lt_flag = False
    for tt in points_list:
        if liantong2(img, tpoint, tt):
            # print("iou = ", box_ii(tpoint, tt))
            # 如果匹配到了，还要进行交集检测，预防距离很短的错误亮点
            if box_ii(tpoint, tt) == 0.:
                out_put = tt + 0.
                lt_flag = True
                break
    if lt_flag:
        out_put = adj_point(out_put, points_list)
    return out_put

def select_measure_point(image_path, img_size, detections):
    img = np.array(Image.open(image_path).convert("L"))

    print(detections.shape)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    # print(detections)
    # 置信度最高点
    st_points = torch.zeros((2, 6), device="cpu")
    # 启动降分查找的阈值
    thr_rev = 0.2
    st_points[0] = detections[0] + 0.
    st_list = detections[detections[:, -1] == st_points[0][-1]]
    # 匹配类别对
    classpair = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                              [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]])
    pair = classpair[1][classpair[0, :] == st_points[0, -1]]
    ed_list = detections[detections[:, -1] == pair]
    if len(ed_list)==0:
        filename = os.path.basename(image_path)
        print("Not found! It is: ", filename)
        return
    st_points[0] = adj_point(st_points[0], st_list)
    st_points[1] = pick_another_point(img, st_points[0], ed_list)
    if st_points[1][0] == 0: # 反向查找
        st_points[0] = ed_list[0] + 0.
        st_points[0] = adj_point(st_points[0], ed_list)
        st_points[1] = pick_another_point(img, st_points[0], st_list) + 0.
    if st_points[1][0] == 0: # 依旧没找到可以启动降分继续查找
        candidats_list = detections[detections[:, 4] > thr_rev]
        for it, candidat in enumerate(candidats_list):
            # 如果是其他类的话，不用管交集（这里暂时没考虑）
            if box_ii(candidat[:4], st_list[0][:4]) < 0.1 and box_ii(candidat[:4], ed_list[0][:4]) < 0.1:
                st_points[0] = candidat + 0.
                pair = classpair[1][classpair[0, :] == st_points[0, -1]]
                st_list2 = detections[detections[:, -1] == st_points[0][-1]]
                ed_list2 = detections[detections[:, -1] == pair]
                st_points[0] = adj_point(st_points[0], st_list2)
                st_points[1] = pick_another_point(img, st_points[0], ed_list2) + 0.
                if st_points[1][0] != 0:
                    break
    if st_points[1][0] == 0: # 可以放弃了
        filename = os.path.basename(image_path)
        print("Not found 2! It is: ", filename)
        return
    return st_points

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
    # if args.model_weights:
    #     pth = args.model_weights
    #     model = load_checkpoint_model(model, pth, device)
    #
    # if args.model_weights:
    #     pth = args.model_weights
    #     # model = load_gpu_parallel_model(model, pth, device)
    #     model = load_checkpoint_model(model, pth, device)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        batch_size=1,
        img_size=416,
        n_cpu=4)

    print("vali:",len(validation_dataloader))

    # mkdirs("mylogs")
    # save_dir = os.path.join("mylogs", datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_.logs"
    # fout_log = open(save_dir, 'w')
    model.eval()
    points_val = []
    label_val= []

    img_path = []
    iimg = []

    for imgpath, imgs, targets, htm in tqdm.tqdm(validation_dataloader, desc="Validating"):
        # print(imgpath, imgs.shape, targets)
        # imgs = Variable(imgs.type(Tensor), requires_grad=False)
        imgs = imgs.to(device)
        targets = targets.to(device)

        iimg.append(np.array(imgs[0].permute(1,2,0)))

        with torch.no_grad():
            outputs, phtm = model(imgs)  # [1, 10647, 25]
            detections = point_filtering_by_thres(outputs, args.conf_thres)
            rela = find_two_points(outputs, targets)
            print("rela ====", rela)
        points_val.extend(detections)  # [b 2 6] x,y,w,h,conf,class
        label_val.append(targets)

        img_path.extend(imgpath)

    toll_num = len(points_val)
    sect_err = 0
    abs_e = []
    rel_e = []
    for (image_path, gt, f_detections) in zip(img_path, label_val, points_val):
        unique_labels = f_detections[:, -1].cpu().unique()
        # print(unique_labels)
        n_cls_preds = len(unique_labels)

        if n_cls_preds > 1:
            two_point = select_measure_point(image_path, 416, f_detections)
            if two_point is not None:
                print(gt.shape)
                img = np.array(Image.open(image_path))
                two_point = rescale_boxes_ag(two_point, 416, img.shape[:2])


                pd = np.linalg.norm(np.array([(two_point[0][0]+two_point[0][2])/2, (two_point[0][1]+two_point[0][3])/2])
                                    - np.array([(two_point[1][0]+two_point[1][2])/2, (two_point[1][1]+two_point[1][3])/2]))
                gd = np.linalg.norm(np.array([gt[0][2]*416,gt[0][3]*416])-np.array([gt[1][2]*416,gt[1][3]*416]))
                abs_e.append(abs(pd-gd))
                rel_e.append(abs(pd-gd)/gd)

            else:
                sect_err = sect_err + 1

    print("sect_err =", sect_err/toll_num, "abs_e =", np.mean(abs_e), "rel_e =", np.mean(rel_e))




if __name__ == "__main__":

    run()

