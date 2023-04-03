import math

import torch
import torch.nn as nn
import numpy as np
from .utils import to_cpu


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    hp_diou = 0.6
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return (1-hp_diou)*iou - hp_diou*rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)

                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def bbox_cos(box1, box2, grid_xy, x1y1x2y2=True, eps=1e-9):
    COSxxyy = nn.CosineSimilarity(eps=eps)
    # COSxxyy = nn.CosineEmbeddingLoss()
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    grid_xy = grid_xy.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        t_x1, t_y1 = box2[0] + grid_xy[0].float(), box2[1] + grid_xy[1].float()
        p_x1, p_y1 = box1[0] + grid_xy[0].float(), box1[1] + grid_xy[1].float()

    with torch.no_grad():  # 去掉梯度，只计算值
        aain = []
        for i in range(0, t_x1.shape[0]//2):
            aain.append(2*i+1)
            aain.append(2 * i)
    t_x2 = t_x1[aain]
    t_y2 = t_y1[aain]

    t_vt = torch.cat(((t_x1-t_x2).view(-1, 1), (t_y1-t_y2).view(-1, 1)), 1)
    p_vt = torch.cat(((p_x1-t_x2).view(-1, 1), (p_y1-t_y2).view(-1, 1)), 1)

    # 计算距离
    alph1 = torch.norm(t_vt, 2, 1)
    alph2 = torch.norm(p_vt, 2, 1)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan((alph1-alph2) / torch.clamp(alph1, min=eps)), 2)
    # alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)

    # alph = 2*torch.min(alph1, alph2)/(alph1+alph2 + eps)
    alph = torch.min(alph1, alph2) / (torch.max(alph1, alph2) + eps)

    cos = COSxxyy(p_vt, t_vt)

    # return cos*alph
    return cos-v

class MeasureLoss(nn.Module):
    def __init__(self, n_classes=20, n_anchors=3, anchors=[[[105, 105], [85, 85], [65, 65]], [[50, 50], [40, 40], [30, 30]], [[17, 17], [12, 12], [7, 7]]], device=None):
        super(MeasureLoss, self).__init__()
        self.device = device
        self.strides = torch.tensor([32, 16, 8], device=device)
        image_size = 416
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        # self.yolo_anchors = torch.tensor([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45]], device=device)
        # self.yolo_anchors = torch.tensor([[8, 8], [16, 16], [32, 32], [64, 64], [128, 128]], device=device)
        self.yolo_anchors = torch.tensor(anchors, device=device)

    def build_target(self, pred, targets):
        tcls, tbox, indices, anch, grid_xy = [], [], [], [], []
        na, nt = 3, targets.shape[0]
        # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        gain = torch.ones(7, device=self.device)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        for i in range(na):
            shape = pred[i].shape
            anchs = (self.yolo_anchors[i] / self.strides[i]).float()
            # print(anchs)
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]
            t = targets * gain  # [5, 4, 7]

            # t = t.view(-1,7)
            ######  挑选anchor  ######
            if nt:
                r = t[:, :, 4:6] / anchs[:, None]  # [3,2,2]
                # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
                j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #倍数
                t = t[j.sum(dim=1) > 1, :].view(-1, 7)  # 选出满足成对的box
            else:
                t = targets[0]

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            # gwh = t[:, 4:6]  # grid wh
            ac_i = t[:, 6].long()
            gwh = (t[:, 4:6] + anchs[ac_i].float())/2.
            # print(gwh)
            # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
            gij = gxy.long()
            # Isolate x and y index dimensions
            gi, gj = gij.T  # grid xy indices

            # Convert anchor indexes to int
            a = t[:, 6].long()
            # Add target tensors for this yolo layer to the output lists
            # Add to index list and limit index range to prevent out of bounds
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
            tbox.append(torch.cat((gxy - gij.float(), gwh), 1))  # box
            # Add correct anchor for each target to the list
            anch.append(anchs[a])
            # Add class for each target to the list
            tcls.append(c)
            grid_xy.append(gij)

        return tcls, tbox, indices, anch, grid_xy

    def forward(self, output, targets=None):

        # print(output.shape)

        lcls, lbox, lobj, lcos = torch.zeros(1, device=self.device), \
                                 torch.zeros(1, device=self.device), \
                                 torch.zeros(1, device=self.device), \
                                 torch.zeros(1, device=self.device)
        lxy = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors, grid_xy = self.build_target(output, targets)
        # print(tbox)
        # BCExy = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor([1.0], device=self.device))
        BCExy = nn.BCELoss()
        # Define different loss functions classification
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0], device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0], device=self.device))

        for i in range(3):
            b, anchor, grid_j, grid_i = indices[i]
            tobj = torch.zeros_like(output[i][..., 0], device=self.device)
            num_targets = b.shape[0]

            if num_targets:
                ps = output[i][b, anchor, grid_j, grid_i]
                # print(ps.shape) # [10 25]

                pxy = ps[:, :2].sigmoid()
                pwh = torch.exp(ps[:, 2:4]) * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                lxy += BCExy(pxy[..., :2], tbox[i][..., :2])

                cos = bbox_cos(pbox.T, tbox[i], grid_xy[i], x1y1x2y2=False)  # [-1,1]
                lcos += (1.0 - cos).mean()

                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, DIoU=True)
                # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
                lbox += (1.0 - iou).mean()
                tobj[b, anchor, grid_j, grid_i] = cos.detach().clamp(0).type(tobj.dtype)
                if ps.size(1) - 5 > 1:
                    # Hot one class encoding
                    t = torch.zeros_like(ps[:, 5:], device=self.device)  # targets 全零背景
                    t[range(num_targets), tcls[i]] = 1
                    # Use the tensor to calculate the BCE loss
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Classification of the objectness the sequel
                # Calculate the BCE loss between the on the fly generated target and the network prediction
            lobj += BCEobj(output[i][..., 4], tobj)  # obj loss 这里没有加sigmoid

        lbox *= 0.01  # iou
        lcos *= 1.0  # cos
        lobj *= 0.5  # confidence
        lcls *= 0.5  # class
        lxy *= 0.01  # xy

        # print(lcos , lobj ,lcls , lxy , lbox)
        # Merge losses
        loss = lcos + lobj + lcls + lbox + lxy

        return loss, [to_cpu(lcos), to_cpu(lobj), to_cpu(lcls), to_cpu(lbox), to_cpu(lxy), to_cpu(loss)]
        # return loss, to_cpu(torch.cat((lcos, lobj, lcls, lbox, lxy, loss),dim=0))

# output = torch.randn((2,5,26,26,25))
# output = torch.randn((2,125,26,26))
# target = torch.tensor([[0.0000, 1.0000, 0.1001, 0.2494, 0.2003, 0.1848],
#                         [0.0000, 2.0000, 0.3230, 0.4515, 0.2003, 0.1848],
#                         [1.0000, 1.0000, 0.5494, 0.4090, 0.1533, 0.1929],
#                         [1.0000, 2.0000, 0.3373, 0.1980, 0.1533, 0.1929]], requires_grad=True)
# # target = torch.randn((2,6))
# l = MeasureLoss()
# loss, loss_components = l(output, target)