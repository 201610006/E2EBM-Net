import time
import torch
from .utils import to_cpu, xywh2xyxy, rescale_boxes
# from utils import to_cpu, xywh2xyxy, rescale_boxes
import numpy as np
from PIL import Image
import os
import cv2
from itertools import chain

# print(os.getcwd())
# print(os.path.abspath('.'))

def make_grid(nx=26, ny=26):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def make_anchor_grid():
    anchors = [[128, 128], [48, 48], [16, 16]]
    anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
    anchors = anchors.clone().view(1, -1, 1, 1, 2)
    return anchors

def point_filtering_by_thres(prediction, conf_thres=0.05):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
         1、去掉置信度低的点 < conf_thres
         2、找到成对的类别点，如类别：1和2， 2和3，从高置信度高到底排序
    """
    img_size = 416
    b, a, w, h, cont = prediction.shape # []
    stride = img_size // w
    grid = make_grid(nx=26, ny=26)
    anchor_grid = make_anchor_grid()

    prediction[..., 0:2] = (prediction[..., 0:2].sigmoid() + grid) * stride  # xy
    prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * anchor_grid  # wh

    prediction[..., 4:] = prediction[..., 4:].sigmoid()

    prediction = prediction.contiguous().view(b, -1, cont)  # =[1, 10647, 25] [2, 10647, 25] batch box value
    max_res = 300

    time_limit = 10.0  # seconds to quit after
    # multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    multi_label = False

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]
    # print('prediction==', prediction.shape)
    for xi, x in enumerate(prediction):  # image index, image inference
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
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


# 进行加权微调
def adj_point(tpoint, points_list):
    cxy, cw = 0, 0
    for ttp in points_list:
        if box_ii(ttp[:4], tpoint[:4]) > 0.8:
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

def select_measure_point(image_path, img_size, detections):
    img = np.array(Image.open(image_path).convert("L"))

    # print(detections.shape)
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

# dd = torch.tensor([[ 8.9423e+02,  1.0588e+02,  1.0696e+03,  2.8638e+02,  5.9019e-01,
#           4.0000e+00],
#         [ 8.0563e+02,  1.3579e+01,  1.1571e+03,  3.8069e+02,  5.7097e-01,
#           4.0000e+00],
#         [ 8.1335e+02,  1.8486e+01,  1.1626e+03,  3.7144e+02,  5.4791e-01,
#           4.0000e+00],
#         [ 9.0337e+02,  1.1306e+02,  1.0717e+03,  2.7475e+02,  5.4188e-01,
#           4.0000e+00],
#         [ 9.3915e+02,  1.4940e+02,  1.0254e+03,  2.4271e+02,  4.7287e-01,
#           4.0000e+00],
#         [ 3.8082e+02, -3.8941e+01,  7.5585e+02,  3.2618e+02,  4.3569e-01,
#           3.0000e+00],
#         [ 9.4235e+02,  1.5089e+02,  1.0318e+03,  2.2918e+02,  3.2085e-01,
#           4.0000e+00],
#         [ 4.8550e+02,  5.7204e+01,  6.5457e+02,  2.2595e+02,  1.9964e-01,
#           3.0000e+00],
#         [ 2.6293e+02,  1.8089e+02,  5.8066e+02,  4.9790e+02,  1.9929e-01,
#           3.0000e+00]])
# img = r'../data/sample_gugu1/20220418_095756.avi_4159.jpg'
# select_measure_point(img,416,dd)

