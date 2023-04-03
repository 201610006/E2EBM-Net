import glob
import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from matplotlib.ticker import NullLocator

#查看数据集
def img_poinst_show():
    #羊水
    imgpath = r"E:\amniotic_fluid\yangshui_data2\images"
    annotation = r'E:\amniotic_fluid\yangshui_data2\annotations.json'
    #股骨
    # imgpath = r"E:\femur_length\gonggugugu_1997_gao\image"
    # annotation = r'E:\femur_length\gonggugugu_1997_gao\annotations.json'
    with open(annotation,'r',encoding='utf8') as fp:
        jd = json.load(fp)
        yangshui = jd["annotations"]
        n = 0
        for key, item in yangshui.items():
            print(key, item["annotations"][0]["vertex"])
            points = np.array(item["annotations"][0]["vertex"])
            # img = cv2.imdecode(np.fromfile(os.path.join(imgpath,key.strip()), dtype=np.uint8), -1)
            img = np.array(Image.open(os.path.join(imgpath,key.strip())).convert('RGB'), dtype=np.uint8)
            h,w,_ = img.shape
            print(w,h)
            plt.scatter(points[0,0],points[0,1], color="skyblue", s=10)
            plt.scatter(points[1, 0], points[1, 1], color="skyblue", s=10)
            plt.imshow(img)
            plt.show()
            n = n + 1
            if n > 10:
                break
# img_poinst_show()

def show_oneimg(imgpath):
    img = np.array(Image.open(imgpath).convert('RGB'), dtype=np.uint8)
    h, w, _ = img.shape
    label_dir = imgpath.replace("images","labels")
    label_file = os.path.splitext(label_dir)[0] + '.txt'
    boxes = np.loadtxt(label_file).reshape(-1, 5)
    plt.scatter(boxes[0, 1]*w, boxes[0, 2]*h, color="skyblue", s=10)
    plt.scatter(boxes[1, 1]*w, boxes[1, 2]*h, color="skyblue", s=10)
    plt.imshow(img)
    plt.show()

pt=["/images/00205c8db3d8432fbe18e35a42f04714.jpg",
"/images/0094a7040c524d7b81e491d9356c500b.jpg",
"/images/00a752fe6a40419db4d9f7b3bb4bbb05.jpg",
"/images/00e6c683644241a2be70f0e877d1cf21.jpg",
"/images/010cb981337e45b385559c3eb78171de.jpg",
"/images/013051f40d2f4c6693ea5b4dd760c1d5.jpg",
"/images/0156842a44a44447a9040fbcba4ee54f.jpg",
"/images/01b73e254b3c46fa960b216b15a8d206.jpg",
"/images/01c6fd552cb04c268a5a853c77dc8529.jpg",
"/images/01dee572131145bf983e6f40598e2fb3.jpg"]

fl = [r"E:\femur_length\gonggugugu_1997_gao\images\20220418_095756.avi_5921.jpg",
      r"E:\femur_length\gonggugugu_1997_gao\images\20220418_095756.avi_4255.jpg",
      r"E:\femur_length\gonggugugu_1997_gao\images\20220418_095756.avi_5893.jpg",
      r"E:\femur_length\gonggugugu_1997_gao\images\20220419_112606.avi_26666.jpg"]
# show_oneimg(pt[3])

# 构造自己的数据集
def fluid_data_pro():
    trainpath=r"E:\amniotic_fluid\yangshui_data2\train.txt"
    trainlist = open(trainpath,'w',encoding='utf8')
    labelbox = r"E:\amniotic_fluid\yangshui_data2\labels"
    traintxt = '../data/yangshui/images/'
    imgpath = r"E:\amniotic_fluid\yangshui_data2\images"
    with open(r'E:\amniotic_fluid\yangshui_data2\annotations.json','r',encoding='utf8') as fp:
        jd = json.load(fp)
        yangshui = jd["annotations"]
    for key, item in yangshui.items():
        points = np.array(item["annotations"][0]["vertex"])
        pt = traintxt+key.strip()
        trainlist.writelines(pt+'\n')
        img = np.array(Image.open(os.path.join(imgpath, key.strip())).convert('RGB'), dtype=np.uint8)
        h,w,_ = img.shape

        isExists = os.path.join(labelbox,os.path.splitext(key.strip())[0]+'.txt')
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(isExists)
        with open(isExists, 'w', encoding='utf8') as wbox:
            if points[0,1]>points[1,1]:
                wbox.writelines("1 " + str(points[1,0]/w) + " " + str(points[1,1]/h) + " " + str((points[0,1]-points[1,1])*0.5/w)+ " " + str((points[0,1]-points[1,1])*0.5/h)+"\n")
                wbox.writelines("2 " + str(points[0,0]/w) + " " + str(points[0,1]/h) + " " + str((points[0,1]-points[1,1])*0.5/w)+ " " + str((points[0,1]-points[1,1])*0.5/h))
            else:
                wbox.writelines("1 " + str(points[0,0]/w) + " " + str(points[0,1]/h) + " " + str((points[1,1]-points[0,1])*0.5/w)+ " " + str((points[1,1]-points[0,1])*0.5/h)+"\n")
                wbox.writelines("2 " + str(points[1,0]/w) + " " + str(points[1,1]/h) + " " + str((points[1,1]-points[0,1])*0.5/w)+ " " + str((points[1,1]-points[0,1])*0.5/h))
        wbox.close()
    trainlist.close()
    #     print(key.strip(),points)
# fluid_data_pro()

# 股骨长测量
def femur_data_pro():
    trainpath=r"E:\femur_length\gonggugugu_1997_gao\train.txt"
    trainlist = open(trainpath,'w',encoding='utf8')
    labelbox = r"E:\femur_length\gonggugugu_1997_gao\labels"
    traintxt = '../data/gugu/images/'
    imgpath = r"E:\femur_length\gonggugugu_1997_gao\images"
    with open(r'E:\femur_length\gonggugugu_1997_gao\annotations.json','r',encoding='utf8') as fp:
        jd = json.load(fp)
        fl = jd["annotations"]
    for key, item in fl.items():
        points = np.array(item["annotations"][0]["vertex"])
        pt = traintxt+key.strip()
        trainlist.writelines(pt+'\n')
        img = np.array(Image.open(os.path.join(imgpath, key.strip())).convert('RGB'), dtype=np.uint8)
        h,w,_ = img.shape

        isExists = os.path.join(labelbox,os.path.splitext(key.strip())[0]+'.txt')
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(isExists)
        with open(isExists, 'w', encoding='utf8') as wbox:
            if points[1,0]<points[0,0]:
                wbox.writelines("3 " + str(points[1,0]/w) + " " + str(points[1,1]/h) + " " + str((points[0,0]-points[1,0])*0.5/w)+ " " + str((points[0,0]-points[1,0])*0.5/h)+"\n")
                wbox.writelines("4 " + str(points[0,0]/w) + " " + str(points[0,1]/h) + " " + str((points[0,0]-points[1,0])*0.5/w)+ " " + str((points[0,0]-points[1,0])*0.5/h))
            else:
                wbox.writelines("3 " + str(points[0,0]/w) + " " + str(points[0,1]/h) + " " + str((points[1,0]-points[0,0])*0.5/w)+ " " + str((points[1,0]-points[0,0])*0.5/h)+"\n")
                wbox.writelines("4 " + str(points[1,0]/w) + " " + str(points[1,1]/h) + " " + str((points[1,0]-points[0,0])*0.5/w)+ " " + str((points[1,0]-points[0,0])*0.5/h))
        wbox.close()
    trainlist.close()

# femur_data_pro()


def show_points():
    annotation = r'E:\femur_length\testset\predict_points.json'
    image_path = r"E:\femur_length\testset\肱骨800"
    output_path = r"E:\femur_length\testset\pre_800"
    with open(annotation, 'r', encoding='utf8') as fp:
        jd = json.load(fp)
    for i in range(len(jd)):
        img_name = list(jd[i].keys())[0]
        it = list(jd[i].values())[0]
        print(img_name, it)
        if it is None:
            continue
        points = np.array(it)
        img = np.array(Image.open(os.path.join(image_path, img_name)))

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        x_11, y_11, x_12, y_12, conf1, cls_pred1 = points[0]
        x_21, y_21, x_22, y_22, conf2, cls_pred2 = points[1]

        plt.plot((x_12 + x_11) / 2, (y_12 + y_11) / 2, 'o', markersize=2, color='red')  # color1)
        # plt.text(
        #     (x_12 + x_11) / 2 + 5,
        #     (y_12 + y_11) / 2,
        #     s='start',
        #     color="white",
        #     verticalalignment="top",
        #     bbox={"color": 'red', "pad": 0})
        rect1 = plt.Rectangle((x_11, y_11), x_12 - x_11, y_12 - y_11, linestyle='dashed', fill=False, edgecolor='red')
        ax.add_patch(rect1)

        plt.plot((x_22 + x_21) / 2, (y_22 + y_21) / 2, 'o', markersize=2, color='green')  # color2)
        # plt.text(
        #     (x_22 + x_21) / 2 + 5,
        #     (y_22 + y_21) / 2,
        #     s='end',
        #     color="white",
        #     verticalalignment="top",
        #     bbox={"color": 'green', "pad": 0})
        rect2 = plt.Rectangle((x_21, y_21), x_22 - x_21, y_22 - y_21, linestyle='dashed', fill=False, edgecolor='green')
        ax.add_patch(rect2)
        plt.plot([(x_12 + x_11) / 2, (x_22 + x_21) / 2], [(y_12 + y_11) / 2, (y_22 + y_21) / 2], linewidth=1, linestyle='-',
                 color='red')
        print(f"\t+ Label: {cls_pred1} | Confidence: {conf1.item():0.4f}")
        print(f"\t+ Label: {cls_pred2} | Confidence: {conf2.item():0.4f}")
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        output_p = os.path.join(output_path, img_name)
        plt.savefig(output_p, bbox_inches="tight", pad_inches=0.0)
        plt.close("all")

# show_points()

def show_seglabs():
    image = r"E:\HC18\training_set\18_HC.png"
    imglab = r"E:\HC18\training_set\18_HC_Annotation.png"
    img = cv2.imread(image, 0)
    lab = cv2.imread(imglab, 0)
    # lab = cv2.applyColorMap(lab, cv2.COLORMAP_JET)
    sss = img + lab*0.6
    plt.figure()
    plt.imshow(sss)
    plt.show()
# show_seglabs()

# p  = glob.glob(r"E:\femur_length\gonggugugu_1997_gao\images\*")
# for itm in p:
#     mask = itm.replace("images","mask")
#     isExists = os.path.exists(mask)
#     if not isExists:
#         # image = cv2.imread(mask.replace("jpg","png"), 1)
#         # cv2.imwrite(mask, image)
#         print("---")
#######################胎盘数据########################3
def show_taipan(masks):
    # image = "custom/images/1_4.png"
    # imglab = "custom/seglabs/1_4.png"

    images = masks.replace("masks","images")
    img = cv2.imread(images, 0)
    lab = cv2.imread(masks, 0)
    print(np.sum(lab, 0).max())
    print(img.shape, lab.shape)
    sss = img + lab*255*0.6
    plt.figure()
    plt.imshow(sss)
    plt.show()

# images = r'E:\taipanqidai_jingxi_mask5319\train_dataset\masks\22W_1.2.410.200001.1.1103.1950654136.3.20170509.1101037296.674.109.jpg'
# images = r'E:\taipanqidai_jingxi_mask5319\train_dataset\masks\25W_1.3.12.2.1107.5.5.2.213422.30000017062906152000000000092.jpg'
# images = r'E:\taipanqidai_jingxi_mask5319\train_dataset\masks\24W_1.2.276.0.26.1.1.1.2.2017.83.32083.276959.12451840.jpg'
#
# show_taipan(images)


def generate_target(joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = 4
    sigma = 2
    heatmap_size = np.array([26,26])
    image_size = np.array([416,416])
    joints_weight = 1
    use_different_joints_weight = True

    target_type = 'gaussian'
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    assert target_type == 'gaussian', \
        'Only support gaussian map now!'

    if target_type == 'gaussian':
        target = np.zeros((num_joints,
                           heatmap_size[1],
                           heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(num_joints):
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if use_different_joints_weight:
        target_weight = np.multiply(target_weight, joints_weight)

    return target, target_weight


imgp = r'E:\amniotic_fluid\yangshui_data1\coco_ys\images\images\000000010002.jpg'
pp = r'E:\amniotic_fluid\yangshui_data1\coco_ys\images\labels\000000010002.txt'
# img = cv2.imdecode(np.fromfile(imgp, dtype=np.uint8), -1)
img = cv2.imread(imgp, 0)
h,w = img.shape
f = open(pp, 'r')
points = f.read().splitlines()
p0 = points[0].split(' ')[1:3]
p1 = points[1].split(' ')[1:3]
p0=list(map(float,p0))
p1=list(map(float,p1))
print(p0,p1,w,h)
plt.figure()
fig, ax = plt.subplots(1)
ax.imshow(img)
plt.plot([p0[0]*w, p1[0]*w],[p0[1]*h, p1[1]*h], 'o', markersize=10, color='green')

plt.show()