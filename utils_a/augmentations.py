import imgaug.augmenters as iaa
from torchvision import transforms
from .ttransforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),  #锐化
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),#仿射变换
            iaa.AddToBrightness((-60, 40)),#RGB颜色扰动
            iaa.AddToHue((-10, 10)),#RGB颜色扰动 增長或減小色相和飽和度
            iaa.Fliplr(0.5),#左右翻转，1是上下翻转
        ])


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])


AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(), #绝对lab
    # DefaultAug(),     #增强img
    PadSquare(),      #补边
    RelativeLabels(), #相对lab
    ToTensor(),       #格式
])

DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

