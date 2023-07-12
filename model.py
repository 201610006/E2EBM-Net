import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from torch.autograd import Variable
from backbone.Resnet import resnet18, resnet34, resnet50
import numpy as np
from einops import rearrange

from functools import partial

nonlinearity = partial(F.relu, inplace=True)
# CBR
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding=(kernel_size - 1)//2)
        self.BN = nn.BatchNorm2d(out_channels)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN(x)
        x = self.Relu(x)
        return x
# CBL
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.BN = nn.BatchNorm2d(out_channels)
        self.Leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN(x)
        x = self.Leakyrelu(x)
        return x

# CBM -> mish=x*tanh(s(x)) s(x)=ln(1+e^x)
class CBM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN(x)
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

# SPP
class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.Maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5//2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9//2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13//2)

    def forward(self, x):
        x1 = self.Maxpool1(x)
        x2 = self.Maxpool2(x)
        x3 = self.Maxpool3(x)
        x = torch.cat((x,x1,x2,x3),dim=1)
        return x

# Res_unit
class Res_unit(nn.Module):
    def __init__(self, out_channels, nblocks=3, shortcut=True):
        super(Res_unit, self).__init__()
        self.CBM1 = CBM(out_channels,out_channels,kernel_size=1,stride=1)
        self.CBM2 = CBM(out_channels,out_channels,kernel_size=3,stride=1)
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(self.CBM1)
            resblock_one.append(self.CBM2)
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

# CSP
class CSP(nn.Module):
    def __init__(self, in_channels, out_channels, nblocks):
        super(CSP, self).__init__()
        self.CBM1 = CBM(in_channels, out_channels, kernel_size=1, stride=1)
        # self.CBM2 = CBM(in_channels, out_channels, kernel_size=1, stride=1)
        self.CBM3 = CBM(out_channels, out_channels, kernel_size=1, stride=1)
        self.Res_unit = Res_unit(out_channels, nblocks=nblocks)
        self.Conv1 = nn.Conv2d(2 * out_channels, out_channels, 1, 1)

    def forward(self, x):
        x1_0 = self.CBM1(x)
        x1 = self.Res_unit(x1_0)
        x1 = self.CBM3(x1)
        # x2 = self.CBM2(x)
        x = torch.cat([x1,x1_0], dim=1)
        x = self.Conv1(x)
        return x

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

def build_backbone(back_bone, pretrained=False):
    if back_bone == "resnet34":
        return resnet34(pretrained=pretrained)
    if back_bone == "resnet50":
        return resnet50(pretrained=pretrained)
    if back_bone == "resnet18":
        return resnet18(pretrained=pretrained)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! ")

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class Dila_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad, dilation, bias=True):
        super().__init__()

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation, bias=bias))
        self.conv.append(nn.BatchNorm2d(out_channels))
        self.conv.append(nn.ReLU(inplace=True))
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class Feature_fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv_c1 = Conv_Bn_Activation(out_channels, out_channels, 3, 1, 'relu')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1)
        # self.conv_c2 = Conv_Bn_Activation(out_channels, out_channels, 3, 1, 'relu')
    def forward(self, x):
        #for l in self.conv:
        #    x = l(x)
        x = self.conv1(x)
        x1 = self.conv_c1(x)
        x2 = self.conv2(x1)
        # x2 = self.conv_c2(x2)
        return x+x2

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)

        if inference:
            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class Decoupling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoupling, self).__init__()
        self.nblock = 1
        self.level1 = CSP(256, 128, self.nblock)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.level2 = CSP(256, 128, self.nblock)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.level3 = CSP(256, 128, self.nblock)

        self.level2_1 = CSP(512, 256, self.nblock)
        self.level1_1 = CSP(512, 256, self.nblock)
        self.out = nn.Conv2d(512, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x1 = self.level1(x)
        x2 = self.level2(self.conv1(x1))
        x3 = self.level3(self.conv2(x2))

        x2_1 = F.upsample(x3, size=(h//2, w//2), mode='bilinear')
        x2_1 = torch.cat([x2_1,x2], 1)
        x2_1 = self.level2_1(x2_1)

        x2_1 = F.upsample(x2_1, size=(h, w), mode='bilinear')
        x2_1 = torch.cat([x2_1, x1], 1)
        x2_1 = self.level1_1(x2_1)

        out = self.out(x2_1)

        return out


class Decouple_mds(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decouple_mds, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)

        self.conv2 = Conv_Bn_Activation(in_channels, in_channels, 3, 1, 'relu')

        self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        out = self.out(x)
        return out


class MeasureNet(nn.Module):
    def __init__(self, back_bone='resnet34',
                 pretrained=False,
                 anchors = 3,
                 nblock = 1,
                 num_classes = 20):
        super(MeasureNet, self).__init__()

        self.anchors = anchors
        self.no = num_classes + 5
        self.nblock = nblock

        self.back_bone = build_backbone(back_bone, pretrained)

        self.get_feature_c4 = CSP(512 + 256, 256, self.nblock)

        self.get_feature_c3 = CSP(256 + 128, 256, self.nblock)

        self.get_feature_c4_2 = CSP(256 + 256, 256, self.nblock)

        self.get_feature_c5 = CSP(256 + 512, 256, self.nblock)

        self.at_f_c5 = CBAM(256)
        self.at_f_c4 = CBAM(256)
        self.at_f_c3 = CBAM(256)

        self.level3 = Feature_fusion(512, 256)
        self.level4 = Feature_fusion(512, 256)
        self.level5 = Feature_fusion(512, 256)

        self.ht_c5 = Conv_Bn_Activation(512, 64, 1, 1, 'relu')
        self.ht_c4 = Conv_Bn_Activation(256, 64, 1, 1, 'relu')
        self.ht_c3 = Conv_Bn_Activation(128, 64, 1, 1, 'relu')
        self.ht_out = nn.Conv2d(64, 2, kernel_size=1, stride=1)

        self.out1 = Decouple_mds(256, self.no * 3)
        self.out2 = Decouple_mds(256, self.no * 3)
        self.out3 = Decouple_mds(256, self.no * 3)

    def forward(self, x):
        low_level_features = self.back_bone(x)
        c5 = low_level_features[0]  # [2, 512, 13, 13]
        c4 = low_level_features[1]  # [2, 256, 26, 26]
        c3 = low_level_features[2]  # [2, 128, 52, 52]
        c2 = low_level_features[3]  # [2, 64, 104, 104]


        feature_c4 = torch.cat([F.interpolate(c5, scale_factor=2, mode='bilinear'), c4], 1)
        feature_c4 = self.get_feature_c4(feature_c4) #[2, 512, 13, 13]

        feature_c3 = torch.cat([F.interpolate(feature_c4, scale_factor=2, mode='bilinear'), c3], 1)
        feature_c3 = self.get_feature_c3(feature_c3)

        feature_c4 = torch.cat([F.interpolate(feature_c3, scale_factor=0.5, mode='bilinear'), feature_c4], 1)
        feature_c4 = self.get_feature_c4_2(feature_c4)

        feature_c5 = torch.cat([F.interpolate(feature_c4, scale_factor=0.5, mode='bilinear'), c5], 1)
        feature_c5 = self.get_feature_c5(feature_c5)


        htm = F.interpolate(self.ht_c5(c5), scale_factor=4, mode='bilinear') + F.interpolate(self.ht_c4(c4), scale_factor=2, mode='bilinear') + self.ht_c3(c3)
        htm = torch.sigmoid(self.ht_out(htm))
        htm_out = F.interpolate(htm, scale_factor=2, mode='bilinear')
        htm = htm.sum(dim=1, keepdim=True)

        l3 = feature_c3 * htm
        l3_a = self.at_f_c3(feature_c3)
        out1 = self.level3(torch.cat([l3, l3_a], 1))
        out1 = self.out1(out1)

        l4 = feature_c4 * F.interpolate(htm, scale_factor=0.5, mode='bilinear')
        l4_a = self.at_f_c4(feature_c4)
        out2 = self.level4(torch.cat([l4, l4_a], 1))
        out2 = self.out2(out2)

        l5 = feature_c5 * F.interpolate(htm, scale_factor=0.25, mode='bilinear')
        l5_a = self.at_f_c5(feature_c5)
        out3 = self.level5(torch.cat([l5, l5_a], 1))
        out3 = self.out3(out3)

        bs, _, _, _ = x.shape
        out1 = out1.view(bs, self.anchors, self.no, out1.shape[-2], out1.shape[-1]).permute(0, 1, 3, 4, 2)
        out2 = out2.view(bs, self.anchors, self.no, out2.shape[-2], out2.shape[-1]).permute(0, 1, 3, 4, 2)
        out3 = out3.view(bs, self.anchors, self.no, out3.shape[-2], out3.shape[-1]).permute(0, 1, 3, 4, 2)

        return [out1, out2, out3], htm_out
