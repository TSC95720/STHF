import torchvision
import torch.nn as nn
from .gem_pool import GeneralizedMeanPooling
import torch
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.in_planes = 2048
        self.num_classes = num_classes

        self.image_encoder1 = RGB_Model()
        self.image_encoder2 = IR_Model()
        self.image_res1 = ResidualBlock1()
        self.image_res2 = ResidualBlock2()
        self.image_res3 = ResidualBlock3()
        self.image_res4 = ResidualBlock4()
        self.classifier = Classifier(self.num_classes)
        self.Gem = GeneralizedMeanPooling(norm=3.0)

        # Inter Modality Encoder
        self.inter_encoder1 = RGB_Model()
        self.inter_encoder2 = IR_Model()
        self.inter_res1 = ResidualBlock1()
        self.inter_res2 = ResidualBlock2()
        self.inter_res3 = ResidualBlock3()
        self.inter_res4 = ResidualBlock4()
        self.classifier_i = Classifier(self.num_classes)
        self.Gem_i = GeneralizedMeanPooling(norm=3.0)

        self.MAM1 = MAM_Block(256)
        self.MAM2 = MAM_Block(512)
        self.MAM3 = MAM_Block(1024)
        self.MAM4 = MAM_Block(2048)

        self.MFA1 = MFA_block(256)
        self.MFA2 = MFA_block(512)

        self.cross_att_1 = DualCrossAtten(1024)
        self.cross_att_2 = DualCrossAtten(2048)

    def forward(self, x1=None, x2=None, i1=None, i2=None):
        if x1 is not None and x2 is not None:
            image_features_map1 = self.image_encoder1(x1)
            image_features_map2 = self.image_encoder2(x2)
            image_features_maps = torch.cat([image_features_map1, image_features_map2], dim=0)

            inter_features_map1 = self.inter_encoder1(i1)
            inter_features_map2 = self.inter_encoder2(i2)
            inter_features_maps = torch.cat([inter_features_map1, inter_features_map2], dim=0)

            image_features_maps_res1 = self.image_res1(image_features_maps)
            inter_features_maps_res1 = self.inter_res1(inter_features_maps)
            image_features_maps_res1 = self.MAM1(image_features_maps_res1)
            image_features_maps_res1 = self.MFA1(image_features_maps_res1, inter_features_maps_res1)

            image_features_maps_res2 = self.image_res2(image_features_maps_res1)
            inter_features_maps_res2 = self.inter_res2(inter_features_maps_res1)
            image_features_maps_res2 = self.MAM2(image_features_maps_res2)
            image_features_maps_res2 = self.MFA2(image_features_maps_res2, inter_features_maps_res2)

            image_features_maps_res3 = self.image_res3(image_features_maps_res2)
            inter_features_maps_res3 = self.inter_res3(inter_features_maps_res2)

            image_features_maps_res3 = self.MAM3(image_features_maps_res3)
            image_features_maps_res3, inter_features_maps_res3 = self.cross_att_1(image_features_maps_res3,inter_features_maps_res3)

            image_features_maps = self.image_res4(image_features_maps_res3)
            inter_features_maps = self.inter_res4(inter_features_maps_res3)

            image_features_maps = self.MAM4(image_features_maps)
            image_features_maps, inter_features_maps = self.cross_att_2(image_features_maps, inter_features_maps)

            image_features = self.Gem(image_features_maps)
            image_features = self.tem_pool(image_features)

            inter_features = self.Gem_i(inter_features_maps)
            inter_features = self.tem_pool(inter_features)

            features, cls_scores, _ = self.classifier(image_features)
            features_i, cls_scores_i, _ = self.classifier_i(inter_features)

            return features, cls_scores, features_i, cls_scores_i

        elif x1 is not None and x2 is None:
            image_features_map1 = self.image_encoder1(x1)
            inter_features_map1 = self.inter_encoder1(i1)

            image_features_map1_res1 = self.image_res1(image_features_map1)
            inter_features_map1_res1 = self.inter_res1(inter_features_map1)
            image_features_map1_res1 = self.MAM1(image_features_map1_res1)
            image_features_map1_res1 = self.MFA1(image_features_map1_res1, inter_features_map1_res1)

            image_features_map1_res2 = self.image_res2(image_features_map1_res1)
            inter_features_map1_res2 = self.inter_res2(inter_features_map1_res1)
            image_features_map1_res2 = self.MAM2(image_features_map1_res2)
            image_features_map1_res2 = self.MFA2(image_features_map1_res2, inter_features_map1_res2)

            image_features_map1_res3 = self.image_res3(image_features_map1_res2)
            inter_features_map1_res3 = self.inter_res3(inter_features_map1_res2)

            image_features_map1_res3 = self.MAM3(image_features_map1_res3)
            image_features_map1_res3, inter_features_map1_res3 = self.cross_att_1(image_features_map1_res3, inter_features_map1_res3)

            image_features_map1 = self.image_res4(image_features_map1_res3)
            inter_features_map1 = self.inter_res4(inter_features_map1_res3)

            image_features_map1 = self.MAM4(image_features_map1)
            image_features_map1, inter_features_map1 = self.cross_att_2(image_features_map1, inter_features_map1)

            image_features1 = self.Gem(image_features_map1)
            image_features1 = self.tem_pool(image_features1)
            _, _, test_features1 = self.classifier(image_features1)

            return test_features1

        elif x1 is None and x2 is not None:
            image_features_map2 = self.image_encoder2(x2)
            inter_features_map2 = self.inter_encoder2(i2)

            image_features_map2_res1 = self.image_res1(image_features_map2)
            inter_features_map2_res1 = self.inter_res1(inter_features_map2)
            image_features_map2_res1 = self.MAM1(image_features_map2_res1)
            image_features_map2_res1 = self.MFA1(image_features_map2_res1, inter_features_map2_res1)

            image_features_map2_res2 = self.image_res2(image_features_map2_res1)
            inter_features_map2_res2 = self.inter_res2(inter_features_map2_res1)
            image_features_map2_res2 = self.MAM2(image_features_map2_res2)
            image_features_map2_res2 = self.MFA2(image_features_map2_res2, inter_features_map2_res2)

            image_features_map2_res3 = self.image_res3(image_features_map2_res2)
            inter_features_map2_res3 = self.inter_res3(inter_features_map2_res2)

            image_features_map2_res3 = self.MAM3(image_features_map2_res3)
            image_features_map2_res3, inter_features_map2_res3 = self.cross_att_1(image_features_map2_res3, inter_features_map2_res3)

            image_features_map2 = self.image_res4(image_features_map2_res3)
            inter_features_map2 = self.inter_res4(inter_features_map2_res3)

            image_features_map2 = self.MAM4(image_features_map2)
            image_features_map2, inter_features_map2 = self.cross_att_2(image_features_map2, inter_features_map2)

            image_features2 = self.Gem(image_features_map2)
            image_features2 = self.tem_pool(image_features2)
            _, _, test_features2 = self.classifier(image_features2)

            return test_features2

    def tem_pool(self,features,t=6):
        features = features.squeeze()
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        return features


class RGB_Model(nn.Module):
    def __init__(self):
        super(RGB_Model, self,).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, rgb):
        rgb_features_map = self.resnet_conv(rgb)
        return rgb_features_map


class IR_Model(nn.Module):
    def __init__(self):
        super(IR_Model, self,).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, ir):
        ir_features_map = self.resnet_conv(ir)
        return ir_features_map


class ResidualBlock1(nn.Module):
    def __init__(self):
        super(ResidualBlock1, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet_conv = nn.Sequential(resnet.layer1)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map


class ResidualBlock2(nn.Module):
    def __init__(self):
        super(ResidualBlock2, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet_conv = nn.Sequential(resnet.layer2)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map


class ResidualBlock3(nn.Module):
    def __init__(self):
        super(ResidualBlock3, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet_conv = nn.Sequential(resnet.layer3)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map


class ResidualBlock4(nn.Module):
    def __init__(self):
        super(ResidualBlock4, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet_conv = nn.Sequential(resnet.layer4)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map


class SELayer(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        att = y.expand_as(x)

        return att


class MAM_Block(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(MAM_Block, self).__init__()
        self.Ca = SELayer(channel, reduction_ratio)
        self.IN = nn.InstanceNorm2d(channel)
        self.IN.apply(weights_init_kaiming)

    def forward(self, f):

        att_f = self.Ca(f)
        att_f_m = 1 - att_f
        f_s = self.IN(f)
        f_ms = f * att_f + f_s * att_f_m

        return f_ms


class CNL(nn.Module):
    def __init__(self, channel):
        super(CNL, self).__init__()
        self.channel = channel
        self.inter_channel = channel // 2

        self.g = nn.Conv2d(self.channel, self.inter_channel, kernel_size=1, stride=1, padding=0)  # 生成 v
        self.theta = nn.Conv2d(self.channel, self.inter_channel, kernel_size=1, stride=1, padding=0)  # 生成 q

        self.phi = nn.Conv2d(self.channel, self.inter_channel, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(nn.Conv2d(self.inter_channel, self.channel, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm2d(channel), )

        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.inter_channel, -1)  # v

        theta_x = self.theta(x_h).view(B, self.inter_channel, -1)  # q
        phi_x = self.phi(x_l).view(B, self.inter_channel, -1).permute(0, 2, 1)  # k

        energy = torch.matmul(theta_x, phi_x)  # q * k
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.inter_channel, *x_l.size()[2:])
        W_y = self.W(y)  # 恢复维度
        z = W_y + x_h  # 残差

        return z


class PNL(nn.Module):
    def __init__(self, channel, reduc_ratio=2):
        super(PNL, self).__init__()
        self.channel = channel
        self.inter_channel = channel // 2
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.channel, self.inter_channel, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.channel, self.inter_channel, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.channel, self.inter_channel, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channel), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.inter_channel, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.inter_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.inter_channel, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_channel, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, channel):
        super(MFA_block, self).__init__()

        self.CNL = CNL(channel)
        self.PNL = PNL(channel)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


class DualCrossAtten(nn.Module):
    def __init__(self, in_channels):
        super(DualCrossAtten, self).__init__()
        self.cross_atten = NonLocalBlockND(in_channels)

    def forward(self, x1, x2):
        B, c, h, w = x1.size()  # (96,512,36,18)
        seq_len = 6
        x1 = x1.view(int(B//seq_len), c, seq_len, h, w)  # [16, 512, 6, 36, 18] [b, c, t, h, w]
        x2 = x2.view(int(B//seq_len), c, seq_len, h, w)

        x1 = self.cross_atten(x1, x2)

        x1 = x1.view(int(B),c, h, w)
        x2 = x2.view(int(B),c, h, w)
        return x1, x2


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential( conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                                    bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x,x2):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x2).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return features, cls_score, self.l2_norm(bn_features)
