from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

from utils.backbone import ResNet

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



def _resnet(arch, block, layers, modality, num_classes, **kwargs):
    model = ResNet(block, layers, modality, num_classes=num_classes, **kwargs)
    return model




class FCReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_C1, s_C2, use_relu=True):
        super(FCReg, self).__init__()
        self.use_relu = use_relu
        self.fc = nn.Linear(s_C1, s_C2)
        self.bn = nn.BatchNorm1d(s_C2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
        return x


class ImageNet(nn.Module):
    def __init__(self, args):
        super(ImageNet, self).__init__()
        self.dataset = 'CREMAD'
        self.args = args
        
        if args.image_arch == 'resnet18':
            layers = [2, 2, 2, 2]
        elif args.image_arch == 'resnet50':
            layers = [3, 4, 6, 3]
        else:
            layers = [2, 2, 2, 2]

        self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='visual', 
                                num_classes=6)
        
        self.head_video = nn.Linear(512, 6)

    def forward(self, x):
        B = x.size(0) 
        if self.dataset != 'CREMAD':
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        v = self.backbone(x)
        
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        
        v = v.permute(0, 2, 1, 3, 4) 
        
        v = F.adaptive_avg_pool3d(v, 1)
        
        features = torch.flatten(v, 1)
        logits = self.head_video(features)
        
        return logits, features, []

    def forward_head(self, feature_vector):
        logits = self.head_video(feature_vector)
        return logits
    
    def fc(self, feature_vector):
        return self.head_video(feature_vector)

    def forward_encoder(self, x):
        B = x.size(0) 

        if self.dataset != 'CREMAD':
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        v = self.backbone(x)
        
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        
        v = v.permute(0, 2, 1, 3, 4) 
        v = F.adaptive_avg_pool3d(v, 1)
        
        feature_vector = torch.flatten(v, 1)

        return feature_vector, []
    
class AudioNet(nn.Module):
    def __init__(self, args):
        super(AudioNet, self).__init__()
        self.args = args
        if args.audio_arch == 'resnet18':
            layers = [2, 2, 2, 2]
        elif args.audio_arch == 'resnet50':
            layers = [3, 4, 6, 3]
        else:
            layers = [2, 2, 2, 2] 

        self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='audio', 
                                num_classes=6)

        self.head_audio = nn.Linear(512, 6)

    def forward(self, x):
        x = self.backbone(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        
        features = torch.flatten(x, 1)

        logits = self.head_audio(features)
        
        return logits, features, []
    
    def fc(self, feature_vector):
        return self.head_audio(feature_vector)

    def forward_head(self, feature_vector):
        logits = self.head_audio(feature_vector)
        return logits

    def forward_encoder(self, x):
        x = self.backbone(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        
        feature_vector = torch.flatten(x, 1)
        return feature_vector, []