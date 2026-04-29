# -*- coding: utf-8 -*-
"""Model definitions for the teacher and student networks."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """InsightFace-style ResNet used as the teacher backbone."""

    def __init__(self, block=IBasicBlock, layers=[3, 13, 30, 3], embedding_size=512, dropout=0.0):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(embedding_size=512, **kwargs):
    """Build the IResNet-100 teacher network."""

    return IResNet(IBasicBlock, [3, 13, 30, 3], embedding_size=embedding_size, **kwargs)


class ConvBNReLU(nn.Module):
    """Convolution followed by batch normalization and PReLU."""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = ConvBNReLU(in_c, in_c, 3, stride, 1, groups=in_c)
        self.pointwise = ConvBNReLU(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, in_c, out_c, stride=1, expand_ratio=2):
        super().__init__()
        self.use_residual = stride == 1 and in_c == out_c
        mid_c = in_c * expand_ratio
        self.block = nn.Sequential(
            ConvBNReLU(in_c, mid_c, 1, 1, 0),
            ConvBNReLU(mid_c, mid_c, 3, stride, 1, groups=mid_c),
            nn.Conv2d(mid_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileFaceNet(nn.Module):
    """MobileFaceNet student model with roughly 1M parameters."""

    def __init__(self, embedding_size=512):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBNReLU(64, 64, 3, 1, 1, groups=64)

        self.bottlenecks = nn.Sequential(
            InvertedResidual(64, 64, 2, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 128, 2, 4),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 128, 2, 4),
            InvertedResidual(128, 128, 1, 2),
            InvertedResidual(128, 128, 1, 2),
        )

        self.conv2 = ConvBNReLU(128, 512, 1, 1, 0)

        # MobileFaceNet uses global depthwise convolution instead of GAP.
        self.gdconv = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(512),
        )

        self.fc = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.gdconv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


def count_parameters(model):
    """Return the total and trainable parameter counts."""

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def load_teacher(weight_path, device="cuda"):
    """Load the pretrained teacher network and freeze its weights."""

    teacher = iresnet100(embedding_size=512)
    state_dict = torch.load(weight_path, map_location=device)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    try:
        teacher.load_state_dict(state_dict, strict=True)
        print("[Teacher] Weights loaded successfully (strict=True)")
    except Exception:
        teacher.load_state_dict(state_dict, strict=False)
        print("[Teacher] Weights loaded with partial key matching (strict=False)")

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    total, _ = count_parameters(teacher)
    print(f"[Teacher] Parameters: {total / 1e6:.1f}M")
    return teacher


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(2, 3, 112, 112).to(device)

    student = MobileFaceNet(512).to(device)
    student_out = student(dummy)
    total, trainable = count_parameters(student)
    print(f"[Student] Output shape: {student_out.shape}")
    print(f"[Student] Parameters: {total / 1e6:.2f}M")

    try:
        from thop import profile

        flops, params = profile(student, inputs=(dummy[:1],), verbose=False)
        print(f"[Student] FLOPs: {flops / 1e9:.2f}G")
    except Exception:
        print("[Hint] Install thop to measure FLOPs: pip install thop")

    teacher = iresnet100(512).to(device)
    teacher_out = teacher(dummy)
    total_teacher, _ = count_parameters(teacher)
    print(f"\n[Teacher] Output shape: {teacher_out.shape}")
    print(f"[Teacher] Parameters: {total_teacher / 1e6:.2f}M")
