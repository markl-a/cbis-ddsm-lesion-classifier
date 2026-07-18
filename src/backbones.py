# -*- coding: utf-8 -*-
"""可切換的 backbone 工廠（單一 logit 頭），供 v2 實驗比較不同架構。

支援：efficientnet_b0（baseline）、efficientnet_v2_s、convnext_tiny、
resnet50、densenet121。皆為 ImageNet 預訓練，classifier 換成單一輸出 logit。
全部走 torchvision，相容 torch-directml（無 CUDA-only 運算）。
"""
from __future__ import annotations

import torch.nn as nn

SUPPORTED = ("efficientnet_b0", "efficientnet_v2_s", "convnext_tiny",
             "resnet50", "densenet121")


def build_backbone(name: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    from torchvision import models

    if name not in SUPPORTED:
        raise ValueError(f"不支援的 backbone: {name}（可用: {SUPPORTED}）")

    def w(enum):
        return enum.IMAGENET1K_V1 if pretrained else None

    if name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=w(models.EfficientNet_B0_Weights))
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
    elif name == "efficientnet_v2_s":
        net = models.efficientnet_v2_s(weights=w(models.EfficientNet_V2_S_Weights))
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
    elif name == "convnext_tiny":
        net = models.convnext_tiny(weights=w(models.ConvNeXt_Tiny_Weights))
        # convnext classifier = Sequential(LayerNorm2d, Flatten, Linear)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, 1)
    elif name == "resnet50":
        net = models.resnet50(weights=w(models.ResNet50_Weights))
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif name == "densenet121":
        net = models.densenet121(weights=w(models.DenseNet121_Weights))
        net.classifier = nn.Linear(net.classifier.in_features, 1)
    return net


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
