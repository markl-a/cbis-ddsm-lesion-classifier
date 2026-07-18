# -*- coding: utf-8 -*-
"""模型與裝置選擇（前處理轉換定義於 data.py 的 get_transforms / SquarePad）。

- Backbone：ImageNet 預訓練 EfficientNet-B0，classifier 換成單一輸出 logit。
- 裝置：先做一次真實 forward/backward probe 才選 DirectML，否則退回 CPU（CUDA 亦支援）。
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def build_model(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0，輸出單一 logit（配 BCEWithLogitsLoss）。

    pretrained=True 時直接使用 ImageNet 權重；建構失敗不靜默降級（讓錯誤浮現）。
    """
    from torchvision import models

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_feats = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_feats, 1)
    return net


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def probe_directml(verbose: bool = True) -> bool:
    """在 DirectML 上實跑一次 EfficientNet forward+backward，成功且 loss 有限才回 True。"""
    try:
        import torch_directml
        if torch_directml.device_count() == 0:
            return False
        dev = torch_directml.device()
        net = build_model(pretrained=False).to(dev)
        x = torch.randn(2, 3, 224, 224, device=dev)
        y = torch.tensor([[0.0], [1.0]], device=dev)
        loss = nn.BCEWithLogitsLoss()(net(x), y)
        loss.backward()
        ok = bool(torch.isfinite(loss).item())
        if verbose:
            print(f"  DirectML probe: loss={loss.item():.4f} finite={ok}")
        return ok
    except Exception as e:
        if verbose:
            print("  DirectML probe 失敗，改用 CPU:", e)
        return False


def get_device(require_gpu: bool = False, verbose: bool = True):
    """裝置順序：DirectML(probe 通過) -> CUDA -> CPU。

    require_gpu=True 時，若 DirectML probe 失敗且無 CUDA 則丟 RuntimeError（不退回 CPU）。
    設環境變數 CBIS_FORCE_CPU=1 可強制用 CPU（CI / 與其他 GPU 工作並行時用）。
    """
    if os.environ.get("CBIS_FORCE_CPU") == "1" and not require_gpu:
        return torch.device("cpu")
    if probe_directml(verbose=verbose):
        import torch_directml
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError("找不到可用的 GPU 後端（DirectML probe 失敗且無 CUDA）")
    return torch.device("cpu")
