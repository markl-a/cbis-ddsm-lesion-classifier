# -*- coding: utf-8 -*-
"""best.pt 的存 / 載。權重一律存 CPU 張量，並帶齊重建所需 metadata。

載入時只靠本檔即可重建模型並在 CPU 上推論，不依賴 notebook 狀態。
"""
from __future__ import annotations

import platform
from pathlib import Path

import torch

from data import get_transforms
from model import build_model
from backbones import build_backbone, SUPPORTED as _BACKBONE_ARCHS


def _package_versions() -> dict:
    # 一律轉成純 str：torch/torchvision 的 __version__ 是 TorchVersion 物件，
    # 直接存會讓 torch.load(weights_only=True) 因未允許的 global 而失敗。
    vers = {"python": platform.python_version()}
    for mod in ("torch", "torchvision", "numpy", "pandas", "sklearn", "PIL"):
        try:
            m = __import__(mod)
            vers[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            vers[mod] = "n/a"
    return vers


def save_checkpoint(path, model, *, img_size, class_names, threshold,
                    best_epoch, best_metric, best_metric_name,
                    split_seed, fold, history, manifest_fp, arch="efficientnet_b0",
                    tta=False, extra=None):
    """把最佳模型存成規格化的 best.pt（CPU 張量）。arch 指定 backbone（預設 efficientnet_b0）。"""
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "model_state": cpu_state,
        "arch": str(arch),
        "tta": bool(tta),
        "num_outputs": 1,
        "img_size": int(img_size),
        "class_names": list(class_names),
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_metric_name": str(best_metric_name),
        "split_seed": int(split_seed),
        "fold": int(fold),
        "history": history,
        "package_versions": _package_versions(),
        "manifest_fingerprint": manifest_fp,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    return path


SUPPORTED_ARCHS = set(_BACKBONE_ARCHS)   # efficientnet_b0/v2_s, convnext_tiny, resnet50, densenet121


def load_for_inference(path, device=None):
    """載入 best.pt，回傳 (model, meta)。預設在 CPU 上重建，不依賴 notebook 狀態。

    優先以 weights_only=True 載入（安全）；舊版 torch 不支援該參數時退回一般載入。
    依 ckpt['arch'] 用 build_backbone 重建；arch 不在支援清單時丟 ValueError。
    """
    device = device or torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")                    # 舊版 torch 無此參數
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # 保險：含非張量 metadata
    arch = ckpt.get("arch")
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"不支援的模型架構: {arch}")
    model = build_backbone(arch, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_image(model, ckpt, image_path, device=None):
    """用載入的模型對單張影像推論，回傳良/惡與惡性機率。"""
    from PIL import Image
    device = device or torch.device("cpu")
    tf = get_transforms(train=False, img_size=ckpt.get("img_size", 224))
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x)).squeeze()
    if ckpt.get("tta"):   # 與評估時一致：水平翻轉 TTA
        prob = (prob + torch.sigmoid(model(torch.flip(x, dims=[3]))).squeeze()) / 2
    prob = float(prob.item())
    thr = ckpt.get("threshold", 0.5)
    names = ckpt.get("class_names", ["benign", "malignant"])
    label = names[1] if prob >= thr else names[0]
    return {"label": label, "prob_malignant": float(prob),
            "threshold": float(thr)}
