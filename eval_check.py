"""Diagnose a trained checkpoint on the CIFAKE test set.

Prints: saved val acc, overall accuracy, confusion matrix, and mean P(FAKE)
for truly-FAKE vs truly-REAL images. If mean P(FAKE) is HIGHER for REAL
images, the label mapping is flipped. If accuracy is ~50%, the model is
just undertrained (2 epochs) and needs more training.

    python eval_check.py --data ./data/CIFAKE --batch 256
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T
from tqdm import tqdm

import config
from ai_detector.model import build_model
from train import flip_fake_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/CIFAKE")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = config.resolve_device()
    pt = os.path.join(config.WEIGHTS_DIR, "cifake_efficientnet_b0.pt")
    ckpt = torch.load(pt, map_location="cpu")
    print(f"checkpoint saved val acc: {ckpt.get('acc', '?')}")
    backbone = ckpt.get("backbone", "efficientnet_b0")

    model = build_model(backbone, pretrained=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval().to(device)

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(config.IMAGE_SIZE),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=list(config.IMAGENET_MEAN), std=list(config.IMAGENET_STD)),
    ])
    ds = datasets.ImageFolder(os.path.join(args.data, "test"),
                              transform=tf, target_transform=flip_fake_target)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # rows = true (REAL=0, FAKE=1), cols = predicted
    conf = torch.zeros(2, 2, dtype=torch.long)
    sum_fake_prob = torch.zeros(2)
    n = torch.zeros(2)
    with torch.inference_mode():
        for x, y in tqdm(ld, desc="eval", unit="batch"):
            x = x.to(device)
            probs = model(x).sigmoid().reshape(-1).float().cpu()  # P(FAKE)
            pred = (probs >= 0.5).long()
            for t, p, pr in zip(y.tolist(), pred.tolist(), probs.tolist()):
                conf[t][p] += 1
                sum_fake_prob[t] += pr
                n[t] += 1

    total = conf.sum().item()
    acc = (conf[0][0] + conf[1][1]).item() / total
    print(f"\ntest accuracy: {acc:.4f} ({total} images)")
    print("confusion [true REAL/FAKE x pred REAL/FAKE]:")
    print(f"  true REAL: pred REAL={conf[0][0]}  pred FAKE={conf[0][1]}")
    print(f"  true FAKE: pred REAL={conf[1][0]}  pred FAKE={conf[1][1]}")
    print(f"mean P(FAKE) on true REAL images: {sum_fake_prob[0] / n[0]:.4f}  (should be LOW)")
    print(f"mean P(FAKE) on true FAKE images: {sum_fake_prob[1] / n[1]:.4f}  (should be HIGH)")
    if acc < 0.2:
        print("DIAGNOSIS: labels are FLIPPED (accuracy far below chance).")
    elif acc < 0.65:
        print("DIAGNOSIS: model is UNDERTRAINED - train more epochs.")
    else:
        print("DIAGNOSIS: model is OK - the 2 hand-picked images were anecdotes.")


if __name__ == "__main__":
    main()
