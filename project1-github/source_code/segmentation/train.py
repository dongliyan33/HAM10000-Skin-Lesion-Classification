"""End-to-end HAM10000 lesion segmentation with a compact U-Net.

This is independent of classification checkpoints and never overwrites them.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = ROOT / "new_training_runs" / "SegmentationUNet"
SEED = 42


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = DoubleConv(3, base); self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4); self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2); self.bridge = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2); self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.dec1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        x = self.bridge(self.pool(e4)); x = self.dec4(torch.cat([self.up4(x), e4], 1))
        x = self.dec3(torch.cat([self.up3(x), e3], 1)); x = self.dec2(torch.cat([self.up2(x), e2], 1))
        return self.output(self.dec1(torch.cat([self.up1(x), e1], 1)))


class SegmentationDataset(Dataset):
    def __init__(self, frame, size=224, augment=False):
        self.frame = frame.reset_index(drop=True); self.size = size; self.augment = augment

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        mask = Image.open(row.mask_path).convert("L")
        image = TF.resize(image, [self.size, self.size], antialias=True)
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        if self.augment and random.random() < 0.5: image, mask = TF.hflip(image), TF.hflip(mask)
        if self.augment and random.random() < 0.5: image, mask = TF.vflip(image), TF.vflip(mask)
        if self.augment:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle); mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = (TF.to_tensor(mask) > 0.5).float()
        return image, mask, row.image_id


def paired_frame(data_dir=DATA):
    metadata_path = data_dir / "HAM10000_metadata"
    if not metadata_path.exists():
        metadata_path = data_dir / "HAM10000_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    images = {}
    for folder in data_dir.glob("HAM10000_images_part_*"):
        images.update({path.stem: str(path) for path in folder.glob("*.jpg")})
    masks = {path.name.replace("_segmentation.png", ""): str(path) for path in (data_dir / "HAM10000_segmentations_lesion_tschandl").glob("*_segmentation.png")}
    metadata["image_path"] = metadata.image_id.map(images); metadata["mask_path"] = metadata.image_id.map(masks)
    return metadata.dropna(subset=["image_path", "mask_path"]).reset_index(drop=True)


def split_by_lesion(frame):
    lesion_ids = frame.lesion_id.drop_duplicates().tolist(); random.Random(SEED).shuffle(lesion_ids)
    val_ids = set(lesion_ids[:int(0.2 * len(lesion_ids))])
    return frame[~frame.lesion_id.isin(val_ids)].copy(), frame[frame.lesion_id.isin(val_ids)].copy()


def soft_dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum((1, 2, 3))
    return 1 - ((2 * intersection + smooth) / (probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + smooth)).mean()


def batch_metrics(logits, targets):
    predictions = torch.sigmoid(logits) >= 0.5; targets = targets.bool()
    intersection = (predictions & targets).sum((1, 2, 3)).float()
    pred_sum = predictions.sum((1, 2, 3)).float(); target_sum = targets.sum((1, 2, 3)).float()
    union = (predictions | targets).sum((1, 2, 3)).float(); eps = 1e-7
    return {
        "dice": ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item(),
        "iou": ((intersection + eps) / (union + eps)).mean().item(),
        "precision": ((intersection + eps) / (pred_sum + eps)).mean().item(),
        "recall": ((intersection + eps) / (target_sum + eps)).mean().item(),
    }


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None; model.train(training)
    totals = {key: 0.0 for key in ["loss", "dice", "iou", "precision", "recall"]}; count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images); loss = nn.functional.binary_cross_entropy_with_logits(logits, masks) + soft_dice_loss(logits, masks)
            if training: optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            metrics = batch_metrics(logits, masks); batch = images.size(0); count += batch; totals["loss"] += loss.item() * batch
            for key, value in metrics.items(): totals[key] += value * batch
    return {key: value / count for key, value in totals.items()}


@torch.inference_mode()
def save_examples(model, loader, device, path, maximum=4):
    images, masks, ids = next(iter(loader)); logits = model(images.to(device)).cpu(); predictions = torch.sigmoid(logits) >= 0.5
    count = min(maximum, len(images)); figure, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for row in range(count):
        image = (images[row] * std + mean).permute(1, 2, 0).clamp(0, 1)
        axes[row, 0].imshow(image); axes[row, 0].set_title(ids[row])
        axes[row, 1].imshow(masks[row, 0], cmap="gray"); axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(predictions[row, 0], cmap="gray"); axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(image); axes[row, 3].imshow(predictions[row, 0], alpha=0.35, cmap="Reds"); axes[row, 3].set_title("Overlay")
        for axis in axes[row]: axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=250); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--size", type=int, default=224); parser.add_argument("--learning-rate", type=float, default=1e-3); args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = paired_frame(args.data_dir); train_frame, val_frame = split_by_lesion(frame)
    train_frame[["image_id", "lesion_id"]].to_csv(OUTPUT / "train_split.csv", index=False); val_frame[["image_id", "lesion_id"]].to_csv(OUTPUT / "validation_split.csv", index=False)
    train_loader = DataLoader(SegmentationDataset(train_frame, args.size, True), args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SegmentationDataset(val_frame, args.size, False), args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "val_precision": [], "val_recall": [], "epoch_time_seconds": []}; best = -1.0
    print(f"Pairs={len(frame)} train={len(train_frame)} val={len(val_frame)} device={device}")
    for epoch in range(1, args.epochs + 1):
        start = time.time(); train = run_epoch(model, train_loader, device, optimizer); val = run_epoch(model, val_loader, device)
        history["train_loss"].append(train["loss"]); history["val_loss"].append(val["loss"])
        for key in ["dice", "iou", "precision", "recall"]: history[f"val_{key}"].append(val[key])
        history["epoch_time_seconds"].append(time.time() - start)
        if val["dice"] > best:
            best = val["dice"]
            torch.save({"model_state_dict": model.state_dict(), "base_channels": 16, "image_size": args.size, "best_val_dice": best, "epoch": epoch}, OUTPUT / "best_model.pth")
        history.update({"model": "UNet", "epochs": list(range(1, epoch + 1)), "best_val_dice": best, "train_samples": len(train_frame), "validation_samples": len(val_frame)})
        with (OUTPUT / "training_history.json").open("w") as file: json.dump(history, file, indent=2)
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={train['loss']:.4f} val_loss={val['loss']:.4f} Dice={val['dice']:.4f} IoU={val['iou']:.4f}")
    if args.epochs:
        checkpoint = torch.load(OUTPUT / "best_model.pth", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        final = run_epoch(model, val_loader, device); json.dump(final, (OUTPUT / "segmentation_metrics.json").open("w"), indent=2); save_examples(model, val_loader, device, OUTPUT / "segmentation_examples.png")
        epochs = history["epochs"]; plt.figure(figsize=(9, 5)); plt.plot(epochs, history["train_loss"], label="Train loss"); plt.plot(epochs, history["val_loss"], label="Validation loss")
        plt.xlabel("Epoch"); plt.ylabel("BCE + Dice loss"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUTPUT / "loss_curves.png", dpi=300); plt.close()
        plt.figure(figsize=(9, 5)); plt.plot(epochs, history["val_dice"], label="Dice"); plt.plot(epochs, history["val_iou"], label="IoU")
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUTPUT / "metric_curves.png", dpi=300); plt.close()
        print(f"Best checkpoint and report assets saved to {OUTPUT}")


if __name__ == "__main__": main()
