from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classification.models import MODEL_NAMES, build_model
from utils.data import HAM10000Dataset, image_transform, lesion_split, load_metadata


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Optional classifier retraining; not needed to reproduce supplied results.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data-dir", type=Path, default=project / "data")
    parser.add_argument("--output-dir", type=Path, default=project / "new_training_runs")
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for full training.")
    torch.manual_seed(42); device = torch.device("mps")
    train_frame, validation_frame = lesion_split(load_metadata(args.data_dir))
    train_loader = DataLoader(HAM10000Dataset(train_frame, image_transform(True)), args.batch_size, shuffle=True)
    val_loader = DataLoader(HAM10000Dataset(validation_frame, image_transform(False)), args.batch_size)
    model = build_model(args.model, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate); criterion = nn.CrossEntropyLoss()
    history = {"epochs": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}; best = -1
    run_dir = args.output_dir / args.model; run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        started = time.time(); model.train(); train_loss = train_correct = train_count = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels); loss.backward(); optimizer.step()
            train_loss += loss.item() * len(labels); train_correct += (logits.argmax(1) == labels).sum().item(); train_count += len(labels)
        model.eval(); val_loss = val_correct = val_count = 0
        with torch.inference_mode():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device); logits = model(images); loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels); val_correct += (logits.argmax(1) == labels).sum().item(); val_count += len(labels)
        values = (train_loss/train_count, 100*train_correct/train_count, val_loss/val_count, 100*val_correct/val_count)
        history["epochs"].append(epoch)
        for key, value in zip(["train_loss", "train_accuracy", "val_loss", "val_accuracy"], values): history[key].append(value)
        if values[3] > best:
            best = values[3]; torch.save(model.state_dict(), run_dir / "best_model.pth")
        json.dump(history | {"best_val_accuracy": best}, (run_dir / "training_history.json").open("w"), indent=2)
        print(f"Epoch {epoch}/{args.epochs}: val_accuracy={values[3]:.2f}% time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
