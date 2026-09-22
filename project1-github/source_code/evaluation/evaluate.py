from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classification.models import MODEL_NAMES, build_model
from utils.data import CLASS_NAMES, HAM10000Dataset, image_transform, lesion_split, load_metadata


WEIGHT_NAMES = {
    "ResNet50": "resnet50_best.pth", "VGG16": "vgg16_best.pth", "VGG19": "vgg19_best.pth",
    "DenseNet121": "densenet121_best.pth", "EfficientNetB0": "efficientnet_b0_best.pth",
    "CNN": "custom_cnn_best.pth",
}


def load_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


@torch.inference_mode()
def evaluate(model_name: str, data_dir: Path, weights_dir: Path, output_dir: Path, batch_size: int = 16):
    if not torch.backends.mps.is_available():
        raise RuntimeError("This project requires Apple MPS for full evaluation. MPS is not available.")
    device = torch.device("mps")
    frame = load_metadata(data_dir)
    _, validation = lesion_split(frame)
    loader = DataLoader(HAM10000Dataset(validation, image_transform(False)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    model = build_model(model_name)
    model.load_state_dict(load_state(weights_dir / WEIGHT_NAMES[model_name]))
    model.to(device).eval()
    truths, predictions = [], []
    for images, labels, _ in loader:
        predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
        truths.extend(labels.tolist())

    folder = output_dir / model_name
    folder.mkdir(parents=True, exist_ok=True)
    report = classification_report(truths, predictions, labels=range(7), target_names=CLASS_NAMES,
                                   zero_division=0, output_dict=True)
    json.dump(report, (folder / "classification_report.json").open("w"), indent=2)
    pd.DataFrame(report).transpose().to_csv(folder / "classification_report.csv")
    pd.DataFrame({"image_id": validation.image_id, "true": [CLASS_NAMES[i] for i in truths],
                  "predicted": [CLASS_NAMES[i] for i in predictions]}).to_csv(folder / "predictions.csv", index=False)
    matrix = confusion_matrix(truths, predictions, labels=range(7))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(folder / "confusion_matrix.csv")
    plt.figure(figsize=(8, 7)); plt.imshow(matrix, cmap="Blues"); plt.colorbar()
    plt.xticks(range(7), CLASS_NAMES, rotation=45, ha="right"); plt.yticks(range(7), CLASS_NAMES)
    threshold = matrix.max() / 2
    for row in range(7):
        for column in range(7):
            plt.text(column, row, matrix[row, column], ha="center", va="center",
                     color="white" if matrix[row, column] > threshold else "black")
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout(); plt.savefig(folder / "confusion_matrix.png", dpi=250); plt.close()
    print(f"{model_name}: accuracy={report['accuracy']:.4f}, device={device}, samples={len(validation)}")


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES + ["all"], default=["EfficientNetB0"])
    parser.add_argument("--data-dir", type=Path, default=project / "data")
    parser.add_argument("--weights-dir", type=Path, default=project / "weights")
    parser.add_argument("--output-dir", type=Path, default=project / "results/recomputed")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    names = MODEL_NAMES if "all" in args.models else args.models
    for name in names:
        evaluate(name, args.data_dir, args.weights_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
