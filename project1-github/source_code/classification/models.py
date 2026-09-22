from __future__ import annotations

import torch.nn as nn
from torchvision import models


MODEL_NAMES = ["ResNet50", "VGG16", "VGG19", "DenseNet121", "EfficientNetB0", "CNN"]


class SkinLesionCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_model(name: str, num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if name == "CNN":
        return SkinLesionCNN(num_classes)
    if name == "ResNet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name in {"VGG16", "VGG19"}:
        model = models.vgg16(weights=weights) if name == "VGG16" else models.vgg19(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    elif name == "DenseNet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model
