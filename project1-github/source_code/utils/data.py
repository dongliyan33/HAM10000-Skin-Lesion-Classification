from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SEED = 42


def load_metadata(data_dir: Path) -> pd.DataFrame:
    candidates = [data_dir / "HAM10000_metadata.csv", data_dir / "HAM10000_metadata"]
    metadata_path = next((path for path in candidates if path.exists()), None)
    if metadata_path is None:
        raise FileNotFoundError(f"HAM10000 metadata not found in {data_dir}")
    image_paths = {}
    for folder in data_dir.glob("HAM10000_images_part_*"):
        image_paths.update({path.stem: str(path) for path in folder.glob("*.jpg")})
    frame = pd.read_csv(metadata_path)
    frame["image_path"] = frame.image_id.map(image_paths)
    frame = frame.dropna(subset=["image_path"]).copy()
    frame["label"] = frame.dx.map(CLASS_TO_INDEX)
    return frame.reset_index(drop=True)


def lesion_split(frame: pd.DataFrame, validation_ratio: float = 0.2, seed: int = SEED):
    lesion_ids = frame.lesion_id.drop_duplicates().tolist()
    random.Random(seed).shuffle(lesion_ids)
    validation_ids = set(lesion_ids[: int(len(lesion_ids) * validation_ratio)])
    train = frame[~frame.lesion_id.isin(validation_ids)].reset_index(drop=True)
    validation = frame[frame.lesion_id.isin(validation_ids)].reset_index(drop=True)
    return train, validation


def image_transform(train: bool = False, size: int = 224):
    operations = [transforms.Resize((size, size))]
    if train:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                       transforms.RandomRotation(15), transforms.ColorJitter(brightness=.1, contrast=.1)]
    operations += [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
    return transforms.Compose(operations)


class HAM10000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform or image_transform(False)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        return self.transform(image), torch.tensor(int(row.label), dtype=torch.long), row.image_id
