# -*- coding: utf-8 -*-
"""Dataset utilities for RecordIO training data and .bin evaluation sets."""

import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class MXFaceDataset(Dataset):
    """Read face training data stored in MXNet RecordIO format."""

    def __init__(self, rec_path, idx_path, transform=None):
        super().__init__()
        self.transform = transform
        self.idx_list = []

        with open(idx_path, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.idx_list.append(int(parts[0]))

        import mxnet as mx

        self.record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
        header, _ = mx.recordio.unpack(self.record.read_idx(0))
        if header.flag > 0:
            self.start_idx = int(header.label[0])
            self.num_samples = int(header.label[1]) - self.start_idx
        else:
            self.start_idx = 1
            self.num_samples = len(self.idx_list) - 1

        print(f"[Dataset] Loaded {self.num_samples} training images")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        import mxnet as mx

        idx = index + self.start_idx
        record = self.record.read_idx(idx)
        header, img_bytes = mx.recordio.unpack(record)

        img = mx.image.imdecode(img_bytes).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(header.label)

        if self.transform:
            img = self.transform(img)

        return img, label


class FaceDatasetFromImages(Dataset):
    """Fallback dataset that reads images from unpacked directories."""

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = []

        for label_name in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = int(label_name)
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.samples.append((img_path, label))

        print(f"[Dataset] Loaded {len(self.samples)} unpacked training images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label


def load_bin(bin_path, image_size=(112, 112)):
    """Load a standard face-verification .bin file."""

    with open(bin_path, "rb") as file:
        try:
            bins, issame_list = pickle.load(file, encoding="bytes")
        except Exception:
            file.seek(0)
            bins, issame_list = pickle.load(file)

    num_pairs = len(issame_list)
    images = []

    for i in range(2 * num_pairs):
        img_bytes = bins[i]
        img = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
                image_size[0], image_size[1], 3
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    print(f"[Validation] Loaded {num_pairs} pairs from {bin_path}")
    return images, issame_list


def get_train_transform():
    """Data augmentation used during student training."""

    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_val_transform():
    """Preprocessing used at validation time."""

    return T.Compose(
        [
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_train_loader(rec_path, idx_path, batch_size=128, num_workers=4):
    """Build the RecordIO training dataloader."""

    dataset = MXFaceDataset(rec_path, idx_path, transform=get_train_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    import config

    for name in config.VAL_TARGETS:
        bin_path = os.path.join(config.VAL_DATA_DIR, f"{name}.bin")
        if os.path.exists(bin_path):
            images, issame = load_bin(bin_path)
            print(f"  {name}: {len(images)} images, {len(issame)} pairs")
        else:
            print(f"  {name}: file not found")
