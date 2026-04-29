# -*- coding: utf-8 -*-
"""Evaluation utilities for LFW, CFP-FP, and AgeDB-30."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

import config
from dataset import get_val_transform, load_bin
from model import MobileFaceNet, count_parameters


def extract_features(model, images, batch_size=64, device="cuda"):
    """Extract normalized embeddings for a list of images."""

    model.eval()
    transform = get_val_transform()
    features = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_tensor = torch.stack([transform(img) for img in batch_images]).to(device)

        with torch.no_grad():
            feat = model(batch_tensor)

        feat = F.normalize(feat, p=2, dim=1)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_accuracy(features, issame_list, nfolds=10):
    """Compute verification accuracy with the standard 10-fold protocol."""

    num_pairs = len(issame_list)
    issame = np.array(issame_list)
    feat1 = features[0::2]
    feat2 = features[1::2]
    scores = np.sum(feat1 * feat2, axis=1)

    kfold = KFold(n_splits=nfolds, shuffle=False)
    accuracies = []
    thresholds = []
    indices = np.arange(num_pairs)

    for train_idx, test_idx in kfold.split(indices):
        train_scores = scores[train_idx]
        train_labels = issame[train_idx]

        best_acc = 0.0
        best_threshold = 0.0
        for threshold in np.arange(-1, 1, 0.001):
            predictions = train_scores > threshold
            acc = np.mean(predictions == train_labels)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        test_scores = scores[test_idx]
        test_labels = issame[test_idx]
        predictions = test_scores > best_threshold
        acc = np.mean(predictions == test_labels)

        accuracies.append(acc)
        thresholds.append(best_threshold)

    return np.mean(accuracies), np.std(accuracies), np.mean(thresholds)


def validate_single(model, bin_path, device="cuda"):
    """Evaluate one benchmark stored in .bin format."""

    if not os.path.exists(bin_path):
        return None

    images, issame_list = load_bin(bin_path)
    features = extract_features(model, images, batch_size=64, device=device)
    acc, std, threshold = compute_accuracy(features, issame_list)
    return acc


def validate_all(model, data_dir, val_targets, device="cuda"):
    """Evaluate all configured validation benchmarks."""

    results = {}
    for name in val_targets:
        bin_path = os.path.join(data_dir, f"{name}.bin")
        acc = validate_single(model, bin_path, device)
        if acc is not None:
            results[name] = acc
    return results


def compute_tar_at_far(features, issame_list, target_far=1e-3):
    """Compute TAR at a target FAR for verification benchmarks."""

    issame = np.array(issame_list)
    feat1 = features[0::2]
    feat2 = features[1::2]
    scores = np.sum(feat1 * feat2, axis=1)

    pos_scores = scores[issame]
    neg_scores = scores[~issame]
    thresholds = np.sort(neg_scores)[::-1]

    best_tar = 0.0
    for threshold in thresholds:
        far = np.mean(neg_scores >= threshold)
        tar = np.mean(pos_scores >= threshold)
        if far <= target_far:
            best_tar = tar
            break

    return best_tar


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MobileFaceNet checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = MobileFaceNet(config.EMBEDDING_SIZE).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if "student" in checkpoint:
        model.load_state_dict(checkpoint["student"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    total, _ = count_parameters(model)
    print(f"\nModel parameters: {total / 1e6:.2f}M")

    try:
        from thop import profile

        dummy = torch.randn(1, 3, 112, 112).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"FLOPs: {flops / 1e9:.2f}G")
    except Exception:
        pass

    model_size = os.path.getsize(args.model_path) / (1024 * 1024)
    print(f"Checkpoint size: {model_size:.1f}MB")

    print(f"\n{'=' * 50}")
    print("Validation Results")
    print(f"{'=' * 50}")

    for name in config.VAL_TARGETS:
        bin_path = os.path.join(config.VAL_DATA_DIR, f"{name}.bin")
        if not os.path.exists(bin_path):
            print(f"  {name}: file not found, skipped")
            continue

        images, issame_list = load_bin(bin_path)
        features = extract_features(model, images, device=device)
        acc, std, threshold = compute_accuracy(features, issame_list)
        print(
            f"  {name:12s}: Accuracy = {acc * 100:.2f}% ± {std * 100:.2f}% "
            f"(threshold = {threshold:.3f})"
        )

        tar_1e3 = compute_tar_at_far(features, issame_list, target_far=1e-3)
        tar_1e4 = compute_tar_at_far(features, issame_list, target_far=1e-4)
        print(
            f"  {'':12s}  TAR@FAR=1e-3: {tar_1e3 * 100:.2f}%  "
            f"TAR@FAR=1e-4: {tar_1e4 * 100:.2f}%"
        )

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
