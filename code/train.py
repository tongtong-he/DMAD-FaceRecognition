# -*- coding: utf-8 -*-
"""Training script for baseline, distillation baselines, and DMAD."""

import argparse
import os
import sys

import torch
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import build_train_loader
from evaluate import validate_all
from losses import (
    ArcFaceLoss,
    CombinedLoss,
    FeatureDistillationLoss,
    FitNetsLoss,
    KDLoss,
)
from model import MobileFaceNet, count_parameters, load_teacher


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight face recognition models")
    parser.add_argument(
        "--mode",
        type=str,
        default="ours",
        choices=["baseline", "kd", "fitnets", "ours", "ablation_feat", "ablation_ang"],
        help="training mode",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    parser.add_argument("--alpha", type=float, default=None, help="override config.ALPHA")
    parser.add_argument("--beta", type=float, default=None, help="override config.BETA")
    return parser.parse_args()


def train_one_epoch(student, teacher, train_loader, criterion, optimizer, epoch, device, mode, print_freq=100):
    student.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100)

    for batch_idx, (images, labels) in enumerate(progress):
        images = images.to(device)
        labels = labels.to(device)
        feat_student = student(images)

        if mode == "baseline":
            loss = criterion(feat_student, labels)
            loss_dict = {"l_cls": loss.item(), "total": loss.item()}

        elif mode in ["ours", "ablation_feat", "ablation_ang"]:
            with torch.no_grad():
                feat_teacher = teacher(images)
            loss, loss_dict = criterion(feat_student, feat_teacher, labels)

        elif mode == "fitnets":
            with torch.no_grad():
                feat_teacher = teacher(images)
            l_cls = criterion["cls"](feat_student, labels)
            l_feat = criterion["feat"](feat_student, feat_teacher)
            loss = l_cls + l_feat
            loss_dict = {
                "l_cls": l_cls.item(),
                "l_feat": l_feat.item(),
                "total": loss.item(),
            }

        elif mode == "kd":
            with torch.no_grad():
                feat_teacher = teacher(images)
                logits_teacher = criterion["cls"].forward_logits(feat_teacher, labels=labels, margin=False)
            l_cls = criterion["cls"](feat_student, labels)
            logits_student = criterion["cls"].forward_logits(feat_student, labels=labels, margin=False)
            l_kd = criterion["kd"](logits_student, logits_teacher)
            loss = l_cls + config.KD_WEIGHT * l_kd
            loss_dict = {
                "l_cls": l_cls.item(),
                "l_kd": l_kd.item(),
                "total": loss.item(),
            }

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % print_freq == 0:
            progress.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

    return total_loss / total_samples


def build_criterion(mode, alpha, beta, device):
    if mode == "baseline":
        return ArcFaceLoss(
            config.EMBEDDING_SIZE,
            config.NUM_CLASSES,
            config.ARCFACE_S,
            config.ARCFACE_M,
        ).to(device)

    if mode == "ours":
        return CombinedLoss(
            config.EMBEDDING_SIZE,
            config.NUM_CLASSES,
            config.ARCFACE_S,
            config.ARCFACE_M,
            alpha=alpha,
            beta=beta,
        ).to(device)

    if mode == "ablation_feat":
        return CombinedLoss(
            config.EMBEDDING_SIZE,
            config.NUM_CLASSES,
            config.ARCFACE_S,
            config.ARCFACE_M,
            alpha=alpha,
            beta=0.0,
        ).to(device)

    if mode == "ablation_ang":
        return CombinedLoss(
            config.EMBEDDING_SIZE,
            config.NUM_CLASSES,
            config.ARCFACE_S,
            config.ARCFACE_M,
            alpha=0.0,
            beta=beta,
        ).to(device)

    if mode == "fitnets":
        return {
            "cls": ArcFaceLoss(
                config.EMBEDDING_SIZE,
                config.NUM_CLASSES,
                config.ARCFACE_S,
                config.ARCFACE_M,
            ).to(device),
            "feat": FitNetsLoss().to(device),
        }

    if mode == "kd":
        return {
            "cls": ArcFaceLoss(
                config.EMBEDDING_SIZE,
                config.NUM_CLASSES,
                config.ARCFACE_S,
                config.ARCFACE_M,
            ).to(device),
            "kd": KDLoss(temperature=config.KD_TEMPERATURE).to(device),
        }

    raise ValueError(f"Unsupported mode: {mode}")


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print("=" * 60 + "\n")

    alpha = args.alpha if args.alpha is not None else config.ALPHA
    beta = args.beta if args.beta is not None else config.BETA
    print(f"  alpha={alpha}, beta={beta}")

    save_dir = os.path.join(config.OUTPUT_DIR, args.mode)
    os.makedirs(save_dir, exist_ok=True)

    student = MobileFaceNet(config.EMBEDDING_SIZE).to(device)
    total_params, _ = count_parameters(student)
    print(f"  Student parameters: {total_params / 1e6:.2f}M")

    teacher = None
    if args.mode != "baseline":
        if not os.path.exists(config.PRETRAINED_TEACHER):
            print(f"\n[Error] Teacher checkpoint not found: {config.PRETRAINED_TEACHER}")
            print("Please download the InsightFace ResNet-100 ArcFace checkpoint and place it under:")
            print(f"  {config.PRETRAINED_TEACHER}")
            print("Reference:")
            print("  https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch")
            sys.exit(1)
        teacher = load_teacher(config.PRETRAINED_TEACHER, device)

    criterion = build_criterion(args.mode, alpha, beta, device)

    if isinstance(criterion, dict):
        params = list(student.parameters())
        for module in criterion.values():
            params.extend(list(module.parameters()))
    else:
        params = list(student.parameters()) + list(criterion.parameters())

    optimizer = optim.SGD(
        params,
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA,
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        student.load_state_dict(checkpoint["student"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}")

    print("\n  Loading training data...")
    train_loader = build_train_loader(
        config.TRAIN_REC,
        config.TRAIN_IDX,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
    )

    best_acc = 0.0
    print(f"\n  Start training for {config.NUM_EPOCHS} epochs\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        avg_loss = train_one_epoch(
            student,
            teacher,
            train_loader,
            criterion,
            optimizer,
            epoch,
            device,
            args.mode,
            config.PRINT_FREQ,
        )
        scheduler.step()

        print(f"\n  Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
        results = validate_all(student, config.VAL_DATA_DIR, config.VAL_TARGETS, device)
        for name, acc in results.items():
            print(f"    {name}: {acc * 100:.2f}%")

        mean_acc = sum(results.values()) / len(results)

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_path = os.path.join(save_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "results": results,
                },
                best_path,
            )
            print(f"    New best checkpoint saved to: {best_path} (mean accuracy: {mean_acc * 100:.2f}%)")

        if (epoch + 1) % config.SAVE_FREQ == 0:
            epoch_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                epoch_path,
            )

        print()

    print("\n" + "=" * 60)
    print(f"  Training finished. Best mean accuracy: {best_acc * 100:.2f}%")
    print(f"  Checkpoints saved in: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
