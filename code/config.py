# -*- coding: utf-8 -*-
"""Project configuration for training and evaluation."""

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "faces_emore"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
PRETRAINED_TEACHER = PRETRAINED_DIR / "r100_arcface_ms1mv2.pth"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Training data
TRAIN_REC = DATA_ROOT / "train.rec"
TRAIN_IDX = DATA_ROOT / "train.idx"

# Validation sets
VAL_TARGETS = ["lfw", "cfp_fp", "agedb_30"]
VAL_DATA_DIR = DATA_ROOT

# Model configuration
EMBEDDING_SIZE = 512
NUM_CLASSES = 85742

# Training hyperparameters
# These values are chosen to match the setup reported in the manuscript.
BATCH_SIZE = 512
NUM_WORKERS = 4
NUM_EPOCHS = 25
INPUT_SIZE = (112, 112)

LR = 0.1
LR_MILESTONES = [10, 18, 24]
LR_GAMMA = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# ArcFace
ARCFACE_S = 64.0
ARCFACE_M = 0.5

# Distillation
# ALPHA and BETA correspond to the loss weights lambda_1 and lambda_2 in the
# manuscript. The default values below match the optimal configuration found
# by grid search and reported in the ablation study.
ALPHA = 0.5
BETA = 1.0
KD_TEMPERATURE = 4.0
KD_WEIGHT = 1.0

# Logging / checkpointing
SAVE_FREQ = 2
PRINT_FREQ = 100
