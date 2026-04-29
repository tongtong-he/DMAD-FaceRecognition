# DMAD for Lightweight Face Recognition

Reference implementation of decoupled magnitude-angle distillation for lightweight face recognition and offline edge deployment.

## Overview

This repository contains:

- `baseline`: MobileFaceNet trained with ArcFace only
- `kd`: Hinton-style logit distillation baseline
- `fitnets`: feature regression baseline
- `ours`: decoupled magnitude-angle distillation
- `ablation_feat`: `L_cls + L_feat`
- `ablation_ang`: `L_cls + L_ang`

## Repository Layout

```text
project_root/
|- code/
|  |- config.py
|  |- dataset.py
|  |- evaluate.py
|  |- losses.py
|  |- model.py
|  |- requirements.txt
|  `- train.py
|- data/
|  `- faces_emore/
|     |- train.rec
|     |- train.idx
|     |- property
|     |- lfw.bin
|     |- cfp_fp.bin
|     `- agedb_30.bin
|- pretrained/
|  `- r100_arcface_ms1mv2.pth
`- output/
```

## Installation

```bash
pip install -r requirements.txt
```

## Teacher Checkpoint

Download the InsightFace ResNet-100 ArcFace checkpoint from the official repository:

- https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

Place the checkpoint at:

```text
pretrained/r100_arcface_ms1mv2.pth
```

## Training

From the `code/` directory:

```bash
python train.py --mode baseline
python train.py --mode kd
python train.py --mode fitnets
python train.py --mode ours
python train.py --mode ablation_feat
python train.py --mode ablation_ang
```

## Evaluation

```bash
python evaluate.py --model_path ../output/ours/best.pth
```

## Notes

- Paths in `config.py` are relative so the project can be moved across machines.
- The KD baseline is implemented as temperature-scaled KL divergence on ArcFace classification logits.
- Hyperparameters in `config.py` are set to match the experimental setup used in the paper.
- Deployment and quantization scripts are not included in this minimal research release.

## Citation

Citation information will be updated after publication.
