# -*- coding: utf-8 -*-
"""Loss functions used by the lightweight face recognition experiments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace classification loss with an internal learnable classifier."""

    def __init__(self, embedding_size=512, num_classes=85742, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.criterion = nn.CrossEntropyLoss()

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward_logits(self, embeddings, labels=None, margin=True):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if not margin:
            return cosine * self.s

        if labels is None:
            raise ValueError("labels are required when margin=True")

        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

    def forward(self, embeddings, labels):
        logits = self.forward_logits(embeddings, labels=labels, margin=True)
        return self.criterion(logits, labels)


class FeatureDistillationLoss(nn.Module):
    """Feature-space MSE on unnormalized embeddings."""

    def forward(self, feat_student, feat_teacher):
        # The paper writes this term as an L2 alignment; here we implement the
        # standard mean-squared formulation used in optimization code.
        return F.mse_loss(feat_student, feat_teacher.detach())


class AngleAwareDistillationLoss(nn.Module):
    """Cosine-based alignment on L2-normalized embeddings."""

    def forward(self, feat_student, feat_teacher):
        feat_student = F.normalize(feat_student, p=2, dim=1)
        feat_teacher = F.normalize(feat_teacher.detach(), p=2, dim=1)
        cosine_sim = (feat_student * feat_teacher).sum(dim=1)
        return (1.0 - cosine_sim).mean()


class KDLoss(nn.Module):
    """Temperature-scaled KL divergence for Hinton-style logit distillation."""

    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits_student, logits_teacher):
        t = self.temperature
        p_student = F.log_softmax(logits_student / t, dim=1)
        p_teacher = F.softmax(logits_teacher / t, dim=1)
        return F.kl_div(p_student, p_teacher, reduction="batchmean") * (t ** 2)


class FitNetsLoss(nn.Module):
    """Direct feature regression baseline."""

    def forward(self, feat_student, feat_teacher):
        return F.mse_loss(feat_student, feat_teacher.detach())


class CombinedLoss(nn.Module):
    """Main method: L_total = L_cls + alpha * L_feat + beta * L_ang."""

    def __init__(self, embedding_size=512, num_classes=85742,
                 s=64.0, m=0.5, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cls_loss = ArcFaceLoss(embedding_size, num_classes, s, m)
        self.feat_loss = FeatureDistillationLoss()
        self.ang_loss = AngleAwareDistillationLoss()

    def forward(self, feat_student, feat_teacher, labels):
        l_cls = self.cls_loss(feat_student, labels)
        l_feat = self.feat_loss(feat_student, feat_teacher)
        l_ang = self.ang_loss(feat_student, feat_teacher)
        total = l_cls + self.alpha * l_feat + self.beta * l_ang
        return total, {
            "l_cls": l_cls.item(),
            "l_feat": l_feat.item(),
            "l_ang": l_ang.item(),
            "total": total.item(),
        }
