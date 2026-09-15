#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Official ORSSD training script for the open-source SFP-Mamba release.

Paper-aligned 100-epoch schedule:
    E001-E010: Base representation
    E011-E033: + Frequency Modeling / semantic-frequency feedback
    E034-E060: + Semantic-Conditioned Propagation and output refinement
    E061-E100: + Region-Level Repair, joint end-to-end refinement

Total = 10 + 23 + 27 + 40 = 100 epochs.

At E060 -> E061:
    teacher := EMA(student at E060), region repair disabled
    student := same EMA state
    teacher is frozen for E061-E100

Initialization:
    - ImageNet-pretrained PVTv2-B2 backbone
    - task-specific modules use model-defined random/zero initialization
    - no task-level ORSSD/R9/R14/R15 checkpoint

ORSSD augmentation:
    identity, rot90, rot180, rot270,
    hflip, hflip+rot90, hflip+rot180, hflip+rot270

Paper training resolution:
    fixed 352 x 352

Effective batch size:
    E001-E060: micro 4 x accumulation 4 = 16
    E061-E100: micro 2 x accumulation 8 = 16

Open-source paths:
    pretrain/pvt_v2_b2.pth
    checkpoints/sfp_mamba_orssd.pth

Dataset defaults intentionally retain the author's local ORSSD paths and can be
changed with command-line arguments.
"""

import argparse
import csv
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import get_loader, test_dataset
from model.MyNet_SF import Net
from losses import (
    R15MainLoss,
    R15RegionTeacherLoss,
    BoundaryAwareStructureLoss,
    SoftFMeasureLoss,
)
from utils.utils import clip_gradient

try:
    from py_sod_metrics import MAE, Smeasure, Fmeasure, Emeasure, WeightedFmeasure
except ImportError as exc:
    raise ImportError("Validation requires pysodmetrics: pip install pysodmetrics") from exc

warnings.filterwarnings(
    "ignore",
    message=r"This class will be removed in the future, please use FmeasureV2 instead!",
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAIN = REPO_ROOT / "pretrain" / "pvt_v2_b2.pth"
DEFAULT_RELEASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "sfp_mamba_orssd.pth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "orssd_100e"

DEFAULT_TRAIN_IMAGE_ROOT = Path("/home/lch/work/sod/dataset/ORSSD/Image-train")
DEFAULT_TRAIN_GT_ROOT = Path("/home/lch/work/sod/dataset/ORSSD/GT-train")
DEFAULT_VAL_IMAGE_ROOT = Path("/home/lch/work/sod/dataset/ORSSD/Image-test")
DEFAULT_VAL_GT_ROOT = Path("/home/lch/work/sod/dataset/ORSSD/GT-test")

# -----------------------------------------------------------------------------
# Paper schedule -- fixed for official reproduction
# -----------------------------------------------------------------------------
PAPER_EPOCHS = 100
PAPER_BASE_END = 10
PAPER_FM_END = 33
PAPER_SCP_END = 60

STAGE_BASE = "base_representation"
STAGE_FM = "semantic_frequency"
STAGE_SCP = "semantic_conditioned_propagation"
STAGE_FULL = "full_region_joint"


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# ORSSD full D4 eight-view augmentation
# -----------------------------------------------------------------------------
AUG8_IDS = (0, 1, 2, 3, 4, 5, 6, 7)
AUG8_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "hflip_rot90",
    "hflip_rot180",
    "hflip_rot270",
)


def apply_d4_tensor(x, aug_id):
    if not torch.is_tensor(x):
        raise TypeError("D4 augmentation expects Tensor, got {}".format(type(x).__name__))

    aug_id = int(aug_id)
    k = aug_id % 4
    use_hflip = aug_id >= 4
    y = x

    if use_hflip:
        y = torch.flip(y, dims=(-1,))
    if k:
        y = torch.rot90(y, k=k, dims=(-2, -1))

    return y.contiguous()


class EightFoldAugmentedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.views = AUG8_IDS
        self.factor = len(self.views)

    def __len__(self):
        return len(self.base_dataset) * self.factor

    def __getitem__(self, index):
        base_index = index // self.factor
        aug_id = self.views[index % self.factor]
        sample = self.base_dataset[base_index]

        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("Training dataset must return at least (image, gt)")

        image = apply_d4_tensor(sample[0], aug_id)
        gt = apply_d4_tensor(sample[1], aug_id)

        if isinstance(sample, tuple):
            return (image, gt, *sample[2:])
        return [image, gt, *sample[2:]]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_eightfold_loader(base_loader, batch_size, seed):
    dataset = EightFoldAugmentedDataset(base_loader.dataset)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    kwargs = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=base_loader.pin_memory,
        drop_last=base_loader.drop_last,
        timeout=base_loader.timeout,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    if base_loader.num_workers > 0:
        kwargs["persistent_workers"] = getattr(base_loader, "persistent_workers", False)
        prefetch_factor = getattr(base_loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def count_params(parameters):
    return sum(p.numel() for p in parameters)


def diagnostic_score(m):
    return (
        m["S_alpha"]
        + m["F_beta_max"]
        + 1.35 * m["F_beta_mean"]
        + 1.35 * m["weighted_F"]
        - m["MAE"]
    )


@torch.no_grad()
def copy_model_state(dst, src):
    dst.load_state_dict(src.state_dict(), strict=True)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(float(decay)).add_(model_value, alpha=1.0 - float(decay))
        else:
            ema_value.copy_(model_value)


# -----------------------------------------------------------------------------
# Progressive runtime configuration
# -----------------------------------------------------------------------------
def training_stage(epoch):
    epoch = int(epoch)
    if epoch <= PAPER_BASE_END:
        return STAGE_BASE
    if epoch <= PAPER_FM_END:
        return STAGE_FM
    if epoch <= PAPER_SCP_END:
        return STAGE_SCP
    return STAGE_FULL


def set_encoder_scales(model, frequency, stage1, stage2, controller):
    model.encoder.set_runtime_scales(
        frequency_scale=float(frequency),
        stage1_scale=float(stage1),
        stage2_scale=float(stage2),
        controller_scale=float(controller),
    )


def set_decoder_scales(model, detail, bgmask, hr176, hr352, corrector, region):
    model.decoder.set_runtime_scales(
        mamba_detail_scale=float(detail),
        bgmask_scale=float(bgmask),
        hr176_scale=float(hr176),
        hr352_scale=float(hr352),
        corrector_scale=float(corrector),
        region_scale=float(region),
        region_feature_scale=float(region),
        region_mask_scale=float(region),
        region_post_scale=float(region),
    )


def configure_runtime_for_stage(model, stage):
    if stage == STAGE_BASE:
        model.set_runtime_modes(
            feedback_enabled=False,
            corrector_enabled=False,
            region_enabled=False,
        )
        set_encoder_scales(model, 0, 0, 0, 0)
        set_decoder_scales(model, 0, 0, 0, 0, 0, 0)

    elif stage == STAGE_FM:
        model.set_runtime_modes(
            feedback_enabled=True,
            corrector_enabled=False,
            region_enabled=False,
        )
        set_encoder_scales(model, 1, 1, 1, 1)
        set_decoder_scales(model, 0, 0, 0, 0, 0, 0)

    elif stage == STAGE_SCP:
        model.set_runtime_modes(
            feedback_enabled=True,
            corrector_enabled=True,
            region_enabled=False,
        )
        set_encoder_scales(model, 1, 1, 1, 1)
        set_decoder_scales(model, 1, 1, 1, 1, 1, 0)

    elif stage == STAGE_FULL:
        model.set_runtime_modes(
            feedback_enabled=True,
            corrector_enabled=True,
            region_enabled=True,
        )
        set_encoder_scales(model, 1, 1, 1, 1)
        set_decoder_scales(model, 1, 1, 1, 1, 1, 1)

    else:
        raise ValueError("Unknown stage: {}".format(stage))


def configure_training_mode(model, stage):
    configure_runtime_for_stage(model, stage)
    model.train()
    if stage != STAGE_FULL:
        model.decoder.region_repair.eval()


def set_teacher_runtime(model):
    # Frozen mature E060 model: FM/SCP/refinement ON, RLR OFF.
    model.set_runtime_modes(
        feedback_enabled=True,
        corrector_enabled=True,
        region_enabled=False,
    )
    set_encoder_scales(model, 1, 1, 1, 1)
    set_decoder_scales(model, 1, 1, 1, 1, 1, 0)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


# -----------------------------------------------------------------------------
# Optimizers
# -----------------------------------------------------------------------------
def build_pre_region_optimizer(model, base_lr, weight_decay):
    # E001-E060: region repair stays frozen; all other parameters may train.
    for p in model.parameters():
        p.requires_grad = False

    encoder_base = []
    encoder_frequency = []
    decoder_main = []

    for name, p in model.named_parameters():
        if name.startswith("decoder.region_repair."):
            continue

        p.requires_grad = True

        if (
            name.startswith("encoder.freq_pyramid.")
            or name.startswith("encoder.stage1_adapter.")
            or name.startswith("encoder.stage2_adapter.")
            or name.startswith("encoder.semantic_controller.")
        ):
            encoder_frequency.append(p)
        elif name.startswith("encoder."):
            encoder_base.append(p)
        elif name.startswith("decoder."):
            decoder_main.append(p)
        else:
            raise RuntimeError("Unclassified parameter: {}".format(name))

    groups = [
        {
            "params": encoder_base,
            "group_name": "encoder_base",
            "lr_mult": 0.20,
            "lr": base_lr * 0.20,
        },
        {
            "params": encoder_frequency,
            "group_name": "encoder_frequency",
            "lr_mult": 1.00,
            "lr": base_lr,
        },
        {
            "params": decoder_main,
            "group_name": "decoder_main",
            "lr_mult": 1.00,
            "lr": base_lr,
        },
    ]

    print("[E001-E060 trainable groups]")
    for g in groups:
        print(
            "  {:<20s} params={:>12d} mult={:.2f}".format(
                g["group_name"], count_params(g["params"]), g["lr_mult"]
            )
        )

    return torch.optim.AdamW(
        groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def build_full_joint_optimizer(model, base_lr, weight_decay):
    # E061-E100: full model jointly refined, with conservative LR on mature paths.
    for p in model.parameters():
        p.requires_grad = True

    meta = OrderedDict(
        [
            ("encoder_base", {"params": [], "lr_mult": 0.10}),
            ("encoder_frequency", {"params": [], "lr_mult": 0.35}),
            ("decoder_main", {"params": [], "lr_mult": 0.35}),
            ("region_main", {"params": [], "lr_mult": 1.00}),
            ("region_mamba", {"params": [], "lr_mult": 0.65}),
        ]
    )

    for name, p in model.named_parameters():
        if (
            name.startswith("decoder.region_repair.global_mamba.")
            or name.startswith("decoder.region_repair.candidate_mamba.")
        ):
            key = "region_mamba"
        elif name.startswith("decoder.region_repair."):
            key = "region_main"
        elif (
            name.startswith("encoder.freq_pyramid.")
            or name.startswith("encoder.stage1_adapter.")
            or name.startswith("encoder.stage2_adapter.")
            or name.startswith("encoder.semantic_controller.")
        ):
            key = "encoder_frequency"
        elif name.startswith("encoder."):
            key = "encoder_base"
        elif name.startswith("decoder."):
            key = "decoder_main"
        else:
            raise RuntimeError("Unclassified joint parameter: {}".format(name))

        meta[key]["params"].append(p)

    groups = []
    print("[E061-E100 joint trainable groups]")

    for group_name, item in meta.items():
        if not item["params"]:
            continue

        group = {
            "params": item["params"],
            "group_name": group_name,
            "lr_mult": float(item["lr_mult"]),
            "lr": base_lr * float(item["lr_mult"]),
        }
        groups.append(group)

        print(
            "  {:<20s} params={:>12d} mult={:.2f}".format(
                group_name, count_params(item["params"]), group["lr_mult"]
            )
        )

    return torch.optim.AdamW(
        groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


# -----------------------------------------------------------------------------
# LR
# -----------------------------------------------------------------------------
def cosine_lr(local_epoch, phase_epochs, base_lr, min_lr_ratio, warmup_epochs):
    local_epoch = int(local_epoch)
    phase_epochs = max(int(phase_epochs), 1)

    progress = float(max(local_epoch - 1, 0)) / max(phase_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    lr = float(base_lr) * (
        float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine
    )

    if warmup_epochs > 0 and local_epoch <= warmup_epochs:
        warmup_ratio = 0.20 + 0.80 * (float(local_epoch) / float(warmup_epochs))
        lr *= warmup_ratio

    return lr


def adjust_lr(optimizer, base_lr):
    parts = []
    for group in optimizer.param_groups:
        group["lr"] = float(base_lr) * float(group["lr_mult"])
        parts.append("{}={:.2e}".format(group["group_name"], group["lr"]))
    print("[LR] base={:.3e} | {}".format(base_lr, " ".join(parts)))


# -----------------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------------
def deep_supervision(outputs, gt, structure_loss, f_loss):
    p5, p4, p3 = outputs[0], outputs[2], outputs[4]

    def one(logits):
        return structure_loss(logits, gt) + 0.35 * f_loss(logits, gt)

    return 0.10 * one(p3) + 0.07 * one(p4) + 0.05 * one(p5)


def region_teacher_weight(epoch):
    # E061-E100: linearly 0.36 -> 0.16.
    local = int(epoch) - PAPER_SCP_END
    phase_len = PAPER_EPOCHS - PAPER_SCP_END
    frac = float(local - 1) / max(phase_len - 1, 1)
    return 0.36 + frac * (0.16 - 0.36)


# -----------------------------------------------------------------------------
# One epoch
# -----------------------------------------------------------------------------
def train_one_epoch(
    loader,
    model,
    ema_model,
    teacher,
    optimizer,
    criterion_main,
    criterion_region,
    deep_structure,
    deep_f,
    args,
    epoch,
):
    stage = training_stage(epoch)
    configure_training_mode(model, stage)

    if teacher is not None:
        set_teacher_runtime(teacher)

    total_steps = len(loader)
    optimizer.zero_grad(set_to_none=True)

    teacher_weight = region_teacher_weight(epoch) if stage == STAGE_FULL else 0.0
    running = {}
    start = time.time()

    for step, pack in enumerate(loader, start=1):
        images = pack[0].cuda(non_blocking=True)
        gts = pack[1].cuda(non_blocking=True).float()

        # Paper protocol: fixed 352 x 352 throughout training.
        if images.shape[-2:] != (args.trainsize, args.trainsize):
            images = F.interpolate(
                images,
                size=(args.trainsize, args.trainsize),
                mode="bilinear",
                align_corners=False,
            )
            gts = F.interpolate(
                gts,
                size=(args.trainsize, args.trainsize),
                mode="nearest",
            )

        teacher_outputs = None
        if stage == STAGE_FULL:
            if teacher is None:
                raise RuntimeError("Full stage requires frozen E060 teacher.")
            with torch.no_grad():
                teacher_outputs = teacher(images)

        outputs = model(images)
        p2 = outputs[6]
        p1 = outputs[8]
        aux = model.training_aux_tensors()

        main_dict = criterion_main(
            final_logits=p1,
            p176_logits=aux["p176_logits"],
            p2_logits=p2,
            gt=gts,
        )

        if stage != STAGE_FULL:
            deep_loss = deep_supervision(outputs, gts, deep_structure, deep_f)
            region_dict = None
            loss = main_dict["total"] + deep_loss
        else:
            teacher_p2 = teacher_outputs[6].detach()
            teacher_p1 = teacher_outputs[8].detach()

            region_dict = criterion_region(
                final_logits=p1,
                teacher_final_logits=teacher_p1,
                teacher_p2_logits=teacher_p2,
                region_route_logits44=aux["r15_region_route_logits44"],
                region_aux_logits44=aux["r15_region_aux_logits44"],
                p2_base_logits=aux["r15_region_p2_base"],
                p2_repaired_logits=aux["r15_region_p2_repaired"],
                applied_d2_88=aux["r15_region_applied_d2_88"],
                applied_p2_88=aux["r15_region_applied_p2_88"],
                applied_d1_88=aux["r15_region_applied_d1_88"],
                gt=gts,
            )

            deep_loss = 0.15 * deep_supervision(
                outputs, gts, deep_structure, deep_f
            )

            loss = (
                main_dict["total"]
                + teacher_weight * region_dict["total"]
                + deep_loss
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss at epoch {} step {}".format(epoch, step)
            )

        (loss / args.accum_steps).backward()

        do_update = step % args.accum_steps == 0 or step == total_steps
        if do_update:
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema_model, model, args.ema_decay)

        values = {
            "total": loss,
            "main": main_dict["total"],
            "deep": deep_loss,
        }
        if region_dict is not None:
            values["region"] = region_dict["total"]

        for key, value in values.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu().item())

        if step % args.print_every == 0 or step == total_steps:
            print(
                "[E{:03d}/{:03d}] {:<34s} step={:04d}/{:04d} "
                "loss={:.4f} main={:.4f} deep={:.4f} teacher_w={:.3f} elapsed={:.0f}s".format(
                    epoch,
                    PAPER_EPOCHS,
                    stage,
                    step,
                    total_steps,
                    running["total"] / step,
                    running["main"] / step,
                    running["deep"] / step,
                    teacher_weight,
                    time.time() - start,
                )
            )

    means = {key: value / max(total_steps, 1) for key, value in running.items()}
    means["teacher_weight"] = teacher_weight
    return stage, means


# -----------------------------------------------------------------------------
# Optional development validation; same protocol as test_orssd.py
# -----------------------------------------------------------------------------
def pred_to_uint8(prob):
    pred = np.asarray(prob, dtype=np.float32)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = np.clip(pred, 0.0, 1.0)
    return np.round(pred * 255.0).astype(np.uint8)


@torch.no_grad()
def validate_sod(model, image_root, gt_root, testsize, stage):
    configure_runtime_for_stage(model, stage)
    model.eval()

    loader = test_dataset(
        str(image_root) + os.sep,
        str(gt_root) + os.sep,
        int(testsize),
    )

    mae = MAE()
    sm = Smeasure()
    fm = Fmeasure(beta=0.3)
    em = Emeasure()
    wfm = WeightedFmeasure()

    for _ in range(loader.size):
        image, gt_pil, _ = loader.load_data()
        image = image.cuda(non_blocking=True)
        gt = np.asarray(gt_pil.convert("L"), dtype=np.uint8)

        logits = model(image)[8]
        if tuple(logits.shape[-2:]) != tuple(gt.shape):
            logits = F.interpolate(
                logits,
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            )

        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = pred_to_uint8(prob)

        mae.step(pred=pred, gt=gt)
        sm.step(pred=pred, gt=gt)
        fm.step(pred=pred, gt=gt)
        em.step(pred=pred, gt=gt)
        wfm.step(pred=pred, gt=gt)

    fm_result = fm.get_results()["fm"]
    em_result = em.get_results()["em"]

    return {
        "MAE": float(mae.get_results()["mae"]),
        "S_alpha": float(sm.get_results()["sm"]),
        "weighted_F": float(wfm.get_results()["wfm"]),
        "F_beta_max": float(np.max(fm_result["curve"])),
        "F_beta_mean": float(np.mean(fm_result["curve"])),
        "F_beta_adp": float(fm_result["adp"]),
        "E_xi_max": float(np.max(em_result["curve"])),
        "E_xi_mean": float(np.mean(em_result["curve"])),
        "E_xi_adp": float(em_result["adp"]),
    }


def print_validation(epoch, stage, m):
    print("=" * 120)
    print("VALIDATION E{:03d} | {}".format(epoch, stage))
    print("=" * 120)
    print("MAE={:.9f} | S={:.9f} | wF={:.9f}".format(m["MAE"], m["S_alpha"], m["weighted_F"]))
    print("F: max={:.9f} | mean={:.9f} | adp={:.9f}".format(m["F_beta_max"], m["F_beta_mean"], m["F_beta_adp"]))
    print("E: max={:.9f} | mean={:.9f} | adp={:.9f}".format(m["E_xi_max"], m["E_xi_mean"], m["E_xi_adp"]))
    print("diagnostic_score={:.9f}".format(diagnostic_score(m)))
    print("=" * 120)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def append_csv(path, row):
    path = Path(path)
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_state_dict(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def save_resume_state(path, epoch, model, ema_model, optimizer, teacher, teacher_ready):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "teacher_ready": bool(teacher_ready),
            "teacher": teacher.state_dict() if teacher is not None else None,
        },
        str(path),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Official 100-epoch ORSSD training for SFP-Mamba."
    )

    parser.add_argument("--train_image_root", default=str(DEFAULT_TRAIN_IMAGE_ROOT))
    parser.add_argument("--train_gt_root", default=str(DEFAULT_TRAIN_GT_ROOT))
    parser.add_argument("--val_image_root", default=str(DEFAULT_VAL_IMAGE_ROOT))
    parser.add_argument("--val_gt_root", default=str(DEFAULT_VAL_GT_ROOT))
    parser.add_argument("--pretrain", default=str(DEFAULT_PRETRAIN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--release_checkpoint", default=str(DEFAULT_RELEASE_CHECKPOINT))

    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--pre_region_batchsize", type=int, default=4)
    parser.add_argument("--pre_region_accum_steps", type=int, default=4)
    parser.add_argument("--full_batchsize", type=int, default=2)
    parser.add_argument("--full_accum_steps", type=int, default=8)

    parser.add_argument("--pre_region_lr", type=float, default=1.0e-4)
    parser.add_argument("--full_lr", type=float, default=4.0e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument(
        "--val_interval",
        type=int,
        default=0,
        help="0 by default; nonzero is development-only validation.",
    )
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--resume", default="")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.trainsize != 352:
        raise ValueError("Paper reproduction uses fixed 352x352 training.")
    if args.pre_region_batchsize * args.pre_region_accum_steps != 16:
        raise ValueError("E001-E060 effective batch must be 16.")
    if args.full_batchsize * args.full_accum_steps != 16:
        raise ValueError("E061-E100 effective batch must be 16.")

    set_seed(args.seed)

    train_image_root = Path(args.train_image_root)
    train_gt_root = Path(args.train_gt_root)
    val_image_root = Path(args.val_image_root)
    val_gt_root = Path(args.val_gt_root)
    pretrain = Path(args.pretrain)
    output = Path(args.output)
    release_checkpoint = Path(args.release_checkpoint)

    if not train_image_root.is_dir():
        raise FileNotFoundError(train_image_root)
    if not train_gt_root.is_dir():
        raise FileNotFoundError(train_gt_root)
    if not pretrain.is_file() and not args.resume:
        raise FileNotFoundError("Missing PVT pretrained checkpoint: {}".format(pretrain))

    weights_dir = output / "weights"
    raw_dir = output / "weights_raw"
    state_dir = output / "state"
    tb_dir = output / "tb"

    for directory in (
        output,
        weights_dir,
        raw_dir,
        state_dir,
        tb_dir,
        release_checkpoint.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("SFP-MAMBA | OFFICIAL ORSSD 100-EPOCH TRAINING")
    print("=" * 120)
    print("GPU              :", torch.cuda.get_device_name(0))
    print("Train images     :", train_image_root)
    print("Train GT         :", train_gt_root)
    print("PVT pretrain     :", pretrain)
    print("Release ckpt     :", release_checkpoint)
    print("Resolution       : 352 x 352 fixed")
    print("Augmentation     : 8x {}".format(AUG8_NAMES))
    print("Schedule         : E1-10 Base | E11-33 +FM | E34-60 +SCP | E61-100 Full")
    print("Teacher          : frozen region-free EMA from E060")
    print("=" * 120)

    # Data
    base_loader = get_loader(
        str(train_image_root) + os.sep,
        str(train_gt_root) + os.sep,
        args.pre_region_batchsize,
        args.trainsize,
        shuffle=True,
        pin_memory=True,
        is_train=True,
    )

    pre_region_loader = build_eightfold_loader(
        base_loader,
        args.pre_region_batchsize,
        args.seed,
    )
    full_loader = build_eightfold_loader(
        base_loader,
        args.full_batchsize,
        args.seed + 100003,
    )

    print("Base train samples     :", len(base_loader.dataset))
    print("Effective 8x samples   :", len(pre_region_loader.dataset))
    print("E001-E060 batches      :", len(pre_region_loader))
    print("E061-E100 batches      :", len(full_loader))

    # Models
    if args.resume:
        model = Net(load_pretrained=False).cuda()
        ema_model = Net(load_pretrained=False).cuda()
    else:
        model = Net(
            load_pretrained=True,
            pretrain_path=str(pretrain),
        ).cuda()
        ema_model = Net(load_pretrained=False).cuda()
        copy_model_state(ema_model, model)

    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    teacher = None
    teacher_ready = False

    # Losses
    criterion_main = R15MainLoss().cuda()
    criterion_region = R15RegionTeacherLoss().cuda()
    deep_structure = BoundaryAwareStructureLoss().cuda()
    deep_f = SoftFMeasureLoss(beta2=0.3).cuda()

    # Optimizer / resume
    start_epoch = 1
    optimizer = build_pre_region_optimizer(
        model,
        args.pre_region_lr,
        args.weight_decay,
    )

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)

        state = torch.load(str(resume_path), map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        ema_model.load_state_dict(state["ema"], strict=True)

        completed_epoch = int(state["epoch"])
        start_epoch = completed_epoch + 1
        teacher_ready = bool(state.get("teacher_ready", False))

        if teacher_ready:
            teacher = Net(load_pretrained=False).cuda()
            teacher.load_state_dict(state["teacher"], strict=True)
            set_teacher_runtime(teacher)

        if start_epoch > PAPER_SCP_END:
            optimizer = build_full_joint_optimizer(
                model,
                args.full_lr,
                args.weight_decay,
            )
        else:
            optimizer = build_pre_region_optimizer(
                model,
                args.pre_region_lr,
                args.weight_decay,
            )

        try:
            optimizer.load_state_dict(state["optimizer"])
            print("[Resume] optimizer restored.")
        except Exception as exc:
            print("[Resume] optimizer not restored after phase/group change:", exc)

        print("[Resume] continuing from E{:03d}".format(start_epoch))

    # Logging
    writer = SummaryWriter(str(tb_dir))
    (output / "run_config.txt").write_text(
        "\n".join(
            [
                "architecture=SFP-Mamba_R15",
                "dataset=ORSSD",
                "paper_epochs=100",
                "schedule=10+23+27+40",
                "stage1=E001-E010_base",
                "stage2=E011-E033_frequency",
                "stage3=E034-E060_scp_refinement",
                "stage4=E061-E100_full_region_joint",
                "teacher=frozen_region_free_EMA_E060",
                "resolution=352",
                "augmentation_views={}".format(",".join(AUG8_NAMES)),
                "effective_batch=16",
                "optimizer=AdamW",
                "weight_decay={}".format(args.weight_decay),
                "pre_region_lr={}".format(args.pre_region_lr),
                "full_lr={}".format(args.full_lr),
                "ema_decay={}".format(args.ema_decay),
                "seed={}".format(args.seed),
                "train_images={}".format(train_image_root),
                "train_gt={}".format(train_gt_root),
                "release_checkpoint={}".format(release_checkpoint),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    train_csv = output / "train_metrics.csv"
    val_csv = output / "val_metrics.csv"

    # Training
    for epoch in range(start_epoch, PAPER_EPOCHS + 1):
        stage = training_stage(epoch)

        if stage == STAGE_FULL:
            train_loader = full_loader
            args.accum_steps = int(args.full_accum_steps)
            micro_batch = int(args.full_batchsize)
        else:
            train_loader = pre_region_loader
            args.accum_steps = int(args.pre_region_accum_steps)
            micro_batch = int(args.pre_region_batchsize)

        # E060 -> E061: region-free EMA becomes student + frozen teacher.
        if epoch == PAPER_SCP_END + 1 and not teacher_ready:
            print("=" * 120)
            print("E060 -> E061: CREATE FROZEN REGION-FREE TEACHER")
            print("=" * 120)

            model.load_state_dict(ema_model.state_dict(), strict=True)

            teacher = Net(load_pretrained=False).cuda()
            teacher.load_state_dict(ema_model.state_dict(), strict=True)
            set_teacher_runtime(teacher)
            teacher_ready = True

            save_state_dict(
                teacher,
                weights_dir / "teacher_region_free_E060.pth",
            )

            ema_model.load_state_dict(model.state_dict(), strict=True)

            optimizer = build_full_joint_optimizer(
                model,
                args.full_lr,
                args.weight_decay,
            )

        # LR schedule
        if stage != STAGE_FULL:
            current_lr = cosine_lr(
                local_epoch=epoch,
                phase_epochs=PAPER_SCP_END,
                base_lr=args.pre_region_lr,
                min_lr_ratio=args.min_lr_ratio,
                warmup_epochs=3,
            )
        else:
            current_lr = cosine_lr(
                local_epoch=epoch - PAPER_SCP_END,
                phase_epochs=PAPER_EPOCHS - PAPER_SCP_END,
                base_lr=args.full_lr,
                min_lr_ratio=args.min_lr_ratio,
                warmup_epochs=2,
            )

        adjust_lr(optimizer, current_lr)

        print()
        print(
            "[Epoch {:03d}/{:03d}] stage={} micro={} accum={} effective=16".format(
                epoch,
                PAPER_EPOCHS,
                stage,
                micro_batch,
                args.accum_steps,
            )
        )

        epoch_start = time.time()

        actual_stage, train_metrics = train_one_epoch(
            loader=train_loader,
            model=model,
            ema_model=ema_model,
            teacher=teacher,
            optimizer=optimizer,
            criterion_main=criterion_main,
            criterion_region=criterion_region,
            deep_structure=deep_structure,
            deep_f=deep_f,
            args=args,
            epoch=epoch,
        )

        epoch_seconds = time.time() - epoch_start

        train_row = {
            "epoch": epoch,
            "stage": actual_stage,
            "loss": train_metrics.get("total", np.nan),
            "main_loss": train_metrics.get("main", np.nan),
            "deep_loss": train_metrics.get("deep", np.nan),
            "region_loss": train_metrics.get("region", np.nan),
            "teacher_weight": train_metrics.get("teacher_weight", 0.0),
            "base_lr": current_lr,
            "seconds": epoch_seconds,
        }
        append_csv(train_csv, train_row)

        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar("train/{}".format(key), value, epoch)
        writer.add_scalar("train/base_lr", current_lr, epoch)

        boundary_epoch = epoch in (
            PAPER_BASE_END,
            PAPER_FM_END,
            PAPER_SCP_END,
            PAPER_EPOCHS,
        )

        if boundary_epoch or (
            args.save_interval > 0 and epoch % args.save_interval == 0
        ):
            save_state_dict(
                ema_model,
                weights_dir / "EMA_E{:03d}.pth".format(epoch),
            )
            save_state_dict(
                model,
                raw_dir / "RAW_E{:03d}.pth".format(epoch),
            )

        # Development-only validation; not used for official checkpoint selection.
        if args.val_interval > 0 and (
            epoch % args.val_interval == 0 or boundary_epoch
        ):
            metrics = validate_sod(
                ema_model,
                val_image_root,
                val_gt_root,
                args.trainsize,
                actual_stage,
            )
            print_validation(epoch, actual_stage, metrics)
            append_csv(
                val_csv,
                {
                    "epoch": epoch,
                    "stage": actual_stage,
                    **metrics,
                    "diagnostic_score": diagnostic_score(metrics),
                },
            )

        save_resume_state(
            path=state_dir / "training_state_last.pt",
            epoch=epoch,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            teacher=teacher,
            teacher_ready=teacher_ready,
        )

        print(
            "Epoch {:03d} finished in {:.1f} min".format(
                epoch, epoch_seconds / 60.0
            )
        )

    # Official release checkpoint = E100 EMA.
    configure_runtime_for_stage(ema_model, STAGE_FULL)
    save_state_dict(ema_model, weights_dir / "last_ema_E100.pth")
    save_state_dict(model, raw_dir / "last_raw_E100.pth")
    save_state_dict(ema_model, release_checkpoint)

    (output / "training_summary.txt").write_text(
        "Training complete\n"
        "paper_epochs=100\n"
        "schedule=10+23+27+40\n"
        "teacher_source=EMA_E060_region_off\n"
        "release_checkpoint={}\n".format(release_checkpoint),
        encoding="utf-8",
    )

    writer.close()

    print("=" * 120)
    print("TRAINING COMPLETE")
    print("Output             :", output)
    print("Frozen teacher     :", weights_dir / "teacher_region_free_E060.pth")
    print("Final E100 EMA     :", weights_dir / "last_ema_E100.pth")
    print("Release checkpoint :", release_checkpoint)
    print("=" * 120)


if __name__ == "__main__":
    main()
