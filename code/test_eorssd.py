#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Official single-view EORSSD evaluation for the open-source SFP-Mamba release.

Default checkpoint:
    checkpoints/sfp_mamba_eorssd.pth

Default local EORSSD test data:
    /home/lch/work/sod/dataset/EORSSD/test-images
    /home/lch/work/sod/dataset/EORSSD/test-labels

Exact paper evaluation protocol:
    1) network input resized to 352 by data.test_dataset
    2) GT remains at original resolution
    3) final logit = outputs[8] / outputs[-2]
    4) resize LOGITS to original GT size, bilinear, align_corners=False
    5) sigmoid
    6) per-image min-max normalization
    7) uint8 [0,255]
    8) py_sod_metrics

No rotation / TTA / multi-view fusion.
The exact uint8 prediction used by the metrics is also saved as PNG.
"""

import argparse
import csv
import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import test_dataset
from model.MyNet_SF import Net

try:
    from py_sod_metrics import MAE, Smeasure, Fmeasure, Emeasure, WeightedFmeasure
except ImportError as exc:
    raise ImportError("Evaluation requires pysodmetrics: pip install pysodmetrics") from exc

warnings.filterwarnings(
    "ignore",
    message=r"This class will be removed in the future, please use FmeasureV2 instead!",
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "sfp_mamba_eorssd.pth"
DEFAULT_IMAGE_ROOT = Path("/home/lch/work/sod/dataset/EORSSD/test-images")
DEFAULT_GT_ROOT = Path("/home/lch/work/sod/dataset/EORSSD/test-labels")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "EORSSD"


def unwrap_checkpoint(obj):
    if not isinstance(obj, dict):
        return obj

    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj

    for key in ("state_dict", "model", "ema", "model_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    return obj


def clean_state_dict(state):
    cleaned = OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_checkpoint_exact(model, checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    raw = torch.load(str(checkpoint), map_location="cpu")
    state = clean_state_dict(unwrap_checkpoint(raw))
    dst = model.state_dict()

    missing = [k for k in dst if k not in state]
    unexpected = [k for k in state if k not in dst]
    mismatched = [
        k
        for k in state
        if k in dst
        and torch.is_tensor(state[k])
        and state[k].shape != dst[k].shape
    ]

    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint/model mismatch:\n"
            "missing={} unexpected={} mismatched={}\n"
            "first_missing={}\nfirst_unexpected={}\nfirst_mismatched={}".format(
                len(missing),
                len(unexpected),
                len(mismatched),
                missing[:10],
                unexpected[:10],
                mismatched[:10],
            )
        )

    model.load_state_dict(state, strict=True)
    print("[Checkpoint] exact-loaded {} tensors from {}".format(len(state), checkpoint))


def configure_full_runtime(model):
    model.set_runtime_modes(
        feedback_enabled=True,
        corrector_enabled=True,
        region_enabled=True,
    )

    model.encoder.set_runtime_scales(
        frequency_scale=1.0,
        stage1_scale=1.0,
        stage2_scale=1.0,
        controller_scale=1.0,
    )

    model.decoder.set_runtime_scales(
        mamba_detail_scale=1.0,
        bgmask_scale=1.0,
        hr176_scale=1.0,
        hr352_scale=1.0,
        corrector_scale=1.0,
        region_scale=1.0,
        region_feature_scale=1.0,
        region_mask_scale=1.0,
        region_post_scale=1.0,
    )

    model.eval()


def normalize_prediction(prob):
    pred = np.asarray(prob, dtype=np.float32)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def to_uint8(pred01):
    return np.round(pred01 * 255.0).astype(np.uint8)


def create_metrics():
    return {
        "mae": MAE(),
        "sm": Smeasure(),
        "fm": Fmeasure(beta=0.3),
        "em": Emeasure(),
        "wfm": WeightedFmeasure(),
    }


def metric_step(meters, pred, gt):
    meters["mae"].step(pred=pred, gt=gt)
    meters["sm"].step(pred=pred, gt=gt)
    meters["fm"].step(pred=pred, gt=gt)
    meters["em"].step(pred=pred, gt=gt)
    meters["wfm"].step(pred=pred, gt=gt)


def collect_metrics(meters):
    fm = meters["fm"].get_results()["fm"]
    em = meters["em"].get_results()["em"]

    return OrderedDict(
        [
            ("MAE", float(meters["mae"].get_results()["mae"])),
            ("S_alpha", float(meters["sm"].get_results()["sm"])),
            ("F_beta_mean", float(np.mean(fm["curve"]))),
            ("F_beta_adp", float(fm["adp"])),
            ("F_beta_max", float(np.max(fm["curve"]))),
            ("E_xi_mean", float(np.mean(em["curve"]))),
            ("E_xi_adp", float(em["adp"])),
            ("E_xi_max", float(np.max(em["curve"]))),
            ("weighted_F", float(meters["wfm"].get_results()["wfm"])),
        ]
    )


@torch.no_grad()
def evaluate(model, image_root, gt_root, testsize, output_dir, save_predictions, print_every):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_dir = output_dir / "predictions_png"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    loader = test_dataset(
        str(image_root) + os.sep,
        str(gt_root) + os.sep,
        int(testsize),
    )

    meters = create_metrics()
    manifest = []

    for index in range(loader.size):
        image, gt_pil, name = loader.load_data()
        image = image.cuda(non_blocking=True)
        gt = np.asarray(gt_pil.convert("L"), dtype=np.uint8)

        # Single-view only.
        outputs = model(image)
        logits = outputs[8]

        # Exact paper protocol: resize LOGITS before sigmoid.
        if tuple(logits.shape[-2:]) != tuple(gt.shape):
            logits = F.interpolate(
                logits,
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            )

        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred_float = normalize_prediction(prob)
        pred_uint8 = to_uint8(pred_float)

        # Same array -> metrics and PNG.
        metric_step(meters, pred_uint8, gt)

        png_path = ""
        if save_predictions:
            png_name = Path(str(name)).stem + ".png"
            png_path_obj = prediction_dir / png_name
            Image.fromarray(pred_uint8, mode="L").save(str(png_path_obj))
            png_path = str(png_path_obj)

        manifest.append(
            {
                "index": index,
                "source_name": str(name),
                "prediction_png": png_path,
                "gt_height": int(gt.shape[0]),
                "gt_width": int(gt.shape[1]),
            }
        )

        if (index + 1) % max(1, int(print_every)) == 0 or (index + 1) == loader.size:
            print("[EORSSD] {}/{}".format(index + 1, loader.size))

    metrics = collect_metrics(meters)

    with (output_dir / "prediction_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "source_name",
                "prediction_png",
                "gt_height",
                "gt_width",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    return metrics, loader.size, prediction_dir


def main():
    parser = argparse.ArgumentParser(
        description="Official single-view EORSSD evaluation for SFP-Mamba."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--image_root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--gt_root", default=str(DEFAULT_GT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--no_save_predictions", action="store_true")
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    checkpoint = Path(args.checkpoint)
    image_root = Path(args.image_root)
    gt_root = Path(args.gt_root)
    output = Path(args.output)

    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if not gt_root.is_dir():
        raise FileNotFoundError(gt_root)

    print("=" * 120)
    print("SFP-MAMBA | OFFICIAL EORSSD SINGLE-VIEW TEST")
    print("=" * 120)
    print("GPU        :", torch.cuda.get_device_name(0))
    print("Checkpoint :", checkpoint)
    print("Images     :", image_root)
    print("GT         :", gt_root)
    print("Output     :", output)
    print("Inference  : single-view, no TTA")
    print("=" * 120)

    model = Net(load_pretrained=False).cuda()
    load_checkpoint_exact(model, checkpoint)
    configure_full_runtime(model)

    metrics, n_samples, prediction_dir = evaluate(
        model=model,
        image_root=image_root,
        gt_root=gt_root,
        testsize=args.testsize,
        output_dir=output,
        save_predictions=not args.no_save_predictions,
        print_every=args.print_every,
    )

    result = OrderedDict(
        [
            ("dataset", "EORSSD"),
            ("inference", "single"),
            ("N", int(n_samples)),
            ("checkpoint", str(checkpoint)),
            *list(metrics.items()),
        ]
    )

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("FINAL EORSSD RESULTS")
    print("=" * 120)
    print(
        "MAE={:.9f} | S={:.9f} | wF={:.9f}".format(
            metrics["MAE"], metrics["S_alpha"], metrics["weighted_F"]
        )
    )
    print(
        "F: max={:.9f} | mean={:.9f} | adp={:.9f}".format(
            metrics["F_beta_max"], metrics["F_beta_mean"], metrics["F_beta_adp"]
        )
    )
    print(
        "E: max={:.9f} | mean={:.9f} | adp={:.9f}".format(
            metrics["E_xi_max"], metrics["E_xi_mean"], metrics["E_xi_adp"]
        )
    )
    print("Metrics CSV :", output / "metrics.csv")
    print("Metrics JSON:", output / "metrics.json")
    if not args.no_save_predictions:
        print("Predictions :", prediction_dir)
    print("=" * 120)


if __name__ == "__main__":
    main()
