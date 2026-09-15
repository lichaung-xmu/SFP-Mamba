# -*- coding: utf-8 -*-
"""
Reset-R15 losses.

R15 moves the new capacity from sparse final-pixel repair to region-level
representation repair. The loss therefore has two responsibilities:

1) final metric alignment:
   - structure / IoU;
   - soft F_beta;
   - evaluation-aligned min-max threshold-curve F;
   - multi-scale structural similarity;
   - object foreground/background consistency.

2) region repair supervision:
   - broaden fixed-R9 teacher errors into semantic neighborhoods;
   - train a KEEP / FG-REPAIR / BG-REPAIR region router;
   - directly supervise the auxiliary region saliency and repaired P2;
   - preserve easy regions without forcing the new representation to zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Generic losses
# ============================================================================

class BoundaryAwareStructureLoss(nn.Module):
    def __init__(
        self,
        pool_kernel=31,
        boundary_factor=5.0,
        eps=1e-8,
    ):
        super().__init__()

        self.pool_kernel = int(
            pool_kernel
        )
        self.boundary_factor = float(
            boundary_factor
        )
        self.eps = float(
            eps
        )

    def forward(
        self,
        logits,
        gt,
    ):
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        gt = gt.float()
        pad = self.pool_kernel // 2

        weight = (
            1.0
            + self.boundary_factor
            * torch.abs(
                F.avg_pool2d(
                    gt,
                    kernel_size=self.pool_kernel,
                    stride=1,
                    padding=pad,
                )
                - gt
            )
        )

        wbce_map = F.binary_cross_entropy_with_logits(
            logits,
            gt,
            reduction="none",
        )

        wbce = (
            (
                weight
                * wbce_map
            ).sum(
                dim=(2, 3)
            )
            / (
                weight.sum(
                    dim=(2, 3)
                )
                + self.eps
            )
        ).mean()

        pred = torch.sigmoid(
            logits
        )

        inter = (
            weight
            * pred
            * gt
        ).sum(
            dim=(2, 3)
        )

        union = (
            weight
            * (
                pred
                + gt
                - pred
                * gt
            )
        ).sum(
            dim=(2, 3)
        )

        wiou = (
            1.0
            - (
                inter + 1.0
            )
            / (
                union + 1.0
            )
        ).mean()

        return wbce + wiou


class SoftFMeasureLoss(nn.Module):
    def __init__(
        self,
        beta2=0.3,
        eps=1e-8,
    ):
        super().__init__()

        self.beta2 = float(
            beta2
        )
        self.eps = float(
            eps
        )

    def forward(
        self,
        logits,
        gt,
    ):
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pred = torch.sigmoid(
            logits
        )
        gt = gt.float()

        dims = (
            1,
            2,
            3,
        )

        tp = (
            pred
            * gt
        ).sum(
            dim=dims
        )

        fp = (
            pred
            * (
                1.0 - gt
            )
        ).sum(
            dim=dims
        )

        fn = (
            (
                1.0 - pred
            )
            * gt
        ).sum(
            dim=dims
        )

        precision = (
            tp + self.eps
        ) / (
            tp + fp + self.eps
        )

        recall = (
            tp + self.eps
        ) / (
            tp + fn + self.eps
        )

        f_beta = (
            (
                1.0 + self.beta2
            )
            * precision
            * recall
        ) / (
            self.beta2
            * precision
            + recall
            + self.eps
        )

        return 1.0 - f_beta.mean()


class EvaluationAlignedCurveFLoss(nn.Module):
    """
    Approximate the actual evaluation protocol more closely.

    Validation saves per-image min-max normalized prediction maps before
    computing the threshold F curve. R15 performs the same min-max transform
    in training (extrema detached for stable gradients), then evaluates soft
    thresholds from 0.10 to 0.90.

    The computation is done at at most 176x176 to control memory.
    """

    def __init__(
        self,
        thresholds=None,
        tau=0.045,
        beta2=0.3,
        max_size=176,
        eps=1e-8,
    ):
        super().__init__()

        if thresholds is None:
            thresholds = (
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            )

        self.thresholds = tuple(
            float(x)
            for x in thresholds
        )

        self.tau = float(
            tau
        )
        self.beta2 = float(
            beta2
        )
        self.max_size = int(
            max_size
        )
        self.eps = float(
            eps
        )

    def forward(
        self,
        logits,
        gt,
    ):
        pred = torch.sigmoid(
            logits
        )
        gt = gt.float()

        h, w = pred.shape[-2:]

        if max(
            h,
            w,
        ) > self.max_size:
            pred = F.interpolate(
                pred,
                size=(
                    self.max_size,
                    self.max_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

            gt = F.interpolate(
                gt,
                size=(
                    self.max_size,
                    self.max_size,
                ),
                mode="nearest",
            )

        p_min = pred.amin(
            dim=(2, 3),
            keepdim=True,
        ).detach()

        p_max = pred.amax(
            dim=(2, 3),
            keepdim=True,
        ).detach()

        norm = (
            pred - p_min
        ) / (
            p_max
            - p_min
            + self.eps
        )

        dims = (
            1,
            2,
            3,
        )

        score_sum = 0.0

        for threshold in self.thresholds:
            soft_mask = torch.sigmoid(
                (
                    norm - threshold
                )
                / self.tau
            )

            tp = (
                soft_mask
                * gt
            ).sum(
                dim=dims
            )

            fp = (
                soft_mask
                * (
                    1.0 - gt
                )
            ).sum(
                dim=dims
            )

            fn = (
                (
                    1.0 - soft_mask
                )
                * gt
            ).sum(
                dim=dims
            )

            precision = (
                tp + self.eps
            ) / (
                tp + fp + self.eps
            )

            recall = (
                tp + self.eps
            ) / (
                tp + fn + self.eps
            )

            f_beta = (
                (
                    1.0 + self.beta2
                )
                * precision
                * recall
            ) / (
                self.beta2
                * precision
                + recall
                + self.eps
            )

            score_sum = (
                score_sum
                + f_beta.mean()
            )

        mean_score = (
            score_sum
            / float(
                len(
                    self.thresholds
                )
            )
        )

        return 1.0 - mean_score


class MultiScaleSSIMStructureLoss(nn.Module):
    """
    Local structural similarity surrogate used to push S-measure-relevant
    region consistency instead of only per-pixel confidence.
    """

    def __init__(
        self,
        windows=(7, 15, 31),
        max_size=176,
        c1=0.01 ** 2,
        c2=0.03 ** 2,
    ):
        super().__init__()

        self.windows = tuple(
            int(x)
            for x in windows
        )
        self.max_size = int(
            max_size
        )
        self.c1 = float(
            c1
        )
        self.c2 = float(
            c2
        )

    def _ssim(
        self,
        x,
        y,
        window,
    ):
        pad = window // 2

        mu_x = F.avg_pool2d(
            x,
            kernel_size=window,
            stride=1,
            padding=pad,
        )

        mu_y = F.avg_pool2d(
            y,
            kernel_size=window,
            stride=1,
            padding=pad,
        )

        sigma_x = (
            F.avg_pool2d(
                x * x,
                kernel_size=window,
                stride=1,
                padding=pad,
            )
            - mu_x * mu_x
        )

        sigma_y = (
            F.avg_pool2d(
                y * y,
                kernel_size=window,
                stride=1,
                padding=pad,
            )
            - mu_y * mu_y
        )

        sigma_xy = (
            F.avg_pool2d(
                x * y,
                kernel_size=window,
                stride=1,
                padding=pad,
            )
            - mu_x * mu_y
        )

        numerator = (
            (
                2.0
                * mu_x
                * mu_y
                + self.c1
            )
            * (
                2.0
                * sigma_xy
                + self.c2
            )
        )

        denominator = (
            (
                mu_x * mu_x
                + mu_y * mu_y
                + self.c1
            )
            * (
                sigma_x
                + sigma_y
                + self.c2
            )
        )

        score = (
            numerator
            / (
                denominator
                + 1e-8
            )
        )

        return torch.clamp(
            score,
            -1.0,
            1.0,
        ).mean()

    def forward(
        self,
        logits,
        gt,
    ):
        pred = torch.sigmoid(
            logits
        )
        gt = gt.float()

        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(
                gt,
                size=pred.shape[-2:],
                mode="nearest",
            )

        if max(
            pred.shape[-2:]
        ) > self.max_size:
            pred = F.interpolate(
                pred,
                size=(
                    self.max_size,
                    self.max_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

            gt = F.interpolate(
                gt,
                size=(
                    self.max_size,
                    self.max_size,
                ),
                mode="nearest",
            )

        total = 0.0

        for window in self.windows:
            total = (
                total
                + (
                    1.0
                    - self._ssim(
                        pred,
                        gt,
                        window,
                    )
                )
            )

        return (
            total
            / float(
                len(
                    self.windows
                )
            )
        )


class ObjectConsistencyLoss(nn.Module):
    """
    Object-level foreground/background consistency surrogate.

    It encourages high, compact foreground responses and low, compact
    background responses without forcing arbitrary fixed pixel logits.
    """

    def __init__(
        self,
        variance_weight=0.15,
        eps=1e-8,
    ):
        super().__init__()

        self.variance_weight = float(
            variance_weight
        )
        self.eps = float(
            eps
        )

    def forward(
        self,
        logits,
        gt,
    ):
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        p = torch.sigmoid(
            logits
        )

        g = gt.float()
        bg = 1.0 - g

        fg_den = g.sum(
            dim=(2, 3),
            keepdim=True,
        )

        bg_den = bg.sum(
            dim=(2, 3),
            keepdim=True,
        )

        fg_mean = (
            (
                p * g
            ).sum(
                dim=(2, 3),
                keepdim=True,
            )
            / (
                fg_den
                + self.eps
            )
        )

        bg_mean = (
            (
                p * bg
            ).sum(
                dim=(2, 3),
                keepdim=True,
            )
            / (
                bg_den
                + self.eps
            )
        )

        fg_var = (
            (
                (
                    p - fg_mean
                ).pow(2)
                * g
            ).sum(
                dim=(2, 3),
                keepdim=True,
            )
            / (
                fg_den
                + self.eps
            )
        )

        bg_var = (
            (
                (
                    p - bg_mean
                ).pow(2)
                * bg
            ).sum(
                dim=(2, 3),
                keepdim=True,
            )
            / (
                bg_den
                + self.eps
            )
        )

        return (
            (
                1.0
                - fg_mean
                + bg_mean
                + self.variance_weight
                * (
                    fg_var
                    + bg_var
                )
            ).mean()
        )


def soft_morphological_edge(
    value,
    kernel_size=3,
):
    pad = kernel_size // 2

    hi = F.max_pool2d(
        value,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )

    lo = -F.max_pool2d(
        -value,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )

    return torch.clamp(
        hi - lo,
        0.0,
        1.0,
    )


class BoundaryDiceLoss(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        eps=1e-6,
    ):
        super().__init__()

        self.kernel_size = int(
            kernel_size
        )
        self.eps = float(
            eps
        )

    def forward(
        self,
        logits,
        gt,
    ):
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pred = torch.sigmoid(
            logits
        )
        gt = gt.float()

        pred_edge = soft_morphological_edge(
            pred,
            self.kernel_size,
        )

        gt_edge = soft_morphological_edge(
            gt,
            self.kernel_size,
        )

        dims = (
            1,
            2,
            3,
        )

        inter = (
            pred_edge
            * gt_edge
        ).sum(
            dim=dims
        )

        denom = (
            pred_edge.sum(
                dim=dims
            )
            + gt_edge.sum(
                dim=dims
            )
        )

        dice = (
            2.0
            * inter
            + self.eps
        ) / (
            denom
            + self.eps
        )

        return 1.0 - dice.mean()


class SoftMAELoss(nn.Module):
    def forward(
        self,
        logits,
        gt,
    ):
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return torch.abs(
            torch.sigmoid(
                logits
            )
            - gt.float()
        ).mean()


# ============================================================================
# R15 final loss
# ============================================================================

class R15MainLoss(nn.Module):
    def __init__(
        self,
        lambda_f=0.65,
        lambda_curve_f=0.30,
        lambda_ssim=0.18,
        lambda_object=0.10,
        lambda_edge=0.08,
        lambda_mae=0.04,
        lambda_p176=0.03,
        lambda_p2=0.03,
    ):
        super().__init__()

        self.structure = BoundaryAwareStructureLoss()
        self.fmeasure = SoftFMeasureLoss(
            beta2=0.3
        )
        self.curve_f = EvaluationAlignedCurveFLoss()
        self.ssim = MultiScaleSSIMStructureLoss()
        self.object = ObjectConsistencyLoss()
        self.edge = BoundaryDiceLoss()
        self.mae = SoftMAELoss()

        self.lambda_f = float(lambda_f)
        self.lambda_curve_f = float(lambda_curve_f)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_object = float(lambda_object)
        self.lambda_edge = float(lambda_edge)
        self.lambda_mae = float(lambda_mae)
        self.lambda_p176 = float(lambda_p176)
        self.lambda_p2 = float(lambda_p2)

    def forward(
        self,
        final_logits,
        p176_logits,
        p2_logits,
        gt,
    ):
        structure = self.structure(
            final_logits,
            gt,
        )

        fmeasure = self.fmeasure(
            final_logits,
            gt,
        )

        curve_f = self.curve_f(
            final_logits,
            gt,
        )

        ssim = self.ssim(
            final_logits,
            gt,
        )

        object_loss = self.object(
            final_logits,
            gt,
        )

        edge = self.edge(
            final_logits,
            gt,
        )

        mae = self.mae(
            final_logits,
            gt,
        )

        p176 = (
            0.60
            * self.structure(
                p176_logits,
                gt,
            )
            + 0.40
            * self.fmeasure(
                p176_logits,
                gt,
            )
        )

        p2 = (
            0.55
            * self.structure(
                p2_logits,
                gt,
            )
            + 0.45
            * self.fmeasure(
                p2_logits,
                gt,
            )
        )

        total = (
            structure
            + self.lambda_f
            * fmeasure
            + self.lambda_curve_f
            * curve_f
            + self.lambda_ssim
            * ssim
            + self.lambda_object
            * object_loss
            + self.lambda_edge
            * edge
            + self.lambda_mae
            * mae
            + self.lambda_p176
            * p176
            + self.lambda_p2
            * p2
        )

        return {
            "total":
                total,

            "structure":
                structure,

            "f":
                fmeasure,

            "curve_f":
                curve_f,

            "ssim":
                ssim,

            "object":
                object_loss,

            "edge":
                edge,

            "mae":
                mae,

            "p176":
                p176,

            "p2":
                p2,
        }


# ============================================================================
# Region teacher supervision
# ============================================================================

class R15RegionTeacherLoss(nn.Module):
    """
    Expand R9 teacher errors into region-level repair targets.

    A single isolated wrong pixel should not force a broad semantic change.
    Conversely, a cluster of FN/FP evidence should activate an entire local
    semantic neighborhood. The target therefore combines max-pooled and
    average-pooled teacher error, constrained by GT class.
    """

    def __init__(
        self,
        region_size=44,
        seed_floor=0.08,
        seed_scale=0.38,
        eps=1e-8,
    ):
        super().__init__()

        self.region_size = int(
            region_size
        )
        self.seed_floor = float(
            seed_floor
        )
        self.seed_scale = float(
            seed_scale
        )
        self.eps = float(
            eps
        )

        self.structure = BoundaryAwareStructureLoss(
            pool_kernel=15,
            boundary_factor=4.0,
        )

        self.fmeasure = SoftFMeasureLoss(
            beta2=0.3
        )

        self.ssim = MultiScaleSSIMStructureLoss(
            windows=(
                5,
                9,
                15,
            ),
            max_size=88,
        )

    def _resize(
        self,
        value,
        size,
        nearest=False,
    ):
        if value.shape[-2:] == size:
            return value.float()

        if nearest:
            return F.interpolate(
                value.float(),
                size=size,
                mode="nearest",
            )

        return F.interpolate(
            value.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _region_mean(
        self,
        value,
        weight,
    ):
        num = (
            value
            * weight
        ).sum(
            dim=(1, 2, 3)
        )

        den = weight.sum(
            dim=(1, 2, 3)
        )

        valid = (
            den > 1e-6
        ).float()

        per_image = (
            num
            / (
                den
                + self.eps
            )
        )

        return (
            (
                per_image
                * valid
            ).sum()
            / (
                valid.sum()
                + self.eps
            )
        )

    def _region_targets(
        self,
        teacher_final_logits,
        gt,
        size,
    ):
        teacher = self._resize(
            teacher_final_logits.detach(),
            size,
        )

        G = self._resize(
            gt,
            size,
            nearest=True,
        )

        P = torch.sigmoid(
            teacher
        )

        fn_seed = (
            G
            * (
                1.0 - P
            )
        )

        fp_seed = (
            (
                1.0 - G
            )
            * P
        )

        fn_max = F.max_pool2d(
            fn_seed,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        fp_max = F.max_pool2d(
            fp_seed,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        fn_avg = F.avg_pool2d(
            fn_seed,
            kernel_size=9,
            stride=1,
            padding=4,
        )

        fp_avg = F.avg_pool2d(
            fp_seed,
            kernel_size=9,
            stride=1,
            padding=4,
        )

        fn_region_raw = (
            0.65
            * fn_max
            + 0.35
            * fn_avg
        )

        fp_region_raw = (
            0.65
            * fp_max
            + 0.35
            * fp_avg
        )

        fn_region = (
            G
            * torch.clamp(
                (
                    fn_region_raw
                    - self.seed_floor
                )
                / self.seed_scale,
                0.0,
                1.0,
            )
        )

        fp_region = (
            (
                1.0 - G
            )
            * torch.clamp(
                (
                    fp_region_raw
                    - self.seed_floor
                )
                / self.seed_scale,
                0.0,
                1.0,
            )
        )

        active = torch.clamp(
            fn_region
            + fp_region,
            0.0,
            1.0,
        )

        keep = (
            1.0
            - active
        )

        route_target = torch.cat(
            [
                keep,
                fn_region,
                fp_region,
            ],
            dim=1,
        )

        route_target = (
            route_target
            / (
                route_target.sum(
                    dim=1,
                    keepdim=True,
                )
                + self.eps
            )
        )

        return {
            "gt":
                G,

            "teacher_prob":
                P,

            "fn_region":
                fn_region,

            "fp_region":
                fp_region,

            "active":
                active,

            "keep":
                keep,

            "route_target":
                route_target,
        }

    def _route_loss(
        self,
        route_logits,
        targets,
    ):
        log_prob = F.log_softmax(
            route_logits,
            dim=1,
        )

        ce = -(
            targets[
                "route_target"
            ]
            * log_prob
        ).sum(
            dim=1,
            keepdim=True,
        )

        keep_loss = self._region_mean(
            ce,
            0.05
            + 0.15
            * targets[
                "keep"
            ],
        )

        fg_loss = self._region_mean(
            ce,
            targets[
                "fn_region"
            ].pow(
                0.60
            ),
        )

        bg_loss = self._region_mean(
            ce,
            targets[
                "fp_region"
            ].pow(
                0.60
            ),
        )

        return (
            0.20
            * keep_loss
            + 0.40
            * fg_loss
            + 0.40
            * bg_loss
        )

    def forward(
        self,
        final_logits,
        teacher_final_logits,
        teacher_p2_logits,
        region_route_logits44,
        region_aux_logits44,
        p2_base_logits,
        p2_repaired_logits,
        applied_d2_88,
        applied_p2_88,
        applied_d1_88,
        gt,
    ):
        size44 = region_route_logits44.shape[-2:]

        targets44 = self._region_targets(
            teacher_final_logits,
            gt,
            size44,
        )

        route = self._route_loss(
            region_route_logits44,
            targets44,
        )

        # Region auxiliary saliency is deliberately supervised as a real
        # segmentation representation, not only as a router classification.
        region_aux = (
            0.50
            * self.structure(
                region_aux_logits44,
                gt,
            )
            + 0.30
            * self.fmeasure(
                region_aux_logits44,
                gt,
            )
            + 0.20
            * self.ssim(
                region_aux_logits44,
                gt,
            )
        )

        p2_repaired = (
            0.55
            * self.structure(
                p2_repaired_logits,
                gt,
            )
            + 0.45
            * self.fmeasure(
                p2_repaired_logits,
                gt,
            )
        )

        # Compare repaired P2 to the frozen inherited P2. Only meaningful
        # regressions are penalized; improvement is handled by p2_repaired.
        gt88 = self._resize(
            gt,
            p2_repaired_logits.shape[-2:],
            nearest=True,
        )

        base_p2_error = torch.abs(
            gt88
            - torch.sigmoid(
                p2_base_logits.detach()
            )
        )

        repaired_p2_error = torch.abs(
            gt88
            - torch.sigmoid(
                p2_repaired_logits
            )
        )

        p2_regression = F.relu(
            repaired_p2_error
            - base_p2_error
            - 0.015
        ).mean()

        # Broad region residuals are allowed, but easy regions should not be
        # rewritten gratuitously.
        keep44 = targets44[
            "keep"
        ]

        def easy_feature_penalty(
            feature,
        ):
            feature44 = F.interpolate(
                feature.abs().mean(
                    dim=1,
                    keepdim=True,
                ),
                size=size44,
                mode="bilinear",
                align_corners=False,
            )

            return self._region_mean(
                feature44,
                0.03
                + keep44.pow(
                    1.5
                ),
            )

        easy_d2 = easy_feature_penalty(
            applied_d2_88
        )

        easy_d1 = easy_feature_penalty(
            applied_d1_88
        )

        p2_delta44 = F.interpolate(
            applied_p2_88.abs(),
            size=size44,
            mode="bilinear",
            align_corners=False,
        )

        easy_p2 = self._region_mean(
            p2_delta44,
            0.03
            + keep44.pow(
                1.5
            ),
        )

        # Final improvement gets additional emphasis on the broadened region
        # target; this connects region reasoning to the actual high-res output.
        final_size = final_logits.shape[-2:]

        final_prob = torch.sigmoid(
            final_logits
        )

        G_final = self._resize(
            gt,
            final_size,
            nearest=True,
        )

        final_error = torch.abs(
            G_final
            - final_prob
        )

        active_final = self._resize(
            targets44[
                "active"
            ],
            final_size,
        )

        hard_final = self._region_mean(
            final_error,
            active_final.pow(
                0.60
            ),
        )

        # Teacher P2 is used only as an EASY-region anchor. On active
        # teacher-error regions the repaired P2 must be free to outperform the
        # inherited prediction rather than distill its mistake.
        teacher_p2 = self._resize(
            teacher_p2_logits.detach(),
            p2_repaired_logits.shape[-2:],
        )

        teacher_p2_prob = torch.sigmoid(
            teacher_p2
        )

        keep88 = self._resize(
            targets44[
                "keep"
            ],
            p2_repaired_logits.shape[-2:],
        )

        p2_teacher_anchor = self._region_mean(
            F.smooth_l1_loss(
                torch.sigmoid(
                    p2_repaired_logits
                ),
                teacher_p2_prob,
                reduction="none",
                beta=0.03,
            ),
            0.03
            + keep88.pow(
                1.5
            ),
        )

        total = (
            0.20
            * route
            + 0.24
            * region_aux
            + 0.20
            * p2_repaired
            + 0.05
            * p2_regression
            + 0.12
            * hard_final
            + 0.05
            * p2_teacher_anchor
            + 0.045
            * easy_d2
            + 0.045
            * easy_d1
            + 0.05
            * easy_p2
        )

        with torch.no_grad():
            route_prob = torch.softmax(
                region_route_logits44,
                dim=1,
            )

            fg_route = route_prob[
                :,
                1:2,
            ]

            bg_route = route_prob[
                :,
                2:3,
            ]

            fg_on_fg = self._region_mean(
                fg_route,
                targets44[
                    "fn_region"
                ].pow(
                    0.60
                ),
            )

            bg_on_fg = self._region_mean(
                bg_route,
                targets44[
                    "fn_region"
                ].pow(
                    0.60
                ),
            )

            bg_on_bg = self._region_mean(
                bg_route,
                targets44[
                    "fp_region"
                ].pow(
                    0.60
                ),
            )

            fg_on_bg = self._region_mean(
                fg_route,
                targets44[
                    "fp_region"
                ].pow(
                    0.60
                ),
            )

        return {
            "total":
                total,

            "route":
                route,

            "region_aux":
                region_aux,

            "p2_repaired":
                p2_repaired,

            "p2_regression":
                p2_regression,

            "hard_final":
                hard_final,

            "p2_teacher_anchor":
                p2_teacher_anchor,

            "easy_d2":
                easy_d2,

            "easy_d1":
                easy_d1,

            "easy_p2":
                easy_p2,

            "region_active_mean":
                targets44[
                    "active"
                ].detach()
                .mean(),

            "region_fg_mean":
                targets44[
                    "fn_region"
                ].detach()
                .mean(),

            "region_bg_mean":
                targets44[
                    "fp_region"
                ].detach()
                .mean(),

            "fg_route_on_fg":
                fg_on_fg.detach(),

            "bg_route_on_fg":
                bg_on_fg.detach(),

            "bg_route_on_bg":
                bg_on_bg.detach(),

            "fg_route_on_bg":
                fg_on_bg.detach(),
        }


# ============================================================================
# R22 clean-anchored continuation losses
# ============================================================================

class R22SupervisedLoss(R15MainLoss):
    """
    Keep the proven R15 supervised objective unchanged.

    R22's novelty is the training distribution and clean anchoring, not a new
    segmentation objective.  This avoids confounding augmentation experiments
    with another loss redesign.
    """

    def __init__(self):
        super().__init__()


class R22CleanAnchorLoss(nn.Module):
    """
    Preserve the strong R15 clean-domain decision surface.

    Every optimizer step contains the ORIGINAL clean image/GT pair.  A frozen
    R15 anchor predicts that same clean image and the student is softly kept
    near it.

    In addition to probability preservation, a small multi-threshold
    preservation term keeps the per-image F-curve score shape from drifting.
    """

    def __init__(
        self,
        p2_weight=0.25,
        curve_weight=0.20,
        beta=0.025,
        tau=0.045,
        thresholds=None,
        eps=1e-8,
    ):
        super().__init__()

        if thresholds is None:
            thresholds = (
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            )

        self.p2_weight = float(p2_weight)
        self.curve_weight = float(curve_weight)
        self.beta = float(beta)
        self.tau = float(tau)
        self.thresholds = tuple(float(x) for x in thresholds)
        self.eps = float(eps)

    def _resize(self, value, size):
        if value.shape[-2:] == size:
            return value
        return F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _minmax(self, p):
        p_min = p.amin(
            dim=(2, 3),
            keepdim=True,
        ).detach()

        p_max = p.amax(
            dim=(2, 3),
            keepdim=True,
        ).detach()

        return (
            p - p_min
        ) / (
            p_max - p_min + self.eps
        )

    def _weighted_prob_distill(self, student_logits, teacher_logits):
        teacher_logits = self._resize(
            teacher_logits.detach(),
            student_logits.shape[-2:],
        )

        ps = torch.sigmoid(student_logits)
        pt = torch.sigmoid(teacher_logits)

        confidence = torch.clamp(
            2.0 * torch.abs(pt - 0.5),
            0.0,
            1.0,
        )

        weight = (
            0.55
            + 0.45
            * confidence
        )

        diff = F.smooth_l1_loss(
            ps,
            pt,
            reduction="none",
            beta=self.beta,
        )

        return (
            diff * weight
        ).sum() / (
            weight.sum() + self.eps
        )

    def _curve_preservation(self, student_logits, teacher_logits):
        teacher_logits = self._resize(
            teacher_logits.detach(),
            student_logits.shape[-2:],
        )

        ps = self._minmax(
            torch.sigmoid(student_logits)
        )

        pt = self._minmax(
            torch.sigmoid(teacher_logits)
        )

        losses = []

        for threshold in self.thresholds:
            s_mask = torch.sigmoid(
                (
                    ps - threshold
                ) / self.tau
            )

            t_mask = torch.sigmoid(
                (
                    pt - threshold
                ) / self.tau
            )

            losses.append(
                F.smooth_l1_loss(
                    s_mask,
                    t_mask,
                    beta=0.03,
                )
            )

        return torch.stack(
            losses
        ).mean()

    def forward(
        self,
        student_final_logits,
        teacher_final_logits,
        student_p2_logits,
        teacher_p2_logits,
    ):
        final = self._weighted_prob_distill(
            student_final_logits,
            teacher_final_logits,
        )

        p2 = self._weighted_prob_distill(
            student_p2_logits,
            teacher_p2_logits,
        )

        curve = self._curve_preservation(
            student_final_logits,
            teacher_final_logits,
        )

        total = (
            final
            + self.p2_weight
            * p2
            + self.curve_weight
            * curve
        )

        return {
            "total": total,
            "final": final,
            "p2": p2,
            "curve": curve,
        }


class R22TrustedAugConsistencyLoss(nn.Module):
    """
    Frozen-R15 weak-view teacher -> strong-view student consistency.

    Consistency is intentionally secondary to GT supervision.

    Quality-gated copy-paste returns a teacher_valid_mask that explicitly
    disables consistency around pasted pixels.  GT-correct confidence filtering
    then removes the remaining weak teacher locations that should not be
    trusted.
    """

    def __init__(
        self,
        fg_conf=0.90,
        bg_conf=0.10,
        p2_weight=0.25,
        beta=0.025,
        eps=1e-8,
    ):
        super().__init__()
        self.fg_conf = float(fg_conf)
        self.bg_conf = float(bg_conf)
        self.p2_weight = float(p2_weight)
        self.beta = float(beta)
        self.eps = float(eps)

    def _resize(self, value, size, nearest=False):
        if value.shape[-2:] == size:
            return value

        if nearest:
            return F.interpolate(
                value,
                size=size,
                mode="nearest",
            )

        return F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _masked_smooth_l1(self, student, teacher, mask):
        diff = F.smooth_l1_loss(
            student,
            teacher,
            reduction="none",
            beta=self.beta,
        )

        return (
            diff * mask
        ).sum() / (
            mask.sum() + self.eps
        )

    def forward(
        self,
        student_final_logits,
        teacher_final_logits,
        student_p2_logits,
        teacher_p2_logits,
        gt,
        teacher_valid_mask,
    ):
        target_size = student_final_logits.shape[-2:]

        G = self._resize(
            gt.float(),
            target_size,
            nearest=True,
        )

        valid = self._resize(
            teacher_valid_mask.float(),
            target_size,
            nearest=True,
        )

        teacher_final = self._resize(
            teacher_final_logits.detach(),
            target_size,
        )

        pt = torch.sigmoid(
            teacher_final
        )

        ps = torch.sigmoid(
            student_final_logits
        )

        trusted_fg = (
            (G > 0.5)
            & (pt >= self.fg_conf)
        )

        trusted_bg = (
            (G <= 0.5)
            & (pt <= self.bg_conf)
        )

        trusted = (
            (trusted_fg | trusted_bg).float()
            * valid
        )

        final = self._masked_smooth_l1(
            ps,
            pt,
            trusted,
        )

        p2_size = student_p2_logits.shape[-2:]

        teacher_p2 = self._resize(
            teacher_p2_logits.detach(),
            p2_size,
        )

        trusted_p2 = self._resize(
            trusted,
            p2_size,
            nearest=True,
        )

        p2 = self._masked_smooth_l1(
            torch.sigmoid(
                student_p2_logits
            ),
            torch.sigmoid(
                teacher_p2
            ),
            trusted_p2,
        )

        total = (
            final
            + self.p2_weight
            * p2
        )

        with torch.no_grad():
            trusted_fraction = trusted.mean()
            valid_fraction = valid.mean()

        return {
            "total": total,
            "final": final,
            "p2": p2,
            "trusted_fraction": trusted_fraction,
            "valid_fraction": valid_fraction,
        }
