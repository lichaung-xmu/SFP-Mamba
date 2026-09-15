# -*- coding: utf-8 -*-
"""
Reset-R15 region-Mamba semantic-frequency repair decoder.

The decoder keeps THMNet's hierarchical Mamba structure but changes two points:

1) MSA2 foreground Mamba receives a moderate, semantic-selected f88 detail
   from the independent RGB-frequency pyramid.

2) Final prediction is no longer limited to an 88x88 P1 followed by bilinear
   upsampling. A semantic-conditioned high-resolution refiner works at 176 and
   352 resolution using the same RGB-frequency pyramid.

No extra column-scan branch is used in R9. R8 already showed that its learned
axis gate was nearly zero.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.VMmamba_decoder import (
    Decoder as THMDecoder,
    MSA_mamba_head as OriginalMambaHead,
)

from model.SF_modules import (
    DecoderFrequencySelector,
    HighResolutionFrequencyRefiner,
    BalancedErrorCorrector,
    HardRegionDecisionCorrector,
    TriRouteSelectiveCorrector,
    ResidualConvBlock,
    inverse_bounded_sigmoid,
)


# ============================================================================
# Original MSA2 state copy
# ============================================================================

def _copy_original_msa_state(
    original_msa,
    new_msa,
):
    src = original_msa.state_dict()
    dst = new_msa.state_dict()

    matched = {}

    for key, value in src.items():
        candidates = [
            key,
        ]

        if key.startswith(
            "F_TA."
        ):
            candidates.insert(
                0,
                "F_TA.base."
                + key[
                    len(
                        "F_TA."
                    ):
                ],
            )

        for candidate in candidates:
            if (
                candidate in dst
                and dst[
                    candidate
                ].shape
                == value.shape
            ):
                matched[
                    candidate
                ] = (
                    value.detach()
                    .clone()
                )
                break

    merged = dict(
        dst
    )

    merged.update(
        matched
    )

    new_msa.load_state_dict(
        merged,
        strict=True,
    )

    print(
        "Reset-R15 inherited MSA2 init: copied {}/{} original tensors".format(
            len(
                matched
            ),
            len(
                src
            ),
        )
    )

    return (
        len(
            matched
        ),
        len(
            src
        ),
    )


# ============================================================================
# Detail-conditioned foreground Mamba
# ============================================================================

class DetailConditionedMambaHead(nn.Module):
    """
    Original THMNet foreground Mamba with one modification:

        masked semantic input
        + alpha_detail * selected RGB-frequency detail

    The detail has already been filtered by:
        physical frequency magnitude
        deep semantic support
        hard-background suppression
        spatial-channel selection

    No detail LayerNorm is applied after this filtering.

    runtime_detail_scale=0 gives the exact original MSA_mamba_head.
    """

    def __init__(
        self,
        dim=128,
        detail_max=0.80,
        detail_init=0.20,
    ):
        super().__init__()

        self.base = OriginalMambaHead(
            dim=dim
        )

        self.detail_max = float(
            detail_max
        )

        self.detail_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    detail_init,
                    detail_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_detail_scale = 1.0
        self._last_stats = {}

    @property
    def detail_strength(
        self,
    ):
        return (
            self.detail_max
            * torch.sigmoid(
                self.detail_logit
            )
        )

    def set_runtime_scale(
        self,
        scale,
    ):
        self.runtime_detail_scale = float(
            scale
        )

    def _pvm_forward(
        self,
        x,
        mask,
        detail,
    ):
        pvm = self.base.attn

        if x.dtype == torch.float16:
            x = x.float()

        if detail.dtype == torch.float16:
            detail = detail.float()

        b, c, h, w = x.shape

        if detail.shape[-2:] != (
            h,
            w,
        ):
            detail = F.interpolate(
                detail,
                size=(
                    h,
                    w,
                ),
                mode="bilinear",
                align_corners=False,
            )

        n = h * w

        x_flat = (
            x.reshape(
                b,
                c,
                n,
            )
            .transpose(
                -1,
                -2,
            )
        )

        x_norm = pvm.norm(
            x_flat
        )

        if mask is not None:
            if mask.shape[-2:] != (
                h,
                w,
            ):
                mask = F.interpolate(
                    mask,
                    size=(
                        h,
                        w,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

            mask_flat = (
                mask.view(
                    b,
                    1,
                    n,
                )
                .expand(
                    b,
                    c,
                    n,
                )
                .transpose(
                    -1,
                    -2,
                )
            )

            x_norm = (
                x_norm
                * mask_flat
            )

        detail_flat = (
            detail.reshape(
                b,
                c,
                n,
            )
            .transpose(
                -1,
                -2,
            )
        )

        applied_detail = (
            self.detail_strength.to(
                dtype=x_norm.dtype
            )
            * float(
                self.runtime_detail_scale
            )
            * detail_flat
        )

        mixed = (
            x_norm
            + applied_detail
        )

        chunks = torch.chunk(
            mixed,
            4,
            dim=2,
        )

        outputs = []

        for chunk in chunks:
            out = (
                pvm.mamba(
                    chunk
                )
                + pvm.skip_scale
                * chunk
            )

            outputs.append(
                out
            )

        y = torch.cat(
            outputs,
            dim=2,
        )

        y = pvm.norm(
            y
        )

        y = pvm.proj(
            y
        )

        y = (
            y.transpose(
                -1,
                -2,
            )
            .reshape(
                b,
                pvm.output_dim,
                h,
                w,
            )
        )

        with torch.no_grad():
            base_abs = x_norm.detach().abs().mean()
            detail_abs = applied_detail.detach().abs().mean()

            self._last_stats = {
                "detail_strength":
                    self.detail_strength.detach(),

                "detail_abs_mean":
                    detail.detach()
                    .abs()
                    .mean(),

                "injection_abs_mean":
                    detail_abs,

                "base_fg_abs_mean":
                    base_abs,

                "injection_ratio":
                    detail_abs
                    / (
                        base_abs
                        + 1e-8
                    ),
            }

        return y

    def forward(
        self,
        x,
        mask=None,
        detail=None,
    ):
        if (
            detail is None
            or abs(
                float(
                    self.runtime_detail_scale
                )
            ) < 1e-12
        ):
            return self.base(
                x,
                mask,
            )

        x_norm = self.base.norm1(
            x
        )

        attn_out = self._pvm_forward(
            x_norm,
            mask,
            detail,
        )

        x = (
            x
            + attn_out
        )

        x = (
            x
            + self.base.ffn(
                self.base.norm2(
                    x
                )
            )
        )

        return x

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )


# ============================================================================
# R9 MSA2
# ============================================================================

class SF_MSA2_module(nn.Module):
    """
    THMNet MSA2 with:
        semantic-selected RGB-frequency detail -> foreground Mamba
        conservative background-frequency mask suppression

    There is no positive mask boost.
    """

    def __init__(
        self,
        dim=128,
        bgmask_max=2.00,
        bgmask_init=0.35,
    ):
        super().__init__()

        self.dim = int(
            dim
        )

        self.B_TA = OriginalMambaHead(
            dim=dim
        )

        self.F_TA = DetailConditionedMambaHead(
            dim=dim,
            detail_max=0.80,
            detail_init=0.20,
        )

        self.TA = OriginalMambaHead(
            dim=dim
        )

        self.Fuse = nn.Conv2d(
            3 * dim,
            dim,
            kernel_size=3,
            padding=1,
        )

        self.Fuse2 = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=1,
            ),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                dim
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        self.selector = (
            DecoderFrequencySelector(
                dim=dim,
                freq_dim=64,
            )
        )

        self.bgmask_max = float(
            bgmask_max
        )

        self.bgmask_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    bgmask_init,
                    bgmask_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_bgmask_scale = 1.0

        self.last_aux = {}
        self._last_stats = {}

    @property
    def bgmask_strength(
        self,
    ):
        return (
            self.bgmask_max
            * torch.sigmoid(
                self.bgmask_logit
            )
        )

    def set_runtime_scales(
        self,
        detail_scale=1.0,
        bgmask_scale=1.0,
    ):
        self.F_TA.set_runtime_scale(
            detail_scale
        )

        self.runtime_bgmask_scale = float(
            bgmask_scale
        )

    def forward(
        self,
        x,
        side_x,
        mask,
        mid_semantic,
        freq_ctx,
    ):
        n, c, h, w = x.shape

        if side_x.shape[-2:] != (
            h,
            w,
        ):
            side_x = F.interpolate(
                side_x,
                size=(
                    h,
                    w,
                ),
                mode="bilinear",
                align_corners=False,
            )

        mask_x = F.interpolate(
            mask,
            size=(
                h,
                w,
            ),
            mode="bilinear",
            align_corners=False,
        )

        S = torch.sigmoid(
            mask_x.detach()
        )

        (
            selected_detail,
            support_prob,
            background_prob,
        ) = self.selector(
            deep_semantic=x,
            mid_semantic=mid_semantic,
            shallow_semantic=side_x,
            saliency=S,
            f88=freq_ctx[
                "f88"
            ],
            physical88=freq_ctx[
                "physical88"
            ],
        )

        physical88 = freq_ctx[
            "physical88"
        ]

        if physical88.shape[-2:] != (
            h,
            w,
        ):
            physical88 = F.interpolate(
                physical88,
                size=(
                    h,
                    w,
                ),
                mode="bilinear",
                align_corners=False,
            )

        if (
            abs(
                float(
                    self.runtime_bgmask_scale
                )
            ) < 1e-12
        ):
            S_ref = S
            bgmask_delta = torch.zeros_like(
                S
            )

        else:
            eps = 1e-4

            base_logit = torch.logit(
                torch.clamp(
                    S,
                    eps,
                    1.0
                    - eps,
                )
            )

            bgmask_delta = (
                self.bgmask_strength.to(
                    dtype=base_logit.dtype
                )
                * float(
                    self.runtime_bgmask_scale
                )
                * background_prob
                * physical88
            )

            S_ref = torch.sigmoid(
                base_logit
                - bgmask_delta
            )

        xf = self.F_TA(
            x,
            S_ref,
            detail=selected_detail,
        )

        xb = self.B_TA(
            x,
            1.0
            - S_ref,
        )

        xg = self.TA(
            x,
            None,
        )

        semantic = torch.cat(
            [
                xb,
                xf,
                xg,
            ],
            dim=1,
        ).view(
            n,
            3 * c,
            h,
            w,
        )

        semantic = self.Fuse(
            semantic
        )

        out = self.Fuse2(
            side_x
            + side_x
            * semantic
        )

        self.last_aux = {
            **self.selector.last_aux,

            "selected_detail":
                selected_detail,

            "refined_saliency":
                S_ref.detach(),

            "bgmask_delta":
                bgmask_delta,
        }

        with torch.no_grad():
            stats = self.selector.runtime_stats()

            stats.update(
                self.F_TA.runtime_stats()
            )

            stats.update(
                {
                    "bgmask_strength":
                        self.bgmask_strength.detach(),

                    "S_ref_abs_delta":
                        (
                            S_ref
                            - S
                        ).detach()
                        .abs()
                        .mean(),

                    "bgmask_delta_mean":
                        bgmask_delta.detach()
                        .mean(),
                }
            )

            self._last_stats = stats

        return out

    def training_aux_tensors(
        self,
    ):
        return dict(
            self.last_aux
        )

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )



# ============================================================================
# R15 region-level semantic-frequency repair
# ============================================================================

class RegionSemanticFrequencyMambaRepair(nn.Module):
    """
    Region-level repair before and after MSA2.

    R14 proved that sparse pixel routing can be correct, but it only changes a
    tiny fraction of pixels. R15 moves the new capacity one level earlier:
    44x44 region reasoning -> 88x88 decoder representation / saliency repair.

    The module consumes:
        - decoder D2 and P2;
        - PVT Stage2 feature x2;
        - stable anchor Stage3/Stage4 semantics;
        - RGB frequency evidence at 44;
        - physical / transition maps.

    A local branch and two Mamba branches are fused:
        - global region Mamba;
        - candidate-region Mamba guided by a learned KEEP/FG/BG region router.

    Three independent zero-initialized outputs are produced:
        1) D2 pre-MSA2 feature residual;
        2) P2 saliency-logit residual, which changes the mask seen by MSA2;
        3) D1 post-MSA2 feature residual.

    Therefore R15 can repair an entire ambiguous object region while still
    starting exactly from the inherited R14-E12 function.
    """

    def __init__(
        self,
        dim=128,
        feature_max=0.90,
        feature_init=0.35,
        mask_max=2.00,
        mask_init=0.80,
        post_max=0.90,
        post_init=0.30,
        route_temperature=0.90,
    ):
        super().__init__()

        self.dim = int(dim)
        self.route_temperature = float(
            route_temperature
        )

        self.d2_proj = nn.Sequential(
            nn.Conv2d(dim, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.x2_proj = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.x3_proj = nn.Sequential(
            nn.Conv2d(320, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.x4_proj = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        self.freq44_proj = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # Scalar maps:
        # p2_logit, p2_prob, uncertainty, prediction-edge,
        # physical-frequency, frequency-transition = 6.
        self.context_fuse = nn.Sequential(
            nn.Conv2d(
                64 + 64 + 64 + 32 + 64 + 6,
                160,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 160),
            nn.GELU(),
            nn.Conv2d(160, dim, 1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.local_branch = nn.Sequential(
            ResidualConvBlock(dim, dilation=1),
            ResidualConvBlock(dim, dilation=2),
            ResidualConvBlock(dim, dilation=3),
        )

        # Existing THMNet Mamba head; at 44x44 this models region continuity.
        self.global_mamba = OriginalMambaHead(
            dim=dim
        )

        self.candidate_mamba = OriginalMambaHead(
            dim=dim
        )

        self.region_route_head = nn.Sequential(
            nn.Conv2d(dim, 96, 3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 3, 1, bias=True),
        )

        # local/global/candidate + 3 route probability maps.
        self.branch_fuse = nn.Sequential(
            nn.Conv2d(
                3 * dim + 3,
                192,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            ResidualConvBlock(192, dilation=1),
            ResidualConvBlock(192, dilation=2),
            nn.Conv2d(192, dim, 1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        # Auxiliary head is outside the actual inference correction path, so
        # it may be nonzero at initialization and can train the internal
        # representation from the first step.
        self.region_aux_head = nn.Conv2d(
            dim,
            1,
            1,
            bias=True,
        )

        # Actual outputs are exact-zero initialized.
        self.d2_delta_head = nn.Conv2d(
            dim,
            dim,
            1,
            bias=True,
        )

        self.p2_delta_head = nn.Conv2d(
            dim,
            1,
            1,
            bias=True,
        )

        self.d1_delta_head = nn.Conv2d(
            dim,
            dim,
            1,
            bias=True,
        )

        for module in (
            self.d2_proj,
            self.x2_proj,
            self.x3_proj,
            self.x4_proj,
            self.freq44_proj,
            self.context_fuse,
            self.local_branch,
            self.branch_fuse,
        ):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        final_route = self.region_route_head[-1]
        nn.init.zeros_(final_route.weight)

        with torch.no_grad():
            final_route.bias.copy_(
                torch.tensor(
                    [1.20, -0.40, -0.40],
                    dtype=final_route.bias.dtype,
                )
            )

        nn.init.normal_(
            self.region_aux_head.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(self.region_aux_head.bias)

        for head in (
            self.d2_delta_head,
            self.p2_delta_head,
            self.d1_delta_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.feature_max = float(feature_max)
        self.mask_max = float(mask_max)
        self.post_max = float(post_max)

        self.feature_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    feature_init,
                    feature_max,
                ),
                dtype=torch.float32,
            )
        )

        self.mask_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    mask_init,
                    mask_max,
                ),
                dtype=torch.float32,
            )
        )

        self.post_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    post_init,
                    post_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_scale = 1.0
        self.runtime_feature_scale = 1.0
        self.runtime_mask_scale = 1.0
        self.runtime_post_scale = 1.0

        self.last_aux = {}
        self._last_stats = {}

    @property
    def feature_strength(self):
        return (
            self.feature_max
            * torch.sigmoid(self.feature_logit)
        )

    @property
    def mask_strength(self):
        return (
            self.mask_max
            * torch.sigmoid(self.mask_logit)
        )

    @property
    def post_strength(self):
        return (
            self.post_max
            * torch.sigmoid(self.post_logit)
        )

    def set_runtime_scales(
        self,
        scale=1.0,
        feature_scale=1.0,
        mask_scale=1.0,
        post_scale=1.0,
    ):
        self.runtime_scale = float(scale)
        self.runtime_feature_scale = float(feature_scale)
        self.runtime_mask_scale = float(mask_scale)
        self.runtime_post_scale = float(post_scale)

    @staticmethod
    def _uncertainty(logits):
        p = torch.sigmoid(logits.detach())
        return torch.clamp(
            4.0 * p * (1.0 - p),
            0.0,
            1.0,
        )

    @staticmethod
    def _edge(logits):
        p = torch.sigmoid(logits.detach())

        hi = F.max_pool2d(
            p,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        lo = -F.max_pool2d(
            -p,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        return torch.clamp(
            hi - lo,
            0.0,
            1.0,
        )

    def forward(
        self,
        d2,
        p2,
        stage2,
        anchor_x3,
        anchor_x4,
        freq_ctx,
    ):
        size44 = stage2.shape[-2:]
        size88 = d2.shape[-2:]

        d2_44 = self.d2_proj(
            F.interpolate(
                d2,
                size=size44,
                mode="bilinear",
                align_corners=False,
            )
        )

        x2_44 = self.x2_proj(stage2)

        x3_44 = self.x3_proj(
            F.interpolate(
                anchor_x3,
                size=size44,
                mode="bilinear",
                align_corners=False,
            )
        )

        x4_44 = self.x4_proj(
            F.interpolate(
                anchor_x4,
                size=size44,
                mode="bilinear",
                align_corners=False,
            )
        )

        f44 = self.freq44_proj(
            freq_ctx["f44"]
        )

        p2_44 = F.interpolate(
            p2,
            size=size44,
            mode="bilinear",
            align_corners=False,
        )

        p2_prob44 = torch.sigmoid(
            p2_44.detach()
        )

        uncertainty44 = self._uncertainty(
            p2_44
        )

        edge44 = self._edge(
            p2_44
        )

        h = self.context_fuse(
            torch.cat(
                [
                    d2_44,
                    x2_44,
                    x3_44,
                    x4_44,
                    f44,
                    p2_44,
                    p2_prob44,
                    uncertainty44,
                    edge44,
                    freq_ctx["physical44"],
                    freq_ctx["transition44"],
                ],
                dim=1,
            )
        )

        local = self.local_branch(h)

        route_logits = self.region_route_head(
            local
        )

        route_prob = torch.softmax(
            route_logits
            / self.route_temperature,
            dim=1,
        )

        keep_prob = route_prob[:, 0:1]
        fg_prob = route_prob[:, 1:2]
        bg_prob = route_prob[:, 2:3]

        candidate_prob = torch.clamp(
            1.0 - keep_prob,
            0.0,
            1.0,
        )

        global_context = self.global_mamba(
            h,
            None,
        )

        # Route is trained by direct region targets. Detaching it here avoids
        # a shortcut where the router learns only what makes Mamba easiest.
        candidate_context = self.candidate_mamba(
            h,
            candidate_prob.detach(),
        )

        fused = self.branch_fuse(
            torch.cat(
                [
                    local,
                    global_context,
                    candidate_context,
                    keep_prob,
                    fg_prob,
                    bg_prob,
                ],
                dim=1,
            )
        )

        region_aux_logits44 = self.region_aux_head(
            fused
        )

        d2_delta44 = self.d2_delta_head(fused)
        p2_delta44 = self.p2_delta_head(fused)
        d1_delta44 = self.d1_delta_head(fused)

        d2_delta88 = F.interpolate(
            d2_delta44,
            size=size88,
            mode="bilinear",
            align_corners=False,
        )

        p2_delta88 = F.interpolate(
            p2_delta44,
            size=size88,
            mode="bilinear",
            align_corners=False,
        )

        d1_delta88 = F.interpolate(
            d1_delta44,
            size=size88,
            mode="bilinear",
            align_corners=False,
        )

        overall = float(self.runtime_scale)

        applied_d2 = (
            overall
            * float(self.runtime_feature_scale)
            * self.feature_strength.to(dtype=d2.dtype)
            * d2_delta88
        )

        applied_p2 = (
            overall
            * float(self.runtime_mask_scale)
            * self.mask_strength.to(dtype=p2.dtype)
            * p2_delta88
        )

        applied_d1 = (
            overall
            * float(self.runtime_post_scale)
            * self.post_strength.to(dtype=d2.dtype)
            * d1_delta88
        )

        d2_repaired = d2 + applied_d2
        p2_repaired = p2 + applied_p2

        self.last_aux = {
            "r15_region_route_logits44":
                route_logits,

            "r15_region_keep44":
                keep_prob,

            "r15_region_fg44":
                fg_prob,

            "r15_region_bg44":
                bg_prob,

            "r15_region_candidate44":
                candidate_prob,

            "r15_region_aux_logits44":
                region_aux_logits44,

            "r15_region_d2_delta44":
                d2_delta44,

            "r15_region_p2_delta44":
                p2_delta44,

            "r15_region_d1_delta44":
                d1_delta44,

            "r15_region_applied_d2_88":
                applied_d2,

            "r15_region_applied_p2_88":
                applied_p2,

            "r15_region_applied_d1_88":
                applied_d1,

            "r15_region_p2_base":
                p2,

            "r15_region_p2_repaired":
                p2_repaired,
        }

        with torch.no_grad():
            d2_base_abs = d2.detach().abs().mean()
            d2_applied_abs = applied_d2.detach().abs().mean()

            self._last_stats = {
                "feature_strength":
                    self.feature_strength.detach(),

                "mask_strength":
                    self.mask_strength.detach(),

                "post_strength":
                    self.post_strength.detach(),

                "keep_mean":
                    keep_prob.detach().mean(),

                "fg_mean":
                    fg_prob.detach().mean(),

                "bg_mean":
                    bg_prob.detach().mean(),

                "candidate_mean":
                    candidate_prob.detach().mean(),

                "d2_applied_abs_mean":
                    d2_applied_abs,

                "d2_applied_ratio":
                    d2_applied_abs
                    / (
                        d2_base_abs + 1e-8
                    ),

                "p2_applied_abs_mean":
                    applied_p2.detach().abs().mean(),

                "d1_applied_abs_mean":
                    applied_d1.detach().abs().mean(),

                "region_aux_abs_mean":
                    region_aux_logits44.detach().abs().mean(),
            }

        return {
            "d2":
                d2_repaired,

            "p2":
                p2_repaired,

            "post_d1_delta":
                applied_d1,

            "context44":
                fused,
        }

    def training_aux_tensors(self):
        return dict(self.last_aux)

    def runtime_stats(self):
        return dict(self._last_stats)


# ============================================================================
# Full decoder
# ============================================================================

class Decoder(THMDecoder):
    """
    THMNet decoder with:
        R9 MSA2 frequency-conditioned foreground Mamba
        R9 176/352 high-resolution final refinement
    """

    def __init__(
        self,
        channels=128,
    ):
        super().__init__(
            channels
        )

        original_msa2 = self.MSA2

        new_msa2 = SF_MSA2_module(
            dim=channels
        )

        _copy_original_msa_state(
            original_msa2,
            new_msa2,
        )

        self.MSA2 = new_msa2

        # Keep the successful R9 high-resolution refiner unchanged.
        self.hr_refiner = (
            HighResolutionFrequencyRefiner(
                d1_dim=channels,
            )
        )

        # R15 main new capacity: region-level representation repair around
        # Stage2 / D2 / P2, with Mamba propagation at 44x44.
        self.region_repair = (
            RegionSemanticFrequencyMambaRepair(
                dim=channels,
            )
        )

        # Keep R14 tri-route as the final fine-grained correction layer.
        self.selective_corrector = (
            TriRouteSelectiveCorrector(
                d1_dim=channels,
                anchor_x3_dim=320,
                alpha_pos_max=2.50,
                alpha_pos_init=1.30,
                alpha_neg_max=2.50,
                alpha_neg_init=1.30,
                route_temperature=0.85,
            )
        )

        self.runtime_hr176_scale = 1.0
        self.runtime_hr352_scale = 1.0

        self._last_aux = {}
        self._last_stats = {}

    def set_runtime_scales(
        self,
        mamba_detail_scale=1.0,
        bgmask_scale=1.0,
        hr176_scale=1.0,
        hr352_scale=1.0,
        corrector_scale=1.0,
        region_scale=1.0,
        region_feature_scale=1.0,
        region_mask_scale=1.0,
        region_post_scale=1.0,
    ):
        self.MSA2.set_runtime_scales(
            detail_scale=mamba_detail_scale,
            bgmask_scale=bgmask_scale,
        )

        self.hr_refiner.set_runtime_scales(
            scale176=hr176_scale,
            scale352=hr352_scale,
        )

        self.runtime_hr176_scale = float(
            hr176_scale
        )

        self.runtime_hr352_scale = float(
            hr352_scale
        )

        self.selective_corrector.set_runtime_scale(
            corrector_scale
        )

        self.region_repair.set_runtime_scales(
            scale=region_scale,
            feature_scale=region_feature_scale,
            mask_scale=region_mask_scale,
            post_scale=region_post_scale,
        )

    def forward(
        self,
        x4,
        x3,
        x2,
        x1,
        raw_rgb,
        freq_ctx,
        shape,
        anchor_context=None,
        use_corrector=True,
        use_region_repair=True,
    ):
        E4 = self.conv1(
            x4
        )

        E3 = self.conv2(
            x3
        )

        E2 = self.conv3(
            x2
        )

        E1 = self.conv4(
            x1
        )

        if E4.size()[2:] != E3.size()[2:]:
            E4 = F.interpolate(
                E4,
                size=E3.size()[2:],
                mode="bilinear",
                align_corners=False,
            )

        if E2.size()[2:] != E3.size()[2:]:
            E2 = F.interpolate(
                E2,
                size=E3.size()[2:],
                mode="bilinear",
                align_corners=False,
            )

        E5 = self.conv_block(
            E4,
            E3,
            E2,
        )

        E5 = (
            E5
            + self.groupmamba_att(
                E5
            )
        )

        E4 = torch.cat(
            (
                E4,
                E5,
            ),
            1,
        )

        E3 = torch.cat(
            (
                E3,
                E5,
            ),
            1,
        )

        E2 = torch.cat(
            (
                E2,
                E5,
            ),
            1,
        )

        E4 = F.relu(
            self.fuse1(
                E4
            ),
            inplace=True,
        )

        E3 = F.relu(
            self.fuse2(
                E3
            ),
            inplace=True,
        )

        E2 = F.relu(
            self.fuse3(
                E2
            ),
            inplace=True,
        )

        P5 = self.predtrans5(
            E5
        )

        D4 = self.MSA5(
            E5,
            E4,
            P5,
        )

        D4 = F.interpolate(
            D4,
            size=E3.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        P4 = self.predtrans4(
            D4
        )

        D3 = self.MSA4(
            D4,
            E3,
            P4,
        )

        D3 = F.interpolate(
            D3,
            size=E2.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        P3 = self.predtrans3(
            D3
        )

        D2 = self.MSA3(
            D3,
            E2,
            P3,
        )

        D2 = F.interpolate(
            D2,
            size=E1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        P2_base = self.predtrans2(
            D2
        )

        if (
            use_region_repair
            and anchor_context is not None
        ):
            region = self.region_repair(
                d2=D2,
                p2=P2_base,
                stage2=x2,
                anchor_x3=anchor_context[
                    "x3"
                ],
                anchor_x4=anchor_context[
                    "x4"
                ],
                freq_ctx=freq_ctx,
            )

            D2_for_msa = region[
                "d2"
            ]

            P2 = region[
                "p2"
            ]

            region_post_d1 = region[
                "post_d1_delta"
            ]

            r15_region_aux = (
                self.region_repair
                .training_aux_tensors()
            )
        else:
            D2_for_msa = D2
            P2 = P2_base
            region_post_d1 = torch.zeros_like(
                D2
            )
            r15_region_aux = {}

        D1 = self.MSA2(
            D2_for_msa,
            E1,
            P2,
            mid_semantic=D3,
            freq_ctx=freq_ctx,
        )

        D1 = (
            D1
            + region_post_d1
        )

        P1_coarse = self.predtrans1(
            D1
        )

        semantic_support88 = (
            self.MSA2
            .selector
            .last_aux[
                "semantic_support"
            ]
        )

        hr = self.hr_refiner(
            d1=D1,
            p1_coarse=P1_coarse,
            raw_rgb=raw_rgb,
            freq_ctx=freq_ctx,
            semantic_support88=semantic_support88,
        )

        # Complete inherited R9 prediction for the current controlled
        # encoder pass.
        P1_r9_base = hr[
            "final_logits"
        ]

        if (
            use_corrector
            and anchor_context is not None
        ):
            P1 = self.selective_corrector(
                base_logits=P1_r9_base,
                p176_logits=hr[
                    "p176_logits"
                ],
                d1=D1,
                anchor_x3=anchor_context[
                    "x3"
                ],
                raw_rgb=raw_rgb,
                freq_ctx=freq_ctx,
                semantic_support88=semantic_support88,
            )

            r14_aux = (
                self.selective_corrector
                .training_aux_tensors()
            )
        else:
            P1 = P1_r9_base
            r14_aux = {}

        # Native high-resolution output, with arbitrary-size support.
        if P1.shape[-2:] != shape:
            P1 = F.interpolate(
                P1,
                size=shape,
                mode="bilinear",
                align_corners=False,
            )

        if P1_r9_base.shape[-2:] != shape:
            P1_r9_base_out = F.interpolate(
                P1_r9_base,
                size=shape,
                mode="bilinear",
                align_corners=False,
            )
        else:
            P1_r9_base_out = P1_r9_base

        P2_out = F.interpolate(
            P2,
            size=shape,
            mode="bilinear",
            align_corners=False,
        )

        P3_out = F.interpolate(
            P3,
            size=shape,
            mode="bilinear",
            align_corners=False,
        )

        P4_out = F.interpolate(
            P4,
            size=shape,
            mode="bilinear",
            align_corners=False,
        )

        P5_out = F.interpolate(
            P5,
            size=shape,
            mode="bilinear",
            align_corners=False,
        )

        self._last_aux = {
            **self.MSA2.training_aux_tensors(),
            **r15_region_aux,
            **r14_aux,

            "p1_r9_base_logits":
                P1_r9_base_out,

            "p1_coarse_logits":
                P1_coarse,

            "p2_base_logits":
                P2_base,

            "p176_logits":
                hr[
                    "p176_logits"
                ],

            "p1_final_logits":
                P1,

            "hr_edge_logits352":
                hr[
                    "edge_logits352"
                ],

            "hr_gate176":
                hr[
                    "gate176"
                ],

            "hr_gate352":
                hr[
                    "gate352"
                ],

            "hr_delta176":
                hr[
                    "delta176"
                ],

            "hr_delta352":
                hr[
                    "delta352"
                ],
        }

        with torch.no_grad():
            stats = self.MSA2.runtime_stats()

            for key, value in self.hr_refiner.runtime_stats().items():
                stats[
                    "r9_hr_{}".format(
                        key
                    )
                ] = value

            if (
                use_region_repair
                and anchor_context is not None
            ):
                for key, value in self.region_repair.runtime_stats().items():
                    stats[
                        "r15_region_{}".format(
                            key
                        )
                    ] = value

            if (
                use_corrector
                and anchor_context is not None
            ):
                for key, value in self.selective_corrector.runtime_stats().items():
                    stats[
                        "r14_{}".format(
                            key
                        )
                    ] = value

            self._last_stats = stats

        return (
            P5_out,
            P4_out,
            P3_out,
            P2_out,
            P1,
        )

    def training_aux_tensors(
        self,
    ):
        return dict(
            self._last_aux
        )

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )
