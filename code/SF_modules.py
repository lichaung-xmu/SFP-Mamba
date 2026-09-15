# -*- coding: utf-8 -*-
"""
Reset-R15 inherited semantic-feedback + selective calibration modules.

FA-PVT = Frequency-Aware PVT.
HR-Frequency = High-Resolution Frequency Semantic Refinement.

R9 addresses two ceilings observed across R1-R8:
1) the PVT backbone itself had remained frequency-agnostic;
2) the final decision was still largely formed at 88x88 and then bilinearly
   upsampled to 352x352.

R9 therefore:
- extracts an independent raw-RGB local-frequency pyramid at 352/176/88/44;
- injects that information into PVT Stage1 and Stage2 with learnable
  spatial-channel residual adapters;
- keeps a moderate selected frequency detail path into foreground Mamba;
- performs final semantic-conditioned refinement at 176 and 352 resolution.

All gates are learned. There is no global non-zero detail floor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Initialization helpers
# ============================================================================

def init_conv_norm(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.zeros_(
                module.bias
            )

    elif isinstance(module, nn.GroupNorm):
        if module.weight is not None:
            nn.init.ones_(
                module.weight
            )
        if module.bias is not None:
            nn.init.zeros_(
                module.bias
            )


def inverse_bounded_sigmoid(
    value,
    maximum,
):
    unit = float(value) / float(maximum)

    if not (
        0.0 < unit < 1.0
    ):
        raise ValueError(
            "Expected 0 < value < maximum, got value={} maximum={}".format(
                value,
                maximum,
            )
        )

    return math.log(
        unit
        / (
            1.0 - unit
        )
    )


# ============================================================================
# Raw RGB stationary high-pass
# ============================================================================

class RGBStationaryHighPass(nn.Module):
    """
    Four fixed stride-1 local high-pass filters on RGB.

    Returned maps preserve exact input HxW coordinates.
    """

    def __init__(
        self,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = float(
            eps
        )

        kernels = torch.tensor(
            [
                [
                    [-1.0, -2.0, -1.0],
                    [ 0.0,  0.0,  0.0],
                    [ 1.0,  2.0,  1.0],
                ],
                [
                    [-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0],
                ],
                [
                    [ 2.0,  1.0,  0.0],
                    [ 1.0,  0.0, -1.0],
                    [ 0.0, -1.0, -2.0],
                ],
                [
                    [ 0.0, -1.0,  0.0],
                    [-1.0,  4.0, -1.0],
                    [ 0.0, -1.0,  0.0],
                ],
            ],
            dtype=torch.float32,
        )

        denom = torch.sqrt(
            kernels.square().sum(
                dim=(1, 2),
                keepdim=True,
            )
            + self.eps
        )

        kernels = kernels / denom

        self.register_buffer(
            "kernels",
            kernels.unsqueeze(
                1
            ),
            persistent=True,
        )

    def forward(
        self,
        rgb,
    ):
        if (
            rgb.ndim != 4
            or rgb.shape[1] != 3
        ):
            raise ValueError(
                "RGBStationaryHighPass expects [B,3,H,W]."
            )

        b, c, h, w = rgb.shape

        weight = (
            self.kernels
            .to(
                device=rgb.device,
                dtype=rgb.dtype,
            )
            .repeat(
                c,
                1,
                1,
                1,
            )
        )

        response = F.conv2d(
            rgb,
            weight,
            bias=None,
            stride=1,
            padding=1,
            groups=c,
        )

        response = response.view(
            b,
            c,
            4,
            h,
            w,
        )

        h_band = response[:, :, 0]
        v_band = response[:, :, 1]
        d_band = response[:, :, 2]
        l_band = response[:, :, 3]

        energy_sq = (
            h_band.square()
            + v_band.square()
            + d_band.square()
            + l_band.square()
        ) / 4.0

        energy = torch.clamp(
            torch.sqrt(
                torch.clamp(
                    energy_sq,
                    min=0.0,
                )
                + self.eps
            )
            - self.eps ** 0.5,
            min=0.0,
        ).mean(
            dim=1,
            keepdim=True,
        )

        mean = energy.mean(
            dim=(2, 3),
            keepdim=True,
        )

        std = energy.std(
            dim=(2, 3),
            keepdim=True,
            unbiased=False,
        )

        strength = torch.clamp(
            energy
            / (
                2.0
                * (
                    mean
                    + std
                    + self.eps
                )
            ),
            0.0,
            1.0,
        )

        local_max = F.max_pool2d(
            strength,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        local_min = -F.max_pool2d(
            -strength,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        transition = torch.clamp(
            local_max
            - local_min,
            0.0,
            1.0,
        )

        return (
            h_band,
            v_band,
            d_band,
            l_band,
            strength,
            transition,
        )


# ============================================================================
# RGB local-frequency pyramid
# ============================================================================

class RGBFrequencyPyramid(nn.Module):
    """
    Independent RGB local-frequency pyramid.

    For a 352x352 input:
        f352 : 32 channels
        f176 : 48 channels
        f88  : 64 channels
        f44  : 128 channels

    The learned feature at each scale is multiplied by a physical frequency
    gate AFTER normalization, so flat regions cannot become strong frequency
    responses merely because GroupNorm rescales them.

    edge_logits352 / edge_logits176 are auxiliary local-structure heads only.
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.highpass = (
            RGBStationaryHighPass()
        )

        # 12 high-pass channels + 3 normalized RGB channels.
        self.stem352 = nn.Sequential(
            nn.Conv2d(
                15,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                groups=32,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
        )

        self.down176 = nn.Sequential(
            nn.Conv2d(
                32,
                48,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),
            nn.Conv2d(
                48,
                48,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),
        )

        self.down88 = nn.Sequential(
            nn.Conv2d(
                48,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.down44 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),
        )

        self.edge352 = nn.Conv2d(
            32,
            1,
            kernel_size=1,
            bias=True,
        )

        self.edge176 = nn.Conv2d(
            48,
            1,
            kernel_size=1,
            bias=True,
        )

        for module in (
            self.stem352,
            self.down176,
            self.down88,
            self.down44,
        ):
            module.apply(
                init_conv_norm
            )

        for head in (
            self.edge352,
            self.edge176,
        ):
            nn.init.normal_(
                head.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(
                head.bias
            )

    @staticmethod
    def _reduce_map(
        value,
        size,
    ):
        if value.shape[-2:] == size:
            return value

        max_map = F.adaptive_max_pool2d(
            value,
            size,
        )

        avg_map = F.adaptive_avg_pool2d(
            value,
            size,
        )

        return torch.clamp(
            0.60 * max_map
            + 0.40 * avg_map,
            0.0,
            1.0,
        )

    @staticmethod
    def _physical_gate(
        strength,
        transition,
    ):
        return torch.clamp(
            strength
            * (
                0.35
                + 0.65
                * transition
            ),
            0.0,
            1.0,
        )

    def forward(
        self,
        rgb,
    ):
        (
            h_band,
            v_band,
            d_band,
            l_band,
            strength352,
            transition352,
        ) = self.highpass(
            rgb
        )

        bands = torch.cat(
            [
                h_band,
                v_band,
                d_band,
                l_band,
                rgb,
            ],
            dim=1,
        )

        f352 = self.stem352(
            bands
        )

        physical352 = self._physical_gate(
            strength352,
            transition352,
        )

        f352 = (
            f352
            * physical352
        )

        f176 = self.down176(
            f352
        )

        size176 = f176.shape[-2:]

        strength176 = self._reduce_map(
            strength352,
            size176,
        )

        transition176 = self._reduce_map(
            transition352,
            size176,
        )

        physical176 = self._physical_gate(
            strength176,
            transition176,
        )

        f176 = (
            f176
            * physical176
        )

        f88 = self.down88(
            f176
        )

        size88 = f88.shape[-2:]

        strength88 = self._reduce_map(
            strength352,
            size88,
        )

        transition88 = self._reduce_map(
            transition352,
            size88,
        )

        physical88 = self._physical_gate(
            strength88,
            transition88,
        )

        f88 = (
            f88
            * physical88
        )

        f44 = self.down44(
            f88
        )

        size44 = f44.shape[-2:]

        strength44 = self._reduce_map(
            strength352,
            size44,
        )

        transition44 = self._reduce_map(
            transition352,
            size44,
        )

        physical44 = self._physical_gate(
            strength44,
            transition44,
        )

        f44 = (
            f44
            * physical44
        )

        edge_logits352 = self.edge352(
            f352
        )

        edge_logits176 = self.edge176(
            f176
        )

        return {
            "f352":
                f352,

            "f176":
                f176,

            "f88":
                f88,

            "f44":
                f44,

            "strength352":
                strength352,

            "strength176":
                strength176,

            "strength88":
                strength88,

            "strength44":
                strength44,

            "transition352":
                transition352,

            "transition176":
                transition176,

            "transition88":
                transition88,

            "transition44":
                transition44,

            "physical352":
                physical352,

            "physical176":
                physical176,

            "physical88":
                physical88,

            "physical44":
                physical44,

            "edge_logits352":
                edge_logits352,

            "edge_logits176":
                edge_logits176,
        }


# ============================================================================
# Frequency-aware PVT shallow-stage adapter
# ============================================================================

class FrequencyStageAdapter(nn.Module):
    """
    Inject an independent RGB-frequency feature into a PVT stage.

    The gate is a full spatial-channel gate.

    delta =
        Phi[x, f, x*f]

    x' =
        x + alpha * gate * delta

    There is no hard-coded non-zero gate floor.
    """

    def __init__(
        self,
        dim,
        freq_dim,
        hidden=None,
        alpha_max=0.80,
        alpha_init=0.20,
        gate_bias=-1.50,
    ):
        super().__init__()

        if hidden is None:
            hidden = dim

        self.freq_proj = nn.Sequential(
            nn.Conv2d(
                freq_dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                dim,
            ),
            nn.GELU(),
        )

        self.delta = nn.Sequential(
            nn.Conv2d(
                3 * dim,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                hidden,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                hidden,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                dim,
                kernel_size=1,
                bias=False,
            ),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                3 * dim,
                dim,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.freq_proj.apply(
            init_conv_norm
        )

        self.delta.apply(
            init_conv_norm
        )

        nn.init.zeros_(
            self.gate[0].weight
        )

        nn.init.constant_(
            self.gate[0].bias,
            gate_bias,
        )

        self.alpha_max = float(
            alpha_max
        )

        self.alpha_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    alpha_init,
                    alpha_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_scale = 1.0
        self._last_stats = {}

    @property
    def alpha(
        self,
    ):
        return (
            self.alpha_max
            * torch.sigmoid(
                self.alpha_logit
            )
        )

    def set_runtime_scale(
        self,
        scale,
    ):
        self.runtime_scale = float(
            scale
        )

    def forward(
        self,
        x,
        freq,
        external_mod=None,
        external_strength=0.80,
    ):
        if freq.shape[-2:] != x.shape[-2:]:
            freq = F.interpolate(
                freq,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        f = self.freq_proj(
            freq
        )

        interaction = (
            x
            * torch.tanh(
                f
            )
        )

        packed = torch.cat(
            [
                x,
                f,
                interaction,
            ],
            dim=1,
        )

        gate = self.gate(
            packed
        )

        delta = self.delta(
            packed
        )

        if external_mod is None:
            modulation = torch.ones_like(
                gate
            )
        else:
            if external_mod.shape[-2:] != x.shape[-2:]:
                external_mod = F.interpolate(
                    external_mod,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if external_mod.shape[1] == 1:
                external_mod = external_mod.expand(
                    -1,
                    gate.shape[1],
                    -1,
                    -1,
                )

            if external_mod.shape[1] != gate.shape[1]:
                raise ValueError(
                    "external_mod channels {} do not match adapter dim {}".format(
                        external_mod.shape[1],
                        gate.shape[1],
                    )
                )

            # Semantic feedback scales the already-successful R9 frequency
            # residual without changing its sign. Range with strength=.8:
            # approximately [0.45, 2.23].
            modulation = torch.exp(
                float(
                    external_strength
                )
                * torch.tanh(
                    external_mod
                )
            )

        applied = (
            self.alpha.to(
                dtype=x.dtype
            )
            * float(
                self.runtime_scale
            )
            * gate
            * delta
            * modulation
        )

        out = (
            x
            + applied
        )

        with torch.no_grad():
            base_abs = x.detach().abs().mean()
            applied_abs = applied.detach().abs().mean()

            self._last_stats = {
                "alpha":
                    self.alpha.detach(),

                "gate_mean":
                    gate.detach()
                    .mean(),

                "delta_abs_mean":
                    delta.detach()
                    .abs()
                    .mean(),

                "applied_abs_mean":
                    applied_abs,

                "applied_ratio":
                    applied_abs
                    / (
                        base_abs
                        + 1e-8
                    ),

                "modulation_mean":
                    modulation.detach()
                    .mean(),

                "modulation_min":
                    modulation.detach()
                    .amin(),

                "modulation_max":
                    modulation.detach()
                    .amax(),
            }

        return (
            out,
            gate,
        )

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )


# ============================================================================
# Decoder frequency selector
# ============================================================================

class DecoderFrequencySelector(nn.Module):
    """
    Select f88 for the foreground Mamba using D2/D3/E1 semantics.

    Unlike R7, there is no global detail floor.
    Unlike R8, the same RGB-frequency information has already influenced
    PVT Stage1/Stage2, so this path is deliberately moderate.

    Outputs:
        selected_detail : [B,128,H,W]
        support_logits  : [B,1,H,W]
        background_logits : [B,1,H,W]
    """

    def __init__(
        self,
        dim=128,
        freq_dim=64,
    ):
        super().__init__()

        self.freq_proj = nn.Sequential(
            nn.Conv2d(
                freq_dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                dim,
            ),
            nn.GELU(),
        )

        self.semantic_fuse = nn.Sequential(
            nn.Conv2d(
                4 * dim + 4,
                2 * dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                2 * dim,
            ),
            nn.GELU(),

            nn.Conv2d(
                2 * dim,
                dim,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                dim,
            ),
            nn.GELU(),
        )

        self.support_head = nn.Conv2d(
            dim,
            1,
            kernel_size=1,
            bias=True,
        )

        self.background_head = nn.Conv2d(
            dim,
            1,
            kernel_size=1,
            bias=True,
        )

        self.channel_gate = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=True,
        )

        self.detail_adapter = nn.Sequential(
            nn.Conv2d(
                3 * dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                dim,
            ),
            nn.GELU(),

            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                dim,
            ),
            nn.GELU(),

            nn.Conv2d(
                dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
        )

        self.freq_proj.apply(
            init_conv_norm
        )

        self.semantic_fuse.apply(
            init_conv_norm
        )

        self.detail_adapter.apply(
            init_conv_norm
        )

        nn.init.normal_(
            self.support_head.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.support_head.bias,
            -1.00,
        )

        nn.init.normal_(
            self.background_head.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.background_head.bias,
            -1.50,
        )

        nn.init.zeros_(
            self.channel_gate.weight
        )

        nn.init.zeros_(
            self.channel_gate.bias
        )

        self.last_aux = {}
        self._last_stats = {}

    def forward(
        self,
        deep_semantic,
        mid_semantic,
        shallow_semantic,
        saliency,
        f88,
        physical88,
    ):
        size = deep_semantic.shape[-2:]

        def resize(
            value,
        ):
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            return value

        mid_semantic = resize(
            mid_semantic
        )

        shallow_semantic = resize(
            shallow_semantic
        )

        f88 = resize(
            f88
        )

        physical88 = resize(
            physical88
        )

        saliency = resize(
            saliency
        )

        S = torch.clamp(
            saliency,
            0.0,
            1.0,
        )

        Smax = F.max_pool2d(
            S,
            kernel_size=9,
            stride=1,
            padding=4,
        )

        Smin = -F.max_pool2d(
            -S,
            kernel_size=9,
            stride=1,
            padding=4,
        )

        uncertainty = torch.clamp(
            4.0
            * S
            * (
                1.0 - S
            ),
            0.0,
            1.0,
        )

        contrast = torch.clamp(
            Smax
            - Smin,
            0.0,
            1.0,
        )

        f = self.freq_proj(
            f88
        )

        latent = self.semantic_fuse(
            torch.cat(
                [
                    deep_semantic,
                    mid_semantic,
                    shallow_semantic,
                    f,
                    S,
                    uncertainty,
                    contrast,
                    physical88,
                ],
                dim=1,
            )
        )

        support_logits = self.support_head(
            latent
        )

        background_logits = self.background_head(
            latent
        )

        support_prob = torch.sigmoid(
            support_logits
        )

        background_prob = torch.sigmoid(
            background_logits
        )

        channel_gate = torch.sigmoid(
            self.channel_gate(
                latent
            )
        )

        aligned = self.detail_adapter(
            torch.cat(
                [
                    f,
                    shallow_semantic,
                    f
                    * torch.tanh(
                        shallow_semantic
                    ),
                ],
                dim=1,
            )
        )

        semantic_support = torch.clamp(
            support_prob
            * (
                1.0
                - 0.90
                * background_prob
            ),
            0.0,
            1.0,
        )

        selected_detail = (
            aligned
            * physical88
            * semantic_support
            * channel_gate
        )

        self.last_aux = {
            "support_logits":
                support_logits,

            "background_logits":
                background_logits,

            "support_prob":
                support_prob,

            "background_prob":
                background_prob,

            "channel_gate":
                channel_gate,

            "channel_gate_mean":
                channel_gate.mean(
                    dim=1,
                    keepdim=True,
                ),

            "semantic_support":
                semantic_support,

            "physical88":
                physical88.detach(),

            "base_saliency":
                S.detach(),
        }

        with torch.no_grad():
            self._last_stats = {
                "support_mean":
                    support_prob.detach()
                    .mean(),

                "background_mean":
                    background_prob.detach()
                    .mean(),

                "channel_gate_mean":
                    channel_gate.detach()
                    .mean(),

                "semantic_support_mean":
                    semantic_support.detach()
                    .mean(),

                "selected_detail_abs_mean":
                    selected_detail.detach()
                    .abs()
                    .mean(),
            }

        return (
            selected_detail,
            support_prob,
            background_prob,
        )

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )


# ============================================================================
# High-resolution refinement blocks
# ============================================================================

class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        dilation=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                dim,
            ),
            nn.GELU(),

            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                dim,
            ),
        )

        self.block.apply(
            init_conv_norm
        )

    def forward(
        self,
        x,
    ):
        return F.gelu(
            x
            + self.block(
                x
            )
        )


class HighResolutionFrequencyRefiner(nn.Module):
    """
    Semantic-conditioned 176/352 final refinement.

    The coarse 88x88 P1 remains the semantic anchor.

    At 176:
        upsample(D1) + f176 + coarse prediction + semantic support

    At 352:
        upsample(HR176) + f352 + raw RGB + coarse prediction
        + semantic support + physical frequency

    Both residual gates are learned from their own fused features.

    The final prediction is:
        P_final =
            up(P1_coarse)
            + alpha176 * gate176 * delta176_up
            + alpha352 * gate352 * delta352
    """

    def __init__(
        self,
        d1_dim=128,
        alpha176_max=1.50,
        alpha176_init=0.30,
        alpha352_max=2.00,
        alpha352_init=0.45,
    ):
        super().__init__()

        self.sem176 = nn.Sequential(
            nn.Conv2d(
                d1_dim,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.freq176 = nn.Sequential(
            nn.Conv2d(
                48,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.fuse176 = nn.Sequential(
            nn.Conv2d(
                64 + 64 + 4,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
            ResidualConvBlock(
                64,
                dilation=1,
            ),
            ResidualConvBlock(
                64,
                dilation=2,
            ),
        )

        self.delta176 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )

        self.gate176 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )

        self.up352 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
        )

        self.freq352 = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
        )

        self.rgb352 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.GELU(),
        )

        self.fuse352 = nn.Sequential(
            nn.Conv2d(
                32 + 32 + 16 + 5,
                48,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),

            ResidualConvBlock(
                48,
                dilation=1,
            ),

            ResidualConvBlock(
                48,
                dilation=2,
            ),

            nn.Conv2d(
                48,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
        )

        self.delta352 = nn.Conv2d(
            32,
            1,
            kernel_size=1,
            bias=True,
        )

        self.gate352 = nn.Conv2d(
            32,
            1,
            kernel_size=1,
            bias=True,
        )

        self.edge352 = nn.Conv2d(
            32,
            1,
            kernel_size=1,
            bias=True,
        )

        for module in (
            self.sem176,
            self.freq176,
            self.fuse176,
            self.up352,
            self.freq352,
            self.rgb352,
            self.fuse352,
        ):
            module.apply(
                init_conv_norm
            )

        for head in (
            self.delta176,
            self.gate176,
            self.delta352,
            self.gate352,
            self.edge352,
        ):
            nn.init.normal_(
                head.weight,
                mean=0.0,
                std=0.01,
            )

        nn.init.zeros_(
            self.delta176.bias
        )

        nn.init.constant_(
            self.gate176.bias,
            -0.50,
        )

        nn.init.zeros_(
            self.delta352.bias
        )

        nn.init.constant_(
            self.gate352.bias,
            -0.30,
        )

        nn.init.zeros_(
            self.edge352.bias
        )

        self.alpha176_max = float(
            alpha176_max
        )

        self.alpha352_max = float(
            alpha352_max
        )

        self.alpha176_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    alpha176_init,
                    alpha176_max,
                ),
                dtype=torch.float32,
            )
        )

        self.alpha352_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    alpha352_init,
                    alpha352_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_scale176 = 1.0
        self.runtime_scale352 = 1.0
        self._last_stats = {}

    @property
    def alpha176(
        self,
    ):
        return (
            self.alpha176_max
            * torch.sigmoid(
                self.alpha176_logit
            )
        )

    @property
    def alpha352(
        self,
    ):
        return (
            self.alpha352_max
            * torch.sigmoid(
                self.alpha352_logit
            )
        )

    def set_runtime_scales(
        self,
        scale176=1.0,
        scale352=1.0,
    ):
        self.runtime_scale176 = float(
            scale176
        )
        self.runtime_scale352 = float(
            scale352
        )

    def forward(
        self,
        d1,
        p1_coarse,
        raw_rgb,
        freq_ctx,
        semantic_support88,
    ):
        f176 = freq_ctx[
            "f176"
        ]

        f352 = freq_ctx[
            "f352"
        ]

        physical176 = freq_ctx[
            "physical176"
        ]

        physical352 = freq_ctx[
            "physical352"
        ]

        size176 = f176.shape[-2:]
        size352 = raw_rgb.shape[-2:]

        sem176 = F.interpolate(
            d1,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        sem176 = self.sem176(
            sem176
        )

        freq176 = self.freq176(
            f176
        )

        coarse176 = F.interpolate(
            p1_coarse,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        support176 = F.interpolate(
            semantic_support88,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty176 = torch.clamp(
            4.0
            * torch.sigmoid(
                coarse176.detach()
            )
            * (
                1.0
                - torch.sigmoid(
                    coarse176.detach()
                )
            ),
            0.0,
            1.0,
        )

        h176 = self.fuse176(
            torch.cat(
                [
                    sem176,
                    freq176,
                    coarse176,
                    support176,
                    uncertainty176,
                    physical176,
                ],
                dim=1,
            )
        )

        delta176 = self.delta176(
            h176
        )

        gate176 = torch.sigmoid(
            self.gate176(
                h176
            )
        )

        applied176 = (
            self.alpha176.to(
                dtype=delta176.dtype
            )
            * float(
                self.runtime_scale176
            )
            * gate176
            * delta176
        )

        p176 = (
            coarse176
            + applied176
        )

        h352 = F.interpolate(
            h176,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        h352 = self.up352(
            h352
        )

        freq352 = self.freq352(
            f352
        )

        rgb352 = self.rgb352(
            raw_rgb
        )

        p176_up = F.interpolate(
            p176,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        support352 = F.interpolate(
            semantic_support88,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty352 = torch.clamp(
            4.0
            * torch.sigmoid(
                p176_up.detach()
            )
            * (
                1.0
                - torch.sigmoid(
                    p176_up.detach()
                )
            ),
            0.0,
            1.0,
        )

        transition352 = freq_ctx[
            "transition352"
        ]

        h352 = self.fuse352(
            torch.cat(
                [
                    h352,
                    freq352,
                    rgb352,
                    p176_up,
                    support352,
                    uncertainty352,
                    physical352,
                    transition352,
                ],
                dim=1,
            )
        )

        delta352 = self.delta352(
            h352
        )

        gate352 = torch.sigmoid(
            self.gate352(
                h352
            )
        )

        applied352 = (
            self.alpha352.to(
                dtype=delta352.dtype
            )
            * float(
                self.runtime_scale352
            )
            * gate352
            * delta352
        )

        final_logits = (
            p176_up
            + applied352
        )

        edge_logits352 = self.edge352(
            h352
        )

        with torch.no_grad():
            self._last_stats = {
                "alpha176":
                    self.alpha176.detach(),

                "alpha352":
                    self.alpha352.detach(),

                "gate176_mean":
                    gate176.detach()
                    .mean(),

                "gate352_mean":
                    gate352.detach()
                    .mean(),

                "delta176_abs_mean":
                    delta176.detach()
                    .abs()
                    .mean(),

                "delta352_abs_mean":
                    delta352.detach()
                    .abs()
                    .mean(),

                "applied176_abs_mean":
                    applied176.detach()
                    .abs()
                    .mean(),

                "applied352_abs_mean":
                    applied352.detach()
                    .abs()
                    .mean(),
            }

        return {
            "final_logits":
                final_logits,

            "p176_logits":
                p176,

            "delta176":
                delta176,

            "delta352":
                delta352,

            "gate176":
                gate176,

            "gate352":
                gate352,

            "edge_logits352":
                edge_logits352,
        }

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )


# ============================================================================
# R12 semantic feedback controller
# ============================================================================

class SemanticFeedbackController(nn.Module):
    """
    Deep semantic controller that READS Stage3/Stage4 semantics and controls
    the already-successful R9 Stage1/Stage2 frequency residuals.

    It does NOT inject frequency into Stage3.

    Inputs:
        anchor_x3 : [B,320,22,22]
        anchor_x4 : [B,512,11,11]
        physical88 / physical44

    Outputs:
        mod1 : [B,64,88,88]
        mod2 : [B,128,44,44]
        scalar1 / scalar2 : [B,1,...] for direct auxiliary supervision.

    Final control projections are zero initialized:
        mod1(init)=mod2(init)=0
    therefore the original R9 adapter modulation is exactly 1 at init.
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.x3_proj = nn.Sequential(
            nn.Conv2d(
                320,
                128,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),
        )

        self.x4_proj = nn.Sequential(
            nn.Conv2d(
                512,
                128,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),
        )

        self.fuse44 = nn.Sequential(
            nn.Conv2d(
                128 + 128 + 1,
                160,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                160,
            ),
            nn.GELU(),

            nn.Conv2d(
                160,
                160,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                160,
            ),
            nn.GELU(),
        )

        self.scalar2 = nn.Conv2d(
            160,
            1,
            kernel_size=1,
            bias=True,
        )

        self.channel2 = nn.Conv2d(
            160,
            128,
            kernel_size=1,
            bias=True,
        )

        self.up88 = nn.Sequential(
            nn.Conv2d(
                160,
                96,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                96,
            ),
            nn.GELU(),
        )

        self.fuse88 = nn.Sequential(
            nn.Conv2d(
                96 + 1,
                96,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                96,
            ),
            nn.GELU(),
            nn.Conv2d(
                96,
                96,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                96,
            ),
            nn.GELU(),
        )

        self.scalar1 = nn.Conv2d(
            96,
            1,
            kernel_size=1,
            bias=True,
        )

        self.channel1 = nn.Conv2d(
            96,
            64,
            kernel_size=1,
            bias=True,
        )

        for module in (
            self.x3_proj,
            self.x4_proj,
            self.fuse44,
            self.up88,
            self.fuse88,
        ):
            module.apply(
                init_conv_norm
            )

        # Exact R9 functional parity.
        for head in (
            self.scalar1,
            self.scalar2,
            self.channel1,
            self.channel2,
        ):
            nn.init.zeros_(
                head.weight
            )
            nn.init.zeros_(
                head.bias
            )

        # R13 keeps all R12 controller weights, but adds a learnable global
        # gain. gain(init)=1.0, so loading R12-E8 preserves its function.
        self.gain_max = 2.50
        self.gain_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    1.0,
                    self.gain_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_scale = 1.0
        self._last_stats = {}
        self.last_aux = {}

    @property
    def learned_gain(
        self,
    ):
        return (
            self.gain_max
            * torch.sigmoid(
                self.gain_logit
            )
        )

    def set_runtime_scale(
        self,
        scale,
    ):
        self.runtime_scale = float(
            scale
        )

    def forward(
        self,
        anchor_x3,
        anchor_x4,
        physical88,
        physical44,
    ):
        size44 = physical44.shape[-2:]
        size88 = physical88.shape[-2:]

        x3 = self.x3_proj(
            anchor_x3
        )

        x4 = self.x4_proj(
            anchor_x4
        )

        x4 = F.interpolate(
            x4,
            size=size44,
            mode="bilinear",
            align_corners=False,
        )

        if x3.shape[-2:] != size44:
            x3 = F.interpolate(
                x3,
                size=size44,
                mode="bilinear",
                align_corners=False,
            )

        h44 = self.fuse44(
            torch.cat(
                [
                    x3,
                    x4,
                    physical44,
                ],
                dim=1,
            )
        )

        scalar2 = self.scalar2(
            h44
        )

        channel2 = self.channel2(
            h44
        )

        mod2 = (
            float(
                self.runtime_scale
            )
            * self.learned_gain.to(
                dtype=scalar2.dtype
            )
            * (
                scalar2
                + 0.50
                * channel2
            )
        )

        h88 = F.interpolate(
            h44,
            size=size88,
            mode="bilinear",
            align_corners=False,
        )

        h88 = self.up88(
            h88
        )

        h88 = self.fuse88(
            torch.cat(
                [
                    h88,
                    physical88,
                ],
                dim=1,
            )
        )

        scalar1 = self.scalar1(
            h88
        )

        channel1 = self.channel1(
            h88
        )

        mod1 = (
            float(
                self.runtime_scale
            )
            * self.learned_gain.to(
                dtype=scalar1.dtype
            )
            * (
                scalar1
                + 0.50
                * channel1
            )
        )

        self.last_aux = {
            "controller_scalar1":
                scalar1,

            "controller_scalar2":
                scalar2,

            "controller_channel1":
                channel1,

            "controller_channel2":
                channel2,

            "controller_mod1":
                mod1,

            "controller_mod2":
                mod2,
        }

        with torch.no_grad():
            self._last_stats = {
                "scalar1_abs_mean":
                    scalar1.detach()
                    .abs()
                    .mean(),

                "scalar2_abs_mean":
                    scalar2.detach()
                    .abs()
                    .mean(),

                "mod1_abs_mean":
                    mod1.detach()
                    .abs()
                    .mean(),

                "mod2_abs_mean":
                    mod2.detach()
                    .abs()
                    .mean(),

                "learned_gain":
                    self.learned_gain.detach(),
            }

        return (
            mod1,
            mod2,
        )

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )

    def training_aux_tensors(
        self,
    ):
        return dict(
            self.last_aux
        )


# ============================================================================
# R12 balanced FP/FN high-resolution corrector
# ============================================================================

class BalancedErrorCorrector(nn.Module):
    """
    Separate recall (FN) and precision (FP) correction paths after the complete
    R9 HR prediction.

    Zero-initialized positive/negative output heads guarantee:
        P_R12(init) = P_R9

    Unlike R11's tiny signed residual, R12 uses two independently supervised
    correction magnitudes:
        positive branch -> recover missed foreground
        negative branch -> suppress false positives

    The branch is high-capacity but its inference inputs do NOT require GT or
    a teacher network.
    """

    def __init__(
        self,
        d1_dim=128,
        anchor_x3_dim=320,
        pos176_max=1.80,
        pos176_init=0.70,
        neg176_max=1.80,
        neg176_init=0.70,
        pos352_max=2.50,
        pos352_init=1.00,
        neg352_max=2.50,
        neg352_init=1.00,
    ):
        super().__init__()

        self.d1_proj = nn.Sequential(
            nn.Conv2d(
                d1_dim,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.anchor3_proj = nn.Sequential(
            nn.Conv2d(
                anchor_x3_dim,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.freq176 = nn.Sequential(
            nn.Conv2d(
                48,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),
        )

        self.fuse176 = nn.Sequential(
            nn.Conv2d(
                64 + 64 + 48 + 7,
                112,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                112,
            ),
            nn.GELU(),
            ResidualConvBlock(
                112,
                dilation=1,
            ),
            ResidualConvBlock(
                112,
                dilation=2,
            ),
            ResidualConvBlock(
                112,
                dilation=3,
            ),
            nn.Conv2d(
                112,
                72,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                72,
            ),
            nn.GELU(),
        )

        self.pos_gate176 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_gate176 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )
        self.pos_raw176 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_raw176 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )

        self.up352 = nn.Sequential(
            nn.Conv2d(
                72,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),
        )

        self.freq352 = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
        )

        self.rgb352 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.GELU(),
        )

        self.fuse352 = nn.Sequential(
            nn.Conv2d(
                48 + 32 + 16 + 7,
                88,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                88,
            ),
            nn.GELU(),
            ResidualConvBlock(
                88,
                dilation=1,
            ),
            ResidualConvBlock(
                88,
                dilation=2,
            ),
            ResidualConvBlock(
                88,
                dilation=3,
            ),
            nn.Conv2d(
                88,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.pos_gate352 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_gate352 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )
        self.pos_raw352 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_raw352 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )

        self.edge352 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )

        for module in (
            self.d1_proj,
            self.anchor3_proj,
            self.freq176,
            self.fuse176,
            self.up352,
            self.freq352,
            self.rgb352,
            self.fuse352,
        ):
            module.apply(
                init_conv_norm
            )

        # Gates are usable immediately, residual raw heads are exact zero.
        for gate_head in (
            self.pos_gate176,
            self.neg_gate176,
            self.pos_gate352,
            self.neg_gate352,
        ):
            nn.init.normal_(
                gate_head.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.constant_(
                gate_head.bias,
                -0.50,
            )

        for raw_head in (
            self.pos_raw176,
            self.neg_raw176,
            self.pos_raw352,
            self.neg_raw352,
        ):
            nn.init.zeros_(
                raw_head.weight
            )
            nn.init.zeros_(
                raw_head.bias
            )

        nn.init.normal_(
            self.edge352.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.edge352.bias
        )

        def make_scale(
            init_value,
            max_value,
        ):
            return nn.Parameter(
                torch.tensor(
                    inverse_bounded_sigmoid(
                        init_value,
                        max_value,
                    ),
                    dtype=torch.float32,
                )
            )

        self.pos176_max = float(
            pos176_max
        )
        self.neg176_max = float(
            neg176_max
        )
        self.pos352_max = float(
            pos352_max
        )
        self.neg352_max = float(
            neg352_max
        )

        self.pos176_logit = make_scale(
            pos176_init,
            pos176_max,
        )
        self.neg176_logit = make_scale(
            neg176_init,
            neg176_max,
        )
        self.pos352_logit = make_scale(
            pos352_init,
            pos352_max,
        )
        self.neg352_logit = make_scale(
            neg352_init,
            neg352_max,
        )

        self.runtime_scale = 1.0
        self._last_stats = {}
        self.last_aux = {}

    def _scale(
        self,
        logit,
        maximum,
    ):
        return (
            maximum
            * torch.sigmoid(
                logit
            )
        )

    @staticmethod
    def _uncertainty(
        logits,
    ):
        p = torch.sigmoid(
            logits.detach()
        )

        return torch.clamp(
            4.0
            * p
            * (
                1.0 - p
            ),
            0.0,
            1.0,
        )

    @staticmethod
    def _edge(
        logits,
    ):
        p = torch.sigmoid(
            logits.detach()
        )

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

    def set_runtime_scale(
        self,
        scale,
    ):
        self.runtime_scale = float(
            scale
        )

    def forward(
        self,
        base_logits,
        p176_logits,
        d1,
        anchor_x3,
        raw_rgb,
        freq_ctx,
        semantic_support88,
    ):
        size176 = freq_ctx[
            "f176"
        ].shape[-2:]

        size352 = raw_rgb.shape[-2:]

        base176 = F.interpolate(
            base_logits,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        d1_176 = self.d1_proj(
            F.interpolate(
                d1,
                size=size176,
                mode="bilinear",
                align_corners=False,
            )
        )

        a3_176 = self.anchor3_proj(
            F.interpolate(
                anchor_x3,
                size=size176,
                mode="bilinear",
                align_corners=False,
            )
        )

        freq176 = self.freq176(
            freq_ctx[
                "f176"
            ]
        )

        support176 = F.interpolate(
            semantic_support88,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty176 = self._uncertainty(
            base176
        )

        edge176 = self._edge(
            base176
        )

        disagreement176 = torch.abs(
            torch.sigmoid(
                base176.detach()
            )
            - torch.sigmoid(
                p176_logits.detach()
            )
        )

        h176 = self.fuse176(
            torch.cat(
                [
                    d1_176,
                    a3_176,
                    freq176,
                    base176,
                    support176,
                    uncertainty176,
                    edge176,
                    freq_ctx[
                        "physical176"
                    ],
                    freq_ctx[
                        "transition176"
                    ],
                    disagreement176,
                ],
                dim=1,
            )
        )

        pos_gate176 = torch.sigmoid(
            self.pos_gate176(
                h176
            )
        )

        neg_gate176 = torch.sigmoid(
            self.neg_gate176(
                h176
            )
        )

        pos_raw176 = self.pos_raw176(
            h176
        )

        neg_raw176 = self.neg_raw176(
            h176
        )

        pos_mag176 = torch.tanh(
            pos_raw176
        )

        neg_mag176 = torch.tanh(
            neg_raw176
        )

        pos176 = (
            self._scale(
                self.pos176_logit,
                self.pos176_max,
            ).to(
                dtype=base_logits.dtype
            )
            * pos_gate176
            * pos_mag176
        )

        neg176 = (
            self._scale(
                self.neg176_logit,
                self.neg176_max,
            ).to(
                dtype=base_logits.dtype
            )
            * neg_gate176
            * neg_mag176
        )

        correction176 = (
            float(
                self.runtime_scale
            )
            * (
                pos176
                - neg176
            )
        )

        correction176_up = F.interpolate(
            correction176,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        pre352 = (
            base_logits
            + correction176_up
        )

        h176_up = self.up352(
            F.interpolate(
                h176,
                size=size352,
                mode="bilinear",
                align_corners=False,
            )
        )

        freq352 = self.freq352(
            freq_ctx[
                "f352"
            ]
        )

        rgb352 = self.rgb352(
            raw_rgb
        )

        support352 = F.interpolate(
            semantic_support88,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty352 = self._uncertainty(
            pre352
        )

        edge352 = self._edge(
            pre352
        )

        p176_up = F.interpolate(
            p176_logits,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        disagreement352 = torch.abs(
            torch.sigmoid(
                pre352.detach()
            )
            - torch.sigmoid(
                p176_up.detach()
            )
        )

        h352 = self.fuse352(
            torch.cat(
                [
                    h176_up,
                    freq352,
                    rgb352,
                    pre352,
                    support352,
                    uncertainty352,
                    edge352,
                    freq_ctx[
                        "physical352"
                    ],
                    freq_ctx[
                        "transition352"
                    ],
                    disagreement352,
                ],
                dim=1,
            )
        )

        pos_gate352 = torch.sigmoid(
            self.pos_gate352(
                h352
            )
        )

        neg_gate352 = torch.sigmoid(
            self.neg_gate352(
                h352
            )
        )

        pos_raw352 = self.pos_raw352(
            h352
        )

        neg_raw352 = self.neg_raw352(
            h352
        )

        pos_mag352 = torch.tanh(
            pos_raw352
        )

        neg_mag352 = torch.tanh(
            neg_raw352
        )

        pos352 = (
            self._scale(
                self.pos352_logit,
                self.pos352_max,
            ).to(
                dtype=base_logits.dtype
            )
            * pos_gate352
            * pos_mag352
        )

        neg352 = (
            self._scale(
                self.neg352_logit,
                self.neg352_max,
            ).to(
                dtype=base_logits.dtype
            )
            * neg_gate352
            * neg_mag352
        )

        correction352 = (
            float(
                self.runtime_scale
            )
            * (
                pos352
                - neg352
            )
        )

        final_logits = (
            pre352
            + correction352
        )

        total_correction = (
            correction176_up
            + correction352
        )

        self.last_aux = {
            "bec_pos_gate176":
                pos_gate176,

            "bec_neg_gate176":
                neg_gate176,

            "bec_pos_raw176":
                pos_raw176,

            "bec_neg_raw176":
                neg_raw176,

            "bec_pos_gate352":
                pos_gate352,

            "bec_neg_gate352":
                neg_gate352,

            "bec_pos_raw352":
                pos_raw352,

            "bec_neg_raw352":
                neg_raw352,

            "bec_correction176":
                correction176,

            "bec_correction352":
                correction352,

            "bec_total_correction":
                total_correction,

            "bec_edge_logits352":
                self.edge352(
                    h352
                ),
        }

        with torch.no_grad():
            self._last_stats = {
                "pos_gate176_mean":
                    pos_gate176.detach()
                    .mean(),

                "neg_gate176_mean":
                    neg_gate176.detach()
                    .mean(),

                "pos_gate352_mean":
                    pos_gate352.detach()
                    .mean(),

                "neg_gate352_mean":
                    neg_gate352.detach()
                    .mean(),

                "correction176_abs_mean":
                    correction176.detach()
                    .abs()
                    .mean(),

                "correction352_abs_mean":
                    correction352.detach()
                    .abs()
                    .mean(),

                "total_correction_abs_mean":
                    total_correction.detach()
                    .abs()
                    .mean(),
            }

        return final_logits

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )

    def training_aux_tensors(
        self,
    ):
        return dict(
            self.last_aux
        )


# ============================================================================
# R13 hard-region decision experts
# ============================================================================

class HardRegionDecisionCorrector(nn.Module):
    """
    R13 replacement for R12 BalancedErrorCorrector.

    Why this differs:
    - no starvation-prone gate multiplication;
    - route masks have a fixed forward floor;
    - positive and negative experts have independent trunks;
    - actual positive / negative logit corrections are directly supervised;
    - zero raw-output initialization still preserves the R12-E8 base exactly.

    The corrector is intentionally strong enough to move decision scores by
    O(1) logits on hard pixels, unlike the ~1e-3 mean corrections observed in
    R12.
    """

    def __init__(
        self,
        d1_dim=128,
        anchor_x3_dim=320,
        route_floor=0.25,
        alpha176_max=2.50,
        alpha176_init=1.00,
        alpha352_max=3.50,
        alpha352_init=1.50,
    ):
        super().__init__()

        self.route_floor = float(
            route_floor
        )

        self.d1_proj = nn.Sequential(
            nn.Conv2d(
                d1_dim,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.anchor3_proj = nn.Sequential(
            nn.Conv2d(
                anchor_x3_dim,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.freq176 = nn.Sequential(
            nn.Conv2d(
                48,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),
        )

        self.shared176 = nn.Sequential(
            nn.Conv2d(
                64 + 64 + 48 + 7,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),
            ResidualConvBlock(
                128,
                dilation=1,
            ),
            ResidualConvBlock(
                128,
                dilation=2,
            ),
        )

        self.pos176_expert = nn.Sequential(
            ResidualConvBlock(
                128,
                dilation=1,
            ),
            ResidualConvBlock(
                128,
                dilation=3,
            ),
            nn.Conv2d(
                128,
                80,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                80,
            ),
            nn.GELU(),
        )

        self.neg176_expert = nn.Sequential(
            ResidualConvBlock(
                128,
                dilation=1,
            ),
            ResidualConvBlock(
                128,
                dilation=3,
            ),
            nn.Conv2d(
                128,
                80,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                80,
            ),
            nn.GELU(),
        )

        self.pos_route176 = nn.Conv2d(
            80,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_route176 = nn.Conv2d(
            80,
            1,
            kernel_size=1,
            bias=True,
        )

        self.pos_raw176 = nn.Conv2d(
            80,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_raw176 = nn.Conv2d(
            80,
            1,
            kernel_size=1,
            bias=True,
        )

        self.up352 = nn.Sequential(
            nn.Conv2d(
                128,
                56,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                56,
            ),
            nn.GELU(),
        )

        self.freq352 = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
        )

        self.rgb352 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.GELU(),
        )

        self.shared352 = nn.Sequential(
            nn.Conv2d(
                56 + 32 + 16 + 7,
                112,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                112,
            ),
            nn.GELU(),
            ResidualConvBlock(
                112,
                dilation=1,
            ),
            ResidualConvBlock(
                112,
                dilation=2,
            ),
        )

        self.pos352_expert = nn.Sequential(
            ResidualConvBlock(
                112,
                dilation=1,
            ),
            ResidualConvBlock(
                112,
                dilation=3,
            ),
            nn.Conv2d(
                112,
                72,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                72,
            ),
            nn.GELU(),
        )

        self.neg352_expert = nn.Sequential(
            ResidualConvBlock(
                112,
                dilation=1,
            ),
            ResidualConvBlock(
                112,
                dilation=3,
            ),
            nn.Conv2d(
                112,
                72,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                72,
            ),
            nn.GELU(),
        )

        self.pos_route352 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_route352 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )

        self.pos_raw352 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )
        self.neg_raw352 = nn.Conv2d(
            72,
            1,
            kernel_size=1,
            bias=True,
        )

        self.edge352 = nn.Conv2d(
            112,
            1,
            kernel_size=1,
            bias=True,
        )

        for module in (
            self.d1_proj,
            self.anchor3_proj,
            self.freq176,
            self.shared176,
            self.pos176_expert,
            self.neg176_expert,
            self.up352,
            self.freq352,
            self.rgb352,
            self.shared352,
            self.pos352_expert,
            self.neg352_expert,
        ):
            module.apply(
                init_conv_norm
            )

        # Routes start moderately open. Raw heads are EXACT zero so the whole
        # corrector initially contributes zero logits while still having a
        # nonzero derivative through tanh(raw).
        for head in (
            self.pos_route176,
            self.neg_route176,
            self.pos_route352,
            self.neg_route352,
        ):
            nn.init.normal_(
                head.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.constant_(
                head.bias,
                -0.85,
            )

        for head in (
            self.pos_raw176,
            self.neg_raw176,
            self.pos_raw352,
            self.neg_raw352,
        ):
            nn.init.zeros_(
                head.weight
            )
            nn.init.zeros_(
                head.bias
            )

        nn.init.normal_(
            self.edge352.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.edge352.bias
        )

        self.alpha176_max = float(
            alpha176_max
        )
        self.alpha352_max = float(
            alpha352_max
        )

        self.alpha176_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    alpha176_init,
                    alpha176_max,
                ),
                dtype=torch.float32,
            )
        )

        self.alpha352_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    alpha352_init,
                    alpha352_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_scale = 1.0
        self.last_aux = {}
        self._last_stats = {}

    @property
    def alpha176(
        self,
    ):
        return (
            self.alpha176_max
            * torch.sigmoid(
                self.alpha176_logit
            )
        )

    @property
    def alpha352(
        self,
    ):
        return (
            self.alpha352_max
            * torch.sigmoid(
                self.alpha352_logit
            )
        )

    def set_runtime_scale(
        self,
        scale,
    ):
        self.runtime_scale = float(
            scale
        )

    @staticmethod
    def _uncertainty(
        logits,
    ):
        p = torch.sigmoid(
            logits.detach()
        )
        return torch.clamp(
            4.0
            * p
            * (
                1.0 - p
            ),
            0.0,
            1.0,
        )

    @staticmethod
    def _prediction_edge(
        logits,
    ):
        p = torch.sigmoid(
            logits.detach()
        )

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

    def _route(
        self,
        logits,
    ):
        return (
            self.route_floor
            + (
                1.0
                - self.route_floor
            )
            * torch.sigmoid(
                logits
            )
        )

    def forward(
        self,
        base_logits,
        p176_logits,
        d1,
        anchor_x3,
        raw_rgb,
        freq_ctx,
        semantic_support88,
    ):
        size176 = freq_ctx[
            "f176"
        ].shape[-2:]
        size352 = raw_rgb.shape[-2:]

        base176 = F.interpolate(
            base_logits,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        d1_176 = self.d1_proj(
            F.interpolate(
                d1,
                size=size176,
                mode="bilinear",
                align_corners=False,
            )
        )

        anchor176 = self.anchor3_proj(
            F.interpolate(
                anchor_x3,
                size=size176,
                mode="bilinear",
                align_corners=False,
            )
        )

        freq176 = self.freq176(
            freq_ctx[
                "f176"
            ]
        )

        support176 = F.interpolate(
            semantic_support88,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty176 = self._uncertainty(
            base176
        )

        edge176 = self._prediction_edge(
            base176
        )

        disagreement176 = torch.abs(
            torch.sigmoid(
                base176.detach()
            )
            - torch.sigmoid(
                p176_logits.detach()
            )
        )

        shared176 = self.shared176(
            torch.cat(
                [
                    d1_176,
                    anchor176,
                    freq176,
                    base176,
                    support176,
                    uncertainty176,
                    edge176,
                    freq_ctx[
                        "physical176"
                    ],
                    freq_ctx[
                        "transition176"
                    ],
                    disagreement176,
                ],
                dim=1,
            )
        )

        pos176_feat = self.pos176_expert(
            shared176
        )
        neg176_feat = self.neg176_expert(
            shared176
        )

        pos_route176_logits = self.pos_route176(
            pos176_feat
        )
        neg_route176_logits = self.neg_route176(
            neg176_feat
        )

        pos_route176 = self._route(
            pos_route176_logits
        )
        neg_route176 = self._route(
            neg_route176_logits
        )

        pos_raw176 = self.pos_raw176(
            pos176_feat
        )
        neg_raw176 = self.neg_raw176(
            neg176_feat
        )

        pos_correction176 = (
            float(
                self.runtime_scale
            )
            * self.alpha176.to(
                dtype=base_logits.dtype
            )
            * pos_route176
            * torch.tanh(
                pos_raw176
            )
        )

        neg_correction176 = (
            float(
                self.runtime_scale
            )
            * self.alpha176.to(
                dtype=base_logits.dtype
            )
            * neg_route176
            * torch.tanh(
                neg_raw176
            )
        )

        correction176 = (
            pos_correction176
            - neg_correction176
        )

        correction176_up = F.interpolate(
            correction176,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        pre352 = (
            base_logits
            + correction176_up
        )

        shared176_up = self.up352(
            F.interpolate(
                shared176,
                size=size352,
                mode="bilinear",
                align_corners=False,
            )
        )

        freq352 = self.freq352(
            freq_ctx[
                "f352"
            ]
        )

        rgb352 = self.rgb352(
            raw_rgb
        )

        support352 = F.interpolate(
            semantic_support88,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty352 = self._uncertainty(
            pre352
        )

        edge352 = self._prediction_edge(
            pre352
        )

        p176_up = F.interpolate(
            p176_logits,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        disagreement352 = torch.abs(
            torch.sigmoid(
                pre352.detach()
            )
            - torch.sigmoid(
                p176_up.detach()
            )
        )

        shared352 = self.shared352(
            torch.cat(
                [
                    shared176_up,
                    freq352,
                    rgb352,
                    pre352,
                    support352,
                    uncertainty352,
                    edge352,
                    freq_ctx[
                        "physical352"
                    ],
                    freq_ctx[
                        "transition352"
                    ],
                    disagreement352,
                ],
                dim=1,
            )
        )

        pos352_feat = self.pos352_expert(
            shared352
        )
        neg352_feat = self.neg352_expert(
            shared352
        )

        pos_route352_logits = self.pos_route352(
            pos352_feat
        )
        neg_route352_logits = self.neg_route352(
            neg352_feat
        )

        pos_route352 = self._route(
            pos_route352_logits
        )
        neg_route352 = self._route(
            neg_route352_logits
        )

        pos_raw352 = self.pos_raw352(
            pos352_feat
        )
        neg_raw352 = self.neg_raw352(
            neg352_feat
        )

        pos_correction352 = (
            float(
                self.runtime_scale
            )
            * self.alpha352.to(
                dtype=base_logits.dtype
            )
            * pos_route352
            * torch.tanh(
                pos_raw352
            )
        )

        neg_correction352 = (
            float(
                self.runtime_scale
            )
            * self.alpha352.to(
                dtype=base_logits.dtype
            )
            * neg_route352
            * torch.tanh(
                neg_raw352
            )
        )

        correction352 = (
            pos_correction352
            - neg_correction352
        )

        final_logits = (
            pre352
            + correction352
        )

        total_correction = (
            correction176_up
            + correction352
        )

        self.last_aux = {
            "r13_pos_route176_logits":
                pos_route176_logits,

            "r13_neg_route176_logits":
                neg_route176_logits,

            "r13_pos_route176":
                pos_route176,

            "r13_neg_route176":
                neg_route176,

            "r13_pos_raw176":
                pos_raw176,

            "r13_neg_raw176":
                neg_raw176,

            "r13_pos_correction176":
                pos_correction176,

            "r13_neg_correction176":
                neg_correction176,

            "r13_pos_route352_logits":
                pos_route352_logits,

            "r13_neg_route352_logits":
                neg_route352_logits,

            "r13_pos_route352":
                pos_route352,

            "r13_neg_route352":
                neg_route352,

            "r13_pos_raw352":
                pos_raw352,

            "r13_neg_raw352":
                neg_raw352,

            "r13_pos_correction352":
                pos_correction352,

            "r13_neg_correction352":
                neg_correction352,

            "r13_total_correction":
                total_correction,

            "r13_edge_logits352":
                self.edge352(
                    shared352
                ),
        }

        with torch.no_grad():
            self._last_stats = {
                "alpha176":
                    self.alpha176.detach(),

                "alpha352":
                    self.alpha352.detach(),

                "pos_route176_mean":
                    pos_route176.detach()
                    .mean(),

                "neg_route176_mean":
                    neg_route176.detach()
                    .mean(),

                "pos_route352_mean":
                    pos_route352.detach()
                    .mean(),

                "neg_route352_mean":
                    neg_route352.detach()
                    .mean(),

                "pos_correction176_abs_mean":
                    pos_correction176.detach()
                    .abs()
                    .mean(),

                "neg_correction176_abs_mean":
                    neg_correction176.detach()
                    .abs()
                    .mean(),

                "pos_correction352_abs_mean":
                    pos_correction352.detach()
                    .abs()
                    .mean(),

                "neg_correction352_abs_mean":
                    neg_correction352.detach()
                    .abs()
                    .mean(),

                "total_correction_abs_mean":
                    total_correction.detach()
                    .abs()
                    .mean(),
            }

        return final_logits

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )

    def training_aux_tensors(
        self,
    ):
        return dict(
            self.last_aux
        )


# ============================================================================
# R14 tri-route selective calibration head
# ============================================================================

class TriRouteSelectiveCorrector(nn.Module):
    """
    R14 replacement for the R13 decision experts.

    Core idea:
        a single mutually-exclusive 3-way router predicts
            KEEP / FN-RAISE / FP-LOWER
        instead of two independent routes that can collapse together.

    Important properties:
        - no artificial route floor;
        - magnitude heads are supervised directly, so they do not starve even
          when the router is initially conservative;
        - the final correction is zero at initialization;
        - only genuinely uncertain / wrong teacher pixels are correction
          targets; already-confident teacher pixels are KEEP.

    The head uses a 176x176 semantic-frequency context trunk and makes its final
    routing/correction decision at 352x352.
    """

    def __init__(
        self,
        d1_dim=128,
        anchor_x3_dim=320,
        alpha_pos_max=2.50,
        alpha_pos_init=1.30,
        alpha_neg_max=2.50,
        alpha_neg_init=1.30,
        route_temperature=0.85,
    ):
        super().__init__()

        self.route_temperature = float(
            route_temperature
        )

        self.d1_proj = nn.Sequential(
            nn.Conv2d(
                d1_dim,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.anchor3_proj = nn.Sequential(
            nn.Conv2d(
                anchor_x3_dim,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.freq176 = nn.Sequential(
            nn.Conv2d(
                48,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),
        )

        # Seven scalar maps:
        # base, support, uncertainty, edge, physical, transition, disagreement.
        self.context176 = nn.Sequential(
            nn.Conv2d(
                64 + 64 + 48 + 7,
                112,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                112,
            ),
            nn.GELU(),
            ResidualConvBlock(
                112,
                dilation=1,
            ),
            ResidualConvBlock(
                112,
                dilation=2,
            ),
            ResidualConvBlock(
                112,
                dilation=3,
            ),
            nn.Conv2d(
                112,
                72,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                72,
            ),
            nn.GELU(),
        )

        self.up352 = nn.Sequential(
            nn.Conv2d(
                72,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.GELU(),
        )

        self.freq352 = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
        )

        self.rgb352 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.GELU(),
        )

        self.context352 = nn.Sequential(
            nn.Conv2d(
                48 + 32 + 16 + 7,
                96,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                96,
            ),
            nn.GELU(),
            ResidualConvBlock(
                96,
                dilation=1,
            ),
            ResidualConvBlock(
                96,
                dilation=2,
            ),
            ResidualConvBlock(
                96,
                dilation=3,
            ),
        )

        self.route_head = nn.Sequential(
            nn.Conv2d(
                96,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
            nn.Conv2d(
                64,
                3,
                kernel_size=1,
                bias=True,
            ),
        )

        self.pos_expert = nn.Sequential(
            nn.Conv2d(
                96,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
            ResidualConvBlock(
                64,
                dilation=2,
            ),
        )

        self.neg_expert = nn.Sequential(
            nn.Conv2d(
                96,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
            ResidualConvBlock(
                64,
                dilation=2,
            ),
        )

        self.pos_raw = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )

        self.neg_raw = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            bias=True,
        )

        self.edge_head = nn.Conv2d(
            96,
            1,
            kernel_size=1,
            bias=True,
        )

        for module in (
            self.d1_proj,
            self.anchor3_proj,
            self.freq176,
            self.context176,
            self.up352,
            self.freq352,
            self.rgb352,
            self.context352,
            self.pos_expert,
            self.neg_expert,
        ):
            module.apply(
                init_conv_norm
            )

        # Conservative KEEP-biased routing at initialization.
        final_route = self.route_head[
            -1
        ]

        nn.init.zeros_(
            final_route.weight
        )

        with torch.no_grad():
            final_route.bias.copy_(
                torch.tensor(
                    [
                        2.0,   # KEEP
                        -1.0,  # FN-RAISE
                        -1.0,  # FP-LOWER
                    ],
                    dtype=final_route.bias.dtype,
                )
            )

        # Exact-zero correction at initialization, with nonzero derivative.
        nn.init.zeros_(
            self.pos_raw.weight
        )
        nn.init.zeros_(
            self.pos_raw.bias
        )
        nn.init.zeros_(
            self.neg_raw.weight
        )
        nn.init.zeros_(
            self.neg_raw.bias
        )

        nn.init.normal_(
            self.edge_head.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.edge_head.bias
        )

        self.alpha_pos_max = float(
            alpha_pos_max
        )

        self.alpha_neg_max = float(
            alpha_neg_max
        )

        self.alpha_pos_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    alpha_pos_init,
                    alpha_pos_max,
                ),
                dtype=torch.float32,
            )
        )

        self.alpha_neg_logit = nn.Parameter(
            torch.tensor(
                inverse_bounded_sigmoid(
                    alpha_neg_init,
                    alpha_neg_max,
                ),
                dtype=torch.float32,
            )
        )

        self.runtime_scale = 1.0
        self.last_aux = {}
        self._last_stats = {}

    @property
    def alpha_pos(
        self,
    ):
        return (
            self.alpha_pos_max
            * torch.sigmoid(
                self.alpha_pos_logit
            )
        )

    @property
    def alpha_neg(
        self,
    ):
        return (
            self.alpha_neg_max
            * torch.sigmoid(
                self.alpha_neg_logit
            )
        )

    def set_runtime_scale(
        self,
        scale,
    ):
        self.runtime_scale = float(
            scale
        )

    @staticmethod
    def _uncertainty(
        logits,
    ):
        p = torch.sigmoid(
            logits.detach()
        )

        return torch.clamp(
            4.0
            * p
            * (
                1.0 - p
            ),
            0.0,
            1.0,
        )

    @staticmethod
    def _prediction_edge(
        logits,
    ):
        p = torch.sigmoid(
            logits.detach()
        )

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
        base_logits,
        p176_logits,
        d1,
        anchor_x3,
        raw_rgb,
        freq_ctx,
        semantic_support88,
    ):
        size176 = freq_ctx[
            "f176"
        ].shape[-2:]

        size352 = raw_rgb.shape[-2:]

        base176 = F.interpolate(
            base_logits,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        d1_176 = self.d1_proj(
            F.interpolate(
                d1,
                size=size176,
                mode="bilinear",
                align_corners=False,
            )
        )

        anchor176 = self.anchor3_proj(
            F.interpolate(
                anchor_x3,
                size=size176,
                mode="bilinear",
                align_corners=False,
            )
        )

        freq176 = self.freq176(
            freq_ctx[
                "f176"
            ]
        )

        support176 = F.interpolate(
            semantic_support88,
            size=size176,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty176 = self._uncertainty(
            base176
        )

        edge176 = self._prediction_edge(
            base176
        )

        disagreement176 = torch.abs(
            torch.sigmoid(
                base176.detach()
            )
            - torch.sigmoid(
                p176_logits.detach()
            )
        )

        h176 = self.context176(
            torch.cat(
                [
                    d1_176,
                    anchor176,
                    freq176,
                    base176,
                    support176,
                    uncertainty176,
                    edge176,
                    freq_ctx[
                        "physical176"
                    ],
                    freq_ctx[
                        "transition176"
                    ],
                    disagreement176,
                ],
                dim=1,
            )
        )

        h176_up = self.up352(
            F.interpolate(
                h176,
                size=size352,
                mode="bilinear",
                align_corners=False,
            )
        )

        freq352 = self.freq352(
            freq_ctx[
                "f352"
            ]
        )

        rgb352 = self.rgb352(
            raw_rgb
        )

        support352 = F.interpolate(
            semantic_support88,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        uncertainty352 = self._uncertainty(
            base_logits
        )

        edge352 = self._prediction_edge(
            base_logits
        )

        p176_up = F.interpolate(
            p176_logits,
            size=size352,
            mode="bilinear",
            align_corners=False,
        )

        disagreement352 = torch.abs(
            torch.sigmoid(
                base_logits.detach()
            )
            - torch.sigmoid(
                p176_up.detach()
            )
        )

        h352 = self.context352(
            torch.cat(
                [
                    h176_up,
                    freq352,
                    rgb352,
                    base_logits,
                    support352,
                    uncertainty352,
                    edge352,
                    freq_ctx[
                        "physical352"
                    ],
                    freq_ctx[
                        "transition352"
                    ],
                    disagreement352,
                ],
                dim=1,
            )
        )

        route_logits = self.route_head(
            h352
        )

        route_prob = torch.softmax(
            route_logits
            / self.route_temperature,
            dim=1,
        )

        keep_prob = route_prob[
            :,
            0:1,
        ]

        fn_prob = route_prob[
            :,
            1:2,
        ]

        fp_prob = route_prob[
            :,
            2:3,
        ]

        pos_raw = self.pos_raw(
            self.pos_expert(
                h352
            )
        )

        neg_raw = self.neg_raw(
            self.neg_expert(
                h352
            )
        )

        # Magnitude heads are free to learn independently from routing.
        # Their sign is handled by the final +/- composition and supervised.
        pos_delta = (
            self.alpha_pos.to(
                dtype=base_logits.dtype
            )
            * torch.tanh(
                pos_raw
            )
        )

        neg_delta = (
            self.alpha_neg.to(
                dtype=base_logits.dtype
            )
            * torch.tanh(
                neg_raw
            )
        )

        correction = (
            float(
                self.runtime_scale
            )
            * (
                fn_prob
                * pos_delta
                - fp_prob
                * neg_delta
            )
        )

        final_logits = (
            base_logits
            + correction
        )

        self.last_aux = {
            "r14_route_logits":
                route_logits,

            "r14_keep_prob":
                keep_prob,

            "r14_fn_prob":
                fn_prob,

            "r14_fp_prob":
                fp_prob,

            "r14_pos_raw":
                pos_raw,

            "r14_neg_raw":
                neg_raw,

            "r14_pos_delta":
                pos_delta,

            "r14_neg_delta":
                neg_delta,

            "r14_correction":
                correction,

            "r14_edge_logits":
                self.edge_head(
                    h352
                ),
        }

        with torch.no_grad():
            self._last_stats = {
                "alpha_pos":
                    self.alpha_pos.detach(),

                "alpha_neg":
                    self.alpha_neg.detach(),

                "keep_mean":
                    keep_prob.detach()
                    .mean(),

                "fn_mean":
                    fn_prob.detach()
                    .mean(),

                "fp_mean":
                    fp_prob.detach()
                    .mean(),

                "pos_delta_abs_mean":
                    pos_delta.detach()
                    .abs()
                    .mean(),

                "neg_delta_abs_mean":
                    neg_delta.detach()
                    .abs()
                    .mean(),

                "correction_abs_mean":
                    correction.detach()
                    .abs()
                    .mean(),
            }

        return final_logits

    def runtime_stats(
        self,
    ):
        return dict(
            self._last_stats
        )

    def training_aux_tensors(
        self,
    ):
        return dict(
            self.last_aux
        )
