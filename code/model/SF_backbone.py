# -*- coding: utf-8 -*-
"""
Reset-R15 region-Mamba semantic-feedback Frequency-Aware PVTv2 wrapper.

The wrapper reuses the original pvt_v2_b2 stage modules directly, preserving
their state_dict names:
    patch_embed1, block1, norm1, ... patch_embed4, block4, norm4

Therefore old THMNet checkpoints with keys such as:
    encoder.patch_embed1.*
still load directly.

New R9 modules are:
    freq_pyramid
    stage1_adapter
    stage2_adapter
"""

import torch
import torch.nn as nn

from model.SF_modules import (
    RGBFrequencyPyramid,
    FrequencyStageAdapter,
    SemanticFeedbackController,
)


class FrequencyAwarePVTv2(nn.Module):
    """
    PVTv2-B2 with frequency-aware Stage1/Stage2 residual adapters.

    Standard PVTv2 stage resolutions for 352 input:
        Stage1 64 x 88 x 88
        Stage2 128 x 44 x 44
        Stage3 320 x 22 x 22
        Stage4 512 x 11 x 11

    The independent RGB-frequency pyramid is fused into Stage1 and Stage2
    BEFORE later PVT stages, allowing Stage3/Stage4 semantics to build on
    frequency-aware shallow representations.
    """

    REQUIRED_ATTRIBUTES = (
        "patch_embed1",
        "block1",
        "norm1",
        "patch_embed2",
        "block2",
        "norm2",
        "patch_embed3",
        "block3",
        "norm3",
        "patch_embed4",
        "block4",
        "norm4",
    )

    def __init__(
        self,
        backbone,
    ):
        super().__init__()

        missing = [
            name
            for name in self.REQUIRED_ATTRIBUTES
            if not hasattr(
                backbone,
                name,
            )
        ]

        if missing:
            raise AttributeError(
                "The local pvt_v2_b2 implementation is missing required "
                "PVTv2 attributes: {}".format(
                    missing
                )
            )

        # Re-register original stage modules directly to preserve checkpoint
        # names under encoder.<stage_name>.
        for name in self.REQUIRED_ATTRIBUTES:
            setattr(
                self,
                name,
                getattr(
                    backbone,
                    name,
                ),
            )

        self.freq_pyramid = (
            RGBFrequencyPyramid()
        )

        self.stage1_adapter = (
            FrequencyStageAdapter(
                dim=64,
                freq_dim=64,
                hidden=64,
                alpha_max=0.80,
                alpha_init=0.18,
                gate_bias=-1.50,
            )
        )

        self.stage2_adapter = (
            FrequencyStageAdapter(
                dim=128,
                freq_dim=128,
                hidden=128,
                alpha_max=0.80,
                alpha_init=0.20,
                gate_bias=-1.35,
            )
        )

        # R12 reads deep semantics and controls shallow frequency usage.
        # The controller is exactly neutral at initialization.
        self.semantic_controller = (
            SemanticFeedbackController()
        )

        self.runtime_frequency_scale = 1.0
        self.runtime_stage1_scale = 1.0
        self.runtime_stage2_scale = 1.0
        self.runtime_controller_scale = 1.0

        self._last_stats = {}
        self.last_aux = {}

    def set_runtime_scales(
        self,
        frequency_scale=1.0,
        stage1_scale=1.0,
        stage2_scale=1.0,
        controller_scale=1.0,
    ):
        self.runtime_frequency_scale = float(
            frequency_scale
        )

        self.runtime_stage1_scale = float(
            stage1_scale
        )

        self.runtime_stage2_scale = float(
            stage2_scale
        )

        self.runtime_controller_scale = float(
            controller_scale
        )

        self.stage1_adapter.set_runtime_scale(
            stage1_scale
        )

        self.stage2_adapter.set_runtime_scale(
            stage2_scale
        )

        self.semantic_controller.set_runtime_scale(
            controller_scale
        )

    @staticmethod
    def _tokens_to_map(
        x,
        batch,
        height,
        width,
    ):
        return (
            x.reshape(
                batch,
                height,
                width,
                -1,
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

    def _run_stage(
        self,
        feature_map,
        patch_embed,
        blocks,
        norm,
    ):
        batch = feature_map.shape[0]

        tokens, height, width = patch_embed(
            feature_map
        )

        for block in blocks:
            tokens = block(
                tokens,
                height,
                width,
            )

        tokens = norm(
            tokens
        )

        return self._tokens_to_map(
            tokens,
            batch,
            height,
            width,
        )

    def forward(
        self,
        rgb,
        feedback_context=None,
    ):
        freq_ctx = self.freq_pyramid(
            rgb
        )

        # Runtime frequency ablation preserves the same module graph while
        # structurally zeroing all new RGB-frequency inputs.
        if abs(
            float(
                self.runtime_frequency_scale
            )
        ) < 1e-12:
            freq_ctx = {
                key:
                    (
                        torch.zeros_like(
                            value
                        )
                        if torch.is_tensor(
                            value
                        )
                        else value
                    )
                for key, value
                in freq_ctx.items()
            }

        elif abs(
            float(
                self.runtime_frequency_scale
            )
            - 1.0
        ) > 1e-12:
            scale = float(
                self.runtime_frequency_scale
            )

            scaled = {}

            for key, value in freq_ctx.items():
                if torch.is_tensor(
                    value
                ):
                    scaled[
                        key
                    ] = (
                        value
                        * scale
                    )
                else:
                    scaled[
                        key
                    ] = value

            freq_ctx = scaled

        if feedback_context is None:
            mod1 = None
            mod2 = None
            controller_aux = {}
            controller_stats = {}
        else:
            if (
                "x3"
                not in feedback_context
                or "x4"
                not in feedback_context
            ):
                raise KeyError(
                    "feedback_context must contain x3 and x4"
                )

            mod1, mod2 = self.semantic_controller(
                anchor_x3=feedback_context[
                    "x3"
                ],
                anchor_x4=feedback_context[
                    "x4"
                ],
                physical88=freq_ctx[
                    "physical88"
                ],
                physical44=freq_ctx[
                    "physical44"
                ],
            )

            controller_aux = (
                self.semantic_controller
                .training_aux_tensors()
            )

            controller_stats = (
                self.semantic_controller
                .runtime_stats()
            )

        # Stage 1: standard PVT feature then frequency-aware residual adapter.
        x1 = self._run_stage(
            rgb,
            self.patch_embed1,
            self.block1,
            self.norm1,
        )

        x1, gate1 = self.stage1_adapter(
            x1,
            freq_ctx[
                "f88"
            ],
            external_mod=mod1,
            external_strength=0.80,
        )

        # Stage 2 receives the already adapted Stage1 representation.
        x2 = self._run_stage(
            x1,
            self.patch_embed2,
            self.block2,
            self.norm2,
        )

        x2, gate2 = self.stage2_adapter(
            x2,
            freq_ctx[
                "f44"
            ],
            external_mod=mod2,
            external_strength=0.80,
        )

        # Deeper PVT stages are standard, but they now consume the adapted
        # Stage2 feature and therefore can build frequency-aware semantics.
        x3 = self._run_stage(
            x2,
            self.patch_embed3,
            self.block3,
            self.norm3,
        )

        x4 = self._run_stage(
            x3,
            self.patch_embed4,
            self.block4,
            self.norm4,
        )

        stats1 = self.stage1_adapter.runtime_stats()
        stats2 = self.stage2_adapter.runtime_stats()

        self.last_aux = {
            **controller_aux,

            "freq_edge_logits352":
                freq_ctx[
                    "edge_logits352"
                ],

            "freq_edge_logits176":
                freq_ctx[
                    "edge_logits176"
                ],

            "freq_physical352":
                freq_ctx[
                    "physical352"
                ].detach(),

            "freq_physical176":
                freq_ctx[
                    "physical176"
                ].detach(),

            "freq_physical88":
                freq_ctx[
                    "physical88"
                ].detach(),

            "freq_physical44":
                freq_ctx[
                    "physical44"
                ].detach(),
        }

        with torch.no_grad():
            self._last_stats = {
                "stage1_alpha":
                    stats1.get(
                        "alpha"
                    ),

                "stage1_gate_mean":
                    stats1.get(
                        "gate_mean"
                    ),

                "stage1_applied_ratio":
                    stats1.get(
                        "applied_ratio"
                    ),

                "stage2_alpha":
                    stats2.get(
                        "alpha"
                    ),

                "stage2_gate_mean":
                    stats2.get(
                        "gate_mean"
                    ),

                "stage2_applied_ratio":
                    stats2.get(
                        "applied_ratio"
                    ),

                "stage1_modulation_mean":
                    stats1.get(
                        "modulation_mean"
                    ),

                "stage2_modulation_mean":
                    stats2.get(
                        "modulation_mean"
                    ),

                "controller_mod1_abs_mean":
                    controller_stats.get(
                        "mod1_abs_mean"
                    ),

                "controller_mod2_abs_mean":
                    controller_stats.get(
                        "mod2_abs_mean"
                    ),

                "controller_learned_gain":
                    controller_stats.get(
                        "learned_gain"
                    ),

                "freq_physical352_mean":
                    freq_ctx[
                        "physical352"
                    ].detach()
                    .mean(),

                "freq_physical88_mean":
                    freq_ctx[
                        "physical88"
                    ].detach()
                    .mean(),

                "freq_physical44_mean":
                    freq_ctx[
                        "physical44"
                    ].detach()
                    .mean(),
            }

        # THMNet expects deep-to-shallow feature order.
        return (
            x4,
            x3,
            x2,
            x1,
            freq_ctx,
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
