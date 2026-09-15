# -*- coding: utf-8 -*-
"""Network entry for Reset-R15 region-Mamba semantic-frequency repair THMNet."""

from pathlib import Path

import torch
import torch.nn as nn

from model.pvtv2 import pvt_v2_b2
from model.SF_backbone import FrequencyAwarePVTv2
from model.SF_VMmamba_decoder import Decoder


class Net(nn.Module):
    def __init__(
        self,
        load_pretrained=True,
        pretrain_path=None,
    ):
        super().__init__()

        backbone = pvt_v2_b2()

        if load_pretrained:
            path = self._resolve_pretrain_path(
                pretrain_path
            )

            self._load_pvt_pretrained(
                backbone,
                path,
            )

        self.encoder = FrequencyAwarePVTv2(
            backbone
        )

        self.decoder = Decoder(
            128
        )

        self.sigmoid = nn.Sigmoid()

        # R12 inference uses one no-grad semantic anchor encoder pass followed
        # by one controlled encoder pass. The decoder runs only once.
        self.feedback_enabled = True
        self.corrector_enabled = True
        self.region_enabled = True
        self._last_anchor_context = None

    @staticmethod
    def _resolve_pretrain_path(
        pretrain_path=None,
    ):
        repo_root = Path(
            __file__
        ).resolve().parents[1]

        candidates = []

        if pretrain_path is not None:
            candidates.append(
                Path(
                    pretrain_path
                ).expanduser()
            )

        candidates.extend(
            [
                repo_root
                / "pretrain"
                / "pvt_v2_b2.pth",

                repo_root
                / "Net_MSA_Decoder"
                / "pretrain"
                / "pvt_v2_b2.pth",

                Path(
                    "Net_MSA_Decoder/pretrain/pvt_v2_b2.pth"
                ),
            ]
        )

        for path in candidates:
            if path.is_file():
                return path

        raise FileNotFoundError(
            "Cannot find PVTv2-B2 pretrained weights.\nChecked:\n{}".format(
                "\n".join(
                    "  - {}".format(
                        p
                    )
                    for p in candidates
                )
            )
        )

    @staticmethod
    def _unwrap_checkpoint(
        obj,
    ):
        if not isinstance(
            obj,
            dict,
        ):
            return obj

        for key in (
            "state_dict",
            "model",
            "model_state_dict",
        ):
            if (
                key in obj
                and isinstance(
                    obj[
                        key
                    ],
                    dict,
                )
            ):
                return obj[
                    key
                ]

        return obj

    def _load_pvt_pretrained(
        self,
        backbone,
        path,
    ):
        model_dict = backbone.state_dict()

        ckpt = torch.load(
            str(
                path
            ),
            map_location="cpu",
        )

        ckpt = self._unwrap_checkpoint(
            ckpt
        )

        cleaned = {}

        for key, value in ckpt.items():
            new_key = key

            if new_key.startswith(
                "module."
            ):
                new_key = new_key[
                    len(
                        "module."
                    ):
                ]

            for prefix in (
                "backbone.",
                "encoder.",
            ):
                if new_key.startswith(
                    prefix
                ):
                    new_key = new_key[
                        len(
                            prefix
                        ):
                    ]

            cleaned[
                new_key
            ] = value

        matched = {
            key:
                value
            for key, value
            in cleaned.items()
            if (
                key in model_dict
                and hasattr(
                    value,
                    "shape",
                )
                and value.shape
                == model_dict[
                    key
                ].shape
            )
        }

        ratio = (
            len(
                matched
            )
            / max(
                len(
                    model_dict
                ),
                1,
            )
        )

        print(
            "PVT pretrained:",
            path,
        )

        print(
            "Matched PVT tensors: {}/{} ({:.2%})".format(
                len(
                    matched
                ),
                len(
                    model_dict
                ),
                ratio,
            )
        )

        if ratio < 0.80:
            raise RuntimeError(
                "PVT checkpoint incompatible: matched {:.2%}".format(
                    ratio
                )
            )

        model_dict.update(
            matched
        )

        backbone.load_state_dict(
            model_dict
        )

        print(
            "Pretrained PVTv2-B2 encoder loaded."
        )

    @staticmethod
    def _pack_outputs(
        sigmoid,
        P5,
        P4,
        P3,
        P2,
        P1,
    ):
        return (
            P5,
            sigmoid(
                P5
            ),

            P4,
            sigmoid(
                P4
            ),

            P3,
            sigmoid(
                P3
            ),

            P2,
            sigmoid(
                P2
            ),

            P1,
            sigmoid(
                P1
            ),
        )

    def set_runtime_modes(
        self,
        feedback_enabled=True,
        corrector_enabled=True,
        region_enabled=True,
    ):
        self.feedback_enabled = bool(
            feedback_enabled
        )
        self.corrector_enabled = bool(
            corrector_enabled
        )

        self.region_enabled = bool(
            region_enabled
        )

    def _semantic_anchor(
        self,
        x,
    ):
        """
        Build a stable no-grad Stage3/Stage4 semantic anchor.

        During training, temporarily switch the encoder to eval mode so
        DropPath does not inject random controller context. Individual module
        training states are restored exactly afterwards.
        """
        states = {
            module:
                module.training
            for module in self.encoder.modules()
        }

        self.encoder.eval()

        with torch.no_grad():
            (
                x4,
                x3,
                _,
                _,
                _,
            ) = self.encoder(
                x,
                feedback_context=None,
            )

        for module, state in states.items():
            module.training = state

        return {
            "x3":
                x3.detach(),

            "x4":
                x4.detach(),
        }

    def forward_base(
        self,
        x,
    ):
        """
        Single-pass R9-compatible path:
            no semantic feedback controller
            no R12 balanced corrector

        Used for the frozen R9 teacher during training and for parity tests.
        """
        shape = x.size()[2:]

        (
            x4,
            x3,
            x2,
            x1,
            freq_ctx,
        ) = self.encoder(
            x,
            feedback_context=None,
        )

        (
            P5,
            P4,
            P3,
            P2,
            P1,
        ) = self.decoder(
            x4,
            x3,
            x2,
            x1,
            raw_rgb=x,
            freq_ctx=freq_ctx,
            shape=shape,
            anchor_context=None,
            use_corrector=False,
            use_region_repair=False,
        )

        return self._pack_outputs(
            self.sigmoid,
            P5,
            P4,
            P3,
            P2,
            P1,
        )

    def forward(
        self,
        x,
    ):
        shape = x.size()[2:]

        if (
            self.feedback_enabled
            or self.corrector_enabled
            or self.region_enabled
        ):
            anchor_context = self._semantic_anchor(
                x
            )
        else:
            anchor_context = None

        (
            x4,
            x3,
            x2,
            x1,
            freq_ctx,
        ) = self.encoder(
            x,
            feedback_context=(
                anchor_context
                if self.feedback_enabled
                else None
            ),
        )

        (
            P5,
            P4,
            P3,
            P2,
            P1,
        ) = self.decoder(
            x4,
            x3,
            x2,
            x1,
            raw_rgb=x,
            freq_ctx=freq_ctx,
            shape=shape,
            anchor_context=(
                anchor_context
                if (
                    self.corrector_enabled
                    or self.region_enabled
                )
                else None
            ),
            use_corrector=(
                self.corrector_enabled
                and anchor_context is not None
            ),
            use_region_repair=(
                self.region_enabled
                and anchor_context is not None
            ),
        )

        self._last_anchor_context = anchor_context

        return self._pack_outputs(
            self.sigmoid,
            P5,
            P4,
            P3,
            P2,
            P1,
        )

    def runtime_stats(
        self,
    ):
        stats = {}

        for prefix, module in (
            (
                "encoder",
                self.encoder,
            ),
            (
                "decoder",
                self.decoder,
            ),
        ):
            if hasattr(
                module,
                "runtime_stats",
            ):
                for key, value in module.runtime_stats().items():
                    stats[
                        "{}_{}".format(
                            prefix,
                            key,
                        )
                    ] = value

        return stats

    def training_aux_tensors(
        self,
    ):
        aux = {}

        aux.update(
            self.encoder.training_aux_tensors()
        )

        aux.update(
            self.decoder.training_aux_tensors()
        )

        return aux
