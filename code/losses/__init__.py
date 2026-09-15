# -*- coding: utf-8 -*-

from .sf_sod_loss import (
    BoundaryAwareStructureLoss,
    SoftFMeasureLoss,
    EvaluationAlignedCurveFLoss,
    MultiScaleSSIMStructureLoss,
    ObjectConsistencyLoss,
    BoundaryDiceLoss,
    SoftMAELoss,
    R15MainLoss,
    R15RegionTeacherLoss,
    R22SupervisedLoss,
    R22CleanAnchorLoss,
    R22TrustedAugConsistencyLoss,
)

__all__ = [
    "BoundaryAwareStructureLoss",
    "SoftFMeasureLoss",
    "EvaluationAlignedCurveFLoss",
    "MultiScaleSSIMStructureLoss",
    "ObjectConsistencyLoss",
    "BoundaryDiceLoss",
    "SoftMAELoss",
    "R15MainLoss",
    "R15RegionTeacherLoss",
    "R22SupervisedLoss",
    "R22CleanAnchorLoss",
    "R22TrustedAugConsistencyLoss",
]
