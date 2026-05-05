"""
Evaluation metrics for segmentation: Dice (primary), IoU and Hausdorff (optional).
"""

from typing import List, Dict
import numpy as np
import torch

from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric


def compute_metrics(
    results: List[dict],
    num_classes: int = 1,
    include_hausdorff: bool = False,
) -> Dict[str, float]:
    """
    Compute aggregate metrics over a list of inference results.

    Returns a dict with mean Dice, mean IoU, and optionally Hausdorff distance,
    plus the per-case Dice list (useful for failure analysis).
    """
    dice = DiceMetric(include_background=False, reduction="mean_batch")
    iou = MeanIoU(include_background=False, reduction="mean_batch")
    hd = HausdorffDistanceMetric(include_background=False, reduction="mean_batch", percentile=95)

    per_case_dice = []

    for r in results:
        pred = r["pred"].unsqueeze(0)   # add batch dim
        label = r["label"].unsqueeze(0)

        # Per-case Dice for failure analysis
        case_dice = DiceMetric(include_background=False, reduction="mean")
        case_dice(y_pred=pred, y=label)
        per_case_dice.append({
            "case_id": r["case_id"],
            "dice": float(case_dice.aggregate().item()),
        })

        # Aggregate metrics
        dice(y_pred=pred, y=label)
        iou(y_pred=pred, y=label)
        if include_hausdorff:
            hd(y_pred=pred, y=label)

    out = {
        "mean_dice": float(dice.aggregate().mean().item()),
        "mean_iou": float(iou.aggregate().mean().item()),
        "per_case_dice": per_case_dice,
    }
    if include_hausdorff:
        out["mean_hausdorff_95"] = float(hd.aggregate().mean().item())

    return out