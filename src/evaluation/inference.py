"""
Inference utilities for trained segmentation models.

Wraps MONAI's sliding_window_inference into a clean function that accepts a model,
a DataLoader, and returns per-case predictions and ground-truth labels.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose


def run_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int = 1,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
) -> List[dict]:
    """
    Run sliding-window inference on every case in the loader.

    Returns a list of dicts, one per case, each containing:
      - 'pred': discrete prediction tensor (CPU, shape [C, D, H, W])
      - 'label': ground-truth tensor (CPU, shape [C, D, H, W])
      - 'case_id': case identifier (if available in the batch)
    """
    model.eval()
    if num_classes == 1:
        post_pred = Compose([AsDiscrete(threshold=0.5)])
    else:
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])

    results = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            case_id = batch.get("case_id", ["unknown"])[0]

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )

            pred = post_pred(logits).cpu()
            results.append({
                "pred": pred[0],
                "label": labels[0],
                "case_id": case_id,
            })

    return results