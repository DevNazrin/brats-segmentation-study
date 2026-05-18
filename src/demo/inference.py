"""
Inference utilities for the Streamlit demo.

Provides a clean API for loading the trained U-Net and running segmentation
on a set of four uploaded MRI modality files.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    CropForeground,
    SpatialPad,
    EnsureType,
)
from monai.inferers import sliding_window_inference

from src.models.unet import build_unet


# Standard BraTS modality order. Must match training.
MODALITY_ORDER = ["t1", "t1ce", "t2", "flair"]


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Build a U-Net and load the trained weights from a checkpoint file.

    Args:
        checkpoint_path: path to the .pt file
        device: torch.device('cuda') or torch.device('cpu')

    Returns:
        A model in eval mode, ready for inference.
    """
    model = build_unet(in_channels=4, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_and_preprocess(
    t1_path: str,
    t1ce_path: str,
    t2_path: str,
    flair_path: str,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[float, float, float]]:
    """
    Load four MRI modalities, apply the same preprocessing as training,
    and return the stacked 4-channel tensor ready for the model.

    Also returns the raw T1ce volume (for visualization background) and the
    voxel spacing (for volume calculations).
    """
    # Load each modality as a numpy array, plus the affine (for voxel spacing)
    modalities = {}
    for name, path in zip(
        MODALITY_ORDER,
        [t1_path, t1ce_path, t2_path, flair_path],
    ):
        img = nib.load(path)
        modalities[name] = img.get_fdata().astype(np.float32)
        if name == "t1ce":
            t1ce_affine = img.affine
            t1ce_raw = modalities[name].copy()

    # Voxel spacing in mm (from affine diagonal)
    voxel_spacing = tuple(float(abs(t1ce_affine[i, i])) for i in range(3))

    # Stack as channel-first array (4, D, H, W)
    image = np.stack(
        [modalities[m] for m in MODALITY_ORDER],
        axis=0,
    )

    # Preprocessing transforms (matching training, minus randomness)
    transform = Compose([
        # Already channel-first, so EnsureChannelFirst expects no channel dim;
        # since image has channel dim, we skip this. Instead apply directly:
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])

    image_tensor = torch.from_numpy(image)
    image_tensor = transform(image_tensor)  # (4, D, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # (1, 4, D, H, W) — add batch dim

    return image_tensor.float(), t1ce_raw, voxel_spacing


def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.25,
) -> np.ndarray:
    """
    Run sliding-window inference on a preprocessed image tensor.

    Returns a 3D numpy mask (D, H, W) with values 0 (background) or 1 (tumor).
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=patch_size,
            sw_batch_size=2,
            predictor=model,
            overlap=overlap,
        )

    # Binary segmentation: apply sigmoid + threshold at 0.5
    probs = torch.sigmoid(logits)
    mask = (probs > 0.5).float()

    # Strip batch + channel dims → (D, H, W)
    mask = mask.squeeze().cpu().numpy().astype(np.uint8)
    return mask


def compute_tumor_volume_mm3(mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> float:
    """
    Compute the total tumor volume in cubic millimeters.

    mask: 3D binary array (D, H, W)
    voxel_spacing: (dz, dy, dx) in mm
    """
    voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    n_tumor_voxels = int(mask.sum())
    return n_tumor_voxels * voxel_volume_mm3