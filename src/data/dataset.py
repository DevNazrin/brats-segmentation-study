"""
BraTS 2021 dataset loading and preprocessing.

Handles locating cases on disk, splitting into train/val/test, and defining
MONAI transforms for both training (augmented) and evaluation (deterministic).
"""

import os
import tarfile
from pathlib import Path
from typing import List, Dict, Tuple, Literal

from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    RandFlipd,
    EnsureTyped,
    ConcatItemsd,
    Lambdad,
)
import numpy as np


# Modalities used as input channels (order matters for the model)
MODALITIES = ["t1", "t1ce", "t2", "flair"]


def extract_brats_if_needed(tar_path: str, target_dir: str) -> str:
    """
    Extract the BraTS tarball into target_dir if not already extracted.

    Returns the path to the extracted data root.
    Idempotent: calling multiple times is safe.
    """
    target = Path(target_dir)

    # Heuristic: if at least 100 BraTS_XXXXX folders exist, assume already extracted
    if target.exists():
        existing = list(target.glob("BraTS2021_*"))
        if len(existing) >= 100:
            print(f"BraTS already extracted at {target} ({len(existing)} cases found).")
            return str(target)

    target.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tar_path} to {target} ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target)
    print("Extraction complete.")

    return str(target)


def find_brats_cases(data_dir: str) -> List[Dict[str, str]]:
    """
    Walk the extracted BraTS directory and return a list of case dicts.

    Each dict contains absolute paths to four modalities and the segmentation label.
    Cases missing any required file are skipped (with a warning).
    """
    data_path = Path(data_dir)
    cases = []

    for case_dir in sorted(data_path.glob("BraTS2021_*")):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name
        case_files = {}
        all_present = True

        # Locate each modality file
        for modality in MODALITIES:
            mod_file = case_dir / f"{case_id}_{modality}.nii.gz"
            if not mod_file.exists():
                all_present = False
                break
            case_files[f"image_{modality}"] = str(mod_file)

        # Locate the segmentation label
        seg_file = case_dir / f"{case_id}_seg.nii.gz"
        if not seg_file.exists():
            all_present = False

        if not all_present:
            print(f"Warning: skipping {case_id} (missing files)")
            continue

        case_files["label"] = str(seg_file)
        case_files["case_id"] = case_id
        cases.append(case_files)

    print(f"Found {len(cases)} valid BraTS cases.")
    return cases


def split_cases(
    cases: List[Dict[str, str]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split cases into train/val/test reproducibly using sklearn's train_test_split.

    First splits off test set, then splits remaining into train/val.
    """
    # First split: separate test set
    train_val, test = train_test_split(
        cases, test_size=test_frac, random_state=seed, shuffle=True
    )

    # Second split: separate val from remaining
    val_size_relative = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_size_relative, random_state=seed, shuffle=True
    )

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def _binarize_label(label):
    """
    Convert BraTS multi-class labels (0, 1, 2, 4) to binary (0, 1).
    Used in binary segmentation mode.
    """
    return (label > 0).astype(np.float32) if hasattr(label, "astype") else (label > 0).float()


def get_transforms(
    mode: Literal["train", "val", "test"],
    num_classes: int = 1,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """
    Return MONAI transforms for the specified mode.

    - mode='train': includes random spatial cropping and flipping (data augmentation)
    - mode='val' or 'test': deterministic transforms only (no random ops),
      preserving full volume for sliding-window inference.

    num_classes=1 means binary segmentation (tumor vs background).
    num_classes=3 means multi-class (WT/TC/ET regions, computed from the raw labels
    by the loss/metric layers, not here).

    The four MRI modalities are concatenated into a 4-channel image tensor.
    Per-modality intensity normalization (NormalizeIntensityd) replaces the global
    ScaleIntensityRanged used in the first-semester baseline.
    """
    image_keys = [f"image_{m}" for m in MODALITIES]
    all_keys = image_keys + ["label"]

    base = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=all_keys, source_key="image_flair"),
        SpatialPadd(keys=all_keys, spatial_size=patch_size),
        ConcatItemsd(keys=image_keys, name="image", dim=0),
    ]

    if num_classes == 1:
        # Binary: collapse all tumor labels (1, 2, 4) into a single foreground class.
        base.append(Lambdad(keys=["label"], func=_binarize_label))

    if mode == "train":
        return Compose(
            base + [
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=patch_size,
                    random_size=False,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
    else:
        # val and test: no random ops. Volume is kept full-size; the training script
        # uses MONAI's sliding_window_inference for evaluation.
        return Compose(base + [EnsureTyped(keys=["image", "label"])])