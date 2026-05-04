"""
DataLoader construction for BraTS segmentation.

Wraps the dataset cases and transforms into PyTorch DataLoaders ready for training,
validation, and testing.
"""

from typing import Tuple, Optional

from torch.utils.data import DataLoader
from monai.data import CacheDataset

from .dataset import (
    extract_brats_if_needed,
    find_brats_cases,
    split_cases,
    get_transforms,
)


def _make_loader(
    cases,
    mode: str,
    num_classes: int,
    patch_size: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    cache_rate: float,
    shuffle: bool,
) -> DataLoader:
    """Build a single CacheDataset + DataLoader for the given cases and mode."""
    transforms = get_transforms(mode=mode, num_classes=num_classes, patch_size=patch_size)

    dataset = CacheDataset(
        data=cases,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_dataloaders(
    tar_path: str,
    extracted_dir: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_classes: int = 1,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 1,
    num_workers: int = 2,
    cache_rate: float = 0.1,
    seed: int = 42,
    train_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    train_fraction supports the data-efficiency study: passing 0.1 uses 10% of the
    training set, 0.5 uses 50%, etc. The validation and test sets are NOT subsetted
    so that comparisons across train_fraction values remain valid.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 1. Make sure the data is extracted
    data_dir = extract_brats_if_needed(tar_path, extracted_dir)

    # 2. Discover cases
    cases = find_brats_cases(data_dir)

    # 3. Split into train/val/test
    train_cases, val_cases, test_cases = split_cases(
        cases, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    # 4. (Optional) Subset the training set for the data-efficiency study
    if train_fraction < 1.0:
        n_keep = max(1, int(len(train_cases) * train_fraction))
        # Deterministic: take the first n_keep after the shuffle in split_cases
        train_cases = train_cases[:n_keep]
        print(f"Using {train_fraction:.0%} of training data: {len(train_cases)} cases")

    # 5. Build a DataLoader for each split
    train_loader = _make_loader(
        train_cases, mode="train", num_classes=num_classes, patch_size=patch_size,
        batch_size=batch_size, num_workers=num_workers, cache_rate=cache_rate,
        shuffle=True,
    )
    val_loader = _make_loader(
        val_cases, mode="val", num_classes=num_classes, patch_size=patch_size,
        batch_size=1, num_workers=num_workers, cache_rate=cache_rate,
        shuffle=False,
    )
    test_loader = _make_loader(
        test_cases, mode="test", num_classes=num_classes, patch_size=patch_size,
        batch_size=1, num_workers=num_workers, cache_rate=cache_rate,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader