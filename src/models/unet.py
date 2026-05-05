"""
3D U-Net model for brain tumor segmentation.

Uses MONAI's pre-built UNet implementation with a configuration appropriate for
4-channel multi-modal MRI input volumes.
"""

from monai.networks.nets import UNet


def build_unet(in_channels: int = 4, out_channels: int = 2) -> UNet:
    """
    Build a 3D U-Net for brain tumor segmentation.

    Default in_channels=4 corresponds to the four BraTS modalities (T1, T1ce, T2, FLAIR).
    Default out_channels=2 corresponds to binary segmentation (background + tumor).
    For multi-class segmentation, set out_channels=4 (background + 3 tumor regions).

    The architecture matches the first-semester baseline:
    - 5-level encoder/decoder
    - Channel sizes: 16, 32, 64, 128, 256
    - 2 residual units per level
    - Stride 2 between levels for downsampling
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )