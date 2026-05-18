"""
Streamlit demo for the BraTS brain tumor segmentation project.

Allows a user to upload four MRI modality files (T1, T1ce, T2, FLAIR),
runs them through the trained U-Net, and displays the segmentation overlay
with a slice slider and tumor volume estimate.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make sure 'src' imports work when running streamlit from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.demo.inference import (
    load_model,
    load_and_preprocess,
    run_inference,
    compute_tumor_volume_mm3,
)


CHECKPOINT_PATH = "outputs/checkpoints/unet_full_v1_best.pt"


# ----------------------------------------------------------------------
# Model loading (cached so it loads only once per Streamlit session)
# ----------------------------------------------------------------------

@st.cache_resource
def get_model():
    """Load the trained U-Net once and reuse across requests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device)
    return model, device


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def save_upload_to_tempfile(uploaded_file) -> str:
    """Streamlit gives us UploadedFile objects; nibabel needs paths."""
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def render_slice(t1ce_volume: np.ndarray, mask: np.ndarray, slice_idx: int):
    """Render one slice with the tumor mask overlaid on the T1ce image."""
    # Pick a slice along the axial axis (last axis for nibabel default loading)
    t1ce_slice = t1ce_volume[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: original T1ce
    axes[0].imshow(t1ce_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"T1ce — Slice {slice_idx}")
    axes[0].axis("off")

    # Right: T1ce with red tumor overlay
    axes[1].imshow(t1ce_slice.T, cmap="gray", origin="lower")
    mask_rgba = np.zeros((*mask_slice.T.shape, 4))
    mask_rgba[..., 0] = 1.0   # red channel
    mask_rgba[..., 3] = mask_slice.T * 0.5  # alpha = 0.5 where tumor
    axes[1].imshow(mask_rgba, origin="lower")
    axes[1].set_title(f"Prediction — Slice {slice_idx}")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Brain Tumor Segmentation",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 AI-Based Brain Tumor Segmentation")
    st.markdown(
        "Upload a patient's multi-modal MRI scans (T1, T1ce, T2, FLAIR) "
        "to automatically segment the tumor region using a 3D U-Net trained on BraTS 2021."
    )
    st.divider()

    # ---------------- Sidebar: file uploads ----------------
    with st.sidebar:
        st.header("📁 Upload MRI Files")
        st.markdown("All four modalities are required.")

        t1_file = st.file_uploader("T1", type=["nii", "gz"], key="t1")
        t1ce_file = st.file_uploader("T1ce (contrast-enhanced)", type=["nii", "gz"], key="t1ce")
        t2_file = st.file_uploader("T2", type=["nii", "gz"], key="t2")
        flair_file = st.file_uploader("FLAIR", type=["nii", "gz"], key="flair")

        st.divider()
        run_button = st.button("Run Segmentation", type="primary", use_container_width=True)

    # ---------------- Main area ----------------
    all_uploaded = all([t1_file, t1ce_file, t2_file, flair_file])

    if not all_uploaded:
        st.info("Please upload all four MRI modalities to begin.")
        st.markdown(
            "**About this project:** A deep learning system for automatic brain tumor "
            "segmentation from multi-modal MRI scans. Part of an ongoing research project "
            "comparing CNN and Transformer-based architectures on the BraTS 2021 dataset. "
            "The model is a 3D U-Net achieving test Dice = 0.92 on held-out cases."
        )
        return

    if not run_button:
        st.success("All four modalities uploaded. Click **Run Segmentation** to proceed.")
        return

    # ---------------- Inference ----------------
    with st.spinner("Loading model..."):
        model, device = get_model()

    with st.spinner("Preprocessing uploaded MRIs..."):
        t1_path = save_upload_to_tempfile(t1_file)
        t1ce_path = save_upload_to_tempfile(t1ce_file)
        t2_path = save_upload_to_tempfile(t2_file)
        flair_path = save_upload_to_tempfile(flair_file)

        image_tensor, t1ce_volume, voxel_spacing = load_and_preprocess(
            t1_path, t1ce_path, t2_path, flair_path
        )

    with st.spinner("Running 3D segmentation (sliding-window inference)..."):
        mask = run_inference(model, image_tensor, device)

    # Cleanup temp files
    for p in [t1_path, t1ce_path, t2_path, flair_path]:
        try:
            os.unlink(p)
        except OSError:
            pass

    # ---------------- Results ----------------
    st.divider()
    st.subheader("✅ Segmentation complete")

    # Compute tumor volume
    volume_mm3 = compute_tumor_volume_mm3(mask, voxel_spacing)
    volume_cm3 = volume_mm3 / 1000.0
    n_voxels = int(mask.sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Tumor volume", f"{volume_cm3:.2f} cm³")
    col2.metric("Tumor voxels", f"{n_voxels:,}")
    col3.metric("Voxel size", f"{voxel_spacing[0]:.2f}×{voxel_spacing[1]:.2f}×{voxel_spacing[2]:.2f} mm")

    # Slice slider for navigation
    n_slices = t1ce_volume.shape[2]
    # Default slice: middle of the tumor if found, else middle of volume
    if mask.sum() > 0:
        tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
        default_slice = int(np.median(tumor_slices))
    else:
        default_slice = n_slices // 2

    slice_idx = st.slider(
        "Navigate through slices (use the slider to find the tumor):",
        min_value=0,
        max_value=n_slices - 1,
        value=default_slice,
    )

    fig = render_slice(t1ce_volume, mask, slice_idx)
    st.pyplot(fig)

    st.caption(
        "⚠️ Research prototype. Not a medical device. For research and demonstration only."
    )


if __name__ == "__main__":
    main()