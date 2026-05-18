"""
Streamlit demo for the BraTS brain tumor segmentation project.

Allows a user to either:
  1. Upload four MRI modality files (T1, T1ce, T2, FLAIR), or
  2. Load a built-in example case from data/example_case/.

Runs them through the trained U-Net and displays the segmentation overlay
alongside the radiologist's ground truth (when available), with a slice
slider and tumor volume estimate.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.demo.inference import (
    load_model,
    load_and_preprocess,
    run_inference,
    compute_tumor_volume_mm3,
    load_ground_truth,
)


CHECKPOINT_PATH = "outputs/checkpoints/unet_full_v1_best.pt"
EXAMPLE_CASE_DIR = Path("data/example_case")
EXAMPLE_CASE_ID = "BraTS2021_00000"


# ----------------------------------------------------------------------
# Model loading (cached so it loads only once per Streamlit session)
# ----------------------------------------------------------------------

@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device)
    return model, device


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def save_upload_to_tempfile(uploaded_file) -> str:
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def get_example_case_paths():
    """Return the four modality paths + the seg path for the example case."""
    base = EXAMPLE_CASE_DIR
    return {
        "t1": str(base / f"{EXAMPLE_CASE_ID}_t1.nii.gz"),
        "t1ce": str(base / f"{EXAMPLE_CASE_ID}_t1ce.nii.gz"),
        "t2": str(base / f"{EXAMPLE_CASE_ID}_t2.nii.gz"),
        "flair": str(base / f"{EXAMPLE_CASE_ID}_flair.nii.gz"),
        "seg": str(base / f"{EXAMPLE_CASE_ID}_seg.nii.gz"),
    }


def render_slice(t1ce_volume: np.ndarray, mask: np.ndarray, ground_truth: np.ndarray, slice_idx: int):
    """Render: T1ce | T1ce + GT overlay | T1ce + prediction overlay."""
    t1ce_slice = t1ce_volume[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]

    has_gt = ground_truth is not None
    n_cols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Panel 1: original T1ce
    axes[0].imshow(t1ce_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"T1ce — Slice {slice_idx}")
    axes[0].axis("off")

    # Panel 2: T1ce + ground truth (green) — only if available
    col_pred = 1
    if has_gt:
        gt_slice = ground_truth[:, :, slice_idx]
        axes[1].imshow(t1ce_slice.T, cmap="gray", origin="lower")
        gt_rgba = np.zeros((*gt_slice.T.shape, 4))
        gt_rgba[..., 1] = 1.0
        gt_rgba[..., 3] = gt_slice.T * 0.5
        axes[1].imshow(gt_rgba, origin="lower")
        axes[1].set_title(f"Ground Truth (Radiologist) — Slice {slice_idx}")
        axes[1].axis("off")
        col_pred = 2

    # Panel 3: T1ce + prediction (red)
    axes[col_pred].imshow(t1ce_slice.T, cmap="gray", origin="lower")
    pred_rgba = np.zeros((*mask_slice.T.shape, 4))
    pred_rgba[..., 0] = 1.0
    pred_rgba[..., 3] = mask_slice.T * 0.5
    axes[col_pred].imshow(pred_rgba, origin="lower")
    axes[col_pred].set_title(f"AI Prediction — Slice {slice_idx}")
    axes[col_pred].axis("off")

    plt.tight_layout()
    return fig


def run_segmentation_from_paths(t1_path, t1ce_path, t2_path, flair_path, seg_path=None):
    """Run the full pipeline starting from file paths. Returns dict of results."""
    model, device = get_model()
    image_tensor, t1ce_volume, voxel_spacing = load_and_preprocess(
        t1_path, t1ce_path, t2_path, flair_path
    )
    mask = run_inference(model, image_tensor, device)

    ground_truth = None
    if seg_path is not None and os.path.exists(seg_path):
        ground_truth = load_ground_truth(seg_path)

    return {
        "mask": mask,
        "t1ce_volume": t1ce_volume,
        "voxel_spacing": voxel_spacing,
        "ground_truth": ground_truth,
    }


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
        "A deep learning system for automatic brain tumor segmentation from multi-modal MRI. "
        "Built with a 3D U-Net trained on the BraTS 2021 dataset "
        "(test Dice = 0.92)."
    )
    st.divider()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Choose Input")

        mode = st.radio(
            "How would you like to provide MRI data?",
            options=["📂 Example case", "⬆️ Upload my own"],
            index=0,
        )

        st.divider()

        if mode == "⬆️ Upload my own":
            st.markdown("All four modalities are required.")
            t1_file = st.file_uploader("T1", type=["nii", "gz"], key="t1")
            t1ce_file = st.file_uploader("T1ce (contrast-enhanced)", type=["nii", "gz"], key="t1ce")
            t2_file = st.file_uploader("T2", type=["nii", "gz"], key="t2")
            flair_file = st.file_uploader("FLAIR", type=["nii", "gz"], key="flair")
            run_button = st.button("Run Segmentation", type="primary", use_container_width=True)
        else:
            st.info(
                f"📋 Example case: **{EXAMPLE_CASE_ID}**\n\n"
                "A BraTS 2021 patient with confirmed glioblastoma. "
                "The ground-truth segmentation (drawn by an expert radiologist) "
                "will be shown alongside the AI prediction for comparison."
            )
            run_button = st.button("Run Segmentation on Example", type="primary", use_container_width=True)

    # ---------------- Main area ----------------
    if not run_button:
        if mode == "⬆️ Upload my own":
            st.info("Please upload all four MRI modalities, then click **Run Segmentation**.")
        else:
            st.info("Click **Run Segmentation on Example** in the sidebar to start.")

        st.markdown(
            "### About this project\n"
            "- **Task:** binary brain-tumor segmentation from 3D MRI volumes\n"
            "- **Model:** 3D U-Net (4 input channels for T1, T1ce, T2, FLAIR)\n"
            "- **Training data:** BraTS 2021 (≈875 cases, full data)\n"
            "- **Inference:** sliding-window with 25% overlap on the full volume\n"
            "- **Test Dice:** 0.92 on held-out cases\n\n"
            "🚧 Research prototype. Not a medical device. For research and demonstration only."
        )
        return

    # ---------------- Resolve paths ----------------
    if mode == "📂 Example case":
        paths = get_example_case_paths()
        # Verify the example files exist
        missing = [k for k, v in paths.items() if not os.path.exists(v)]
        if missing:
            st.error(f"Example case files missing: {missing}. Check data/example_case/ exists.")
            return
        t1_path, t1ce_path, t2_path, flair_path = paths["t1"], paths["t1ce"], paths["t2"], paths["flair"]
        seg_path = paths["seg"]
        temp_files = []
    else:
        if not all([t1_file, t1ce_file, t2_file, flair_file]):
            st.error("Please upload all four MRI modalities before running.")
            return
        t1_path = save_upload_to_tempfile(t1_file)
        t1ce_path = save_upload_to_tempfile(t1ce_file)
        t2_path = save_upload_to_tempfile(t2_file)
        flair_path = save_upload_to_tempfile(flair_file)
        seg_path = None
        temp_files = [t1_path, t1ce_path, t2_path, flair_path]

    # ---------------- Run inference ----------------
    with st.spinner("Loading model and running 3D segmentation..."):
        results = run_segmentation_from_paths(
            t1_path, t1ce_path, t2_path, flair_path, seg_path
        )

    # Cleanup temp uploads
    for p in temp_files:
        try:
            os.unlink(p)
        except OSError:
            pass

    # ---------------- Results ----------------
    st.divider()
    st.subheader("✅ Segmentation complete")

    mask = results["mask"]
    t1ce_volume = results["t1ce_volume"]
    voxel_spacing = results["voxel_spacing"]
    ground_truth = results["ground_truth"]

    volume_mm3 = compute_tumor_volume_mm3(mask, voxel_spacing)
    volume_cm3 = volume_mm3 / 1000.0
    n_voxels = int(mask.sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Tumor volume (predicted)", f"{volume_cm3:.2f} cm³")
    col2.metric("Tumor voxels", f"{n_voxels:,}")
    col3.metric("Voxel size", f"{voxel_spacing[0]:.2f}×{voxel_spacing[1]:.2f}×{voxel_spacing[2]:.2f} mm")

    if ground_truth is not None:
        gt_volume_mm3 = compute_tumor_volume_mm3(ground_truth, voxel_spacing)
        gt_volume_cm3 = gt_volume_mm3 / 1000.0
        diff_cm3 = volume_cm3 - gt_volume_cm3
        st.caption(
            f"Ground-truth tumor volume: **{gt_volume_cm3:.2f} cm³**  ·  "
            f"Prediction differs by **{diff_cm3:+.2f} cm³**"
        )

    # Slice slider
    n_slices = t1ce_volume.shape[2]
    if mask.sum() > 0:
        tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
        default_slice = int(np.median(tumor_slices))
    else:
        default_slice = n_slices // 2

    slice_idx = st.slider(
        "Navigate through slices (drag to find the tumor):",
        min_value=0,
        max_value=n_slices - 1,
        value=default_slice,
    )

    fig = render_slice(t1ce_volume, mask, ground_truth, slice_idx)
    st.pyplot(fig)

    st.caption(
        "🟢 Green = radiologist's ground truth  ·  🔴 Red = AI prediction  ·  "
        "Use the slider to scroll through the brain."
    )


if __name__ == "__main__":
    main()