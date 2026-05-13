"""
Streamlit demo for the BraTS brain tumor segmentation project.

Allows a user to upload four MRI modality files (T1, T1ce, T2, FLAIR),
runs them through a trained model, and displays the segmentation overlay.

Current state: skeleton only. Model inference is wired up on Day 2.
"""

import streamlit as st


def main():
    # Page setup
    st.set_page_config(
        page_title="Brain Tumor Segmentation",
        page_icon="🧠",
        layout="wide",
    )

    # Header
    st.title("🧠 AI-Based Brain Tumor Segmentation")
    st.markdown(
        "Upload a patient's multi-modal MRI scans (T1, T1ce, T2, FLAIR) "
        "to automatically segment the tumor region."
    )

    st.divider()

    # Sidebar: file upload + controls
    with st.sidebar:
        st.header("📁 Upload MRI Files")
        st.markdown("All four modalities are required.")

        t1_file = st.file_uploader("T1", type=["nii", "nii.gz"], key="t1")
        t1ce_file = st.file_uploader("T1ce (contrast-enhanced)", type=["nii", "nii.gz"], key="t1ce")
        t2_file = st.file_uploader("T2", type=["nii", "nii.gz"], key="t2")
        flair_file = st.file_uploader("FLAIR", type=["nii", "nii.gz"], key="flair")

        st.divider()

        run_button = st.button("Run Segmentation", type="primary", use_container_width=True)

    # Main area
    all_uploaded = all([t1_file, t1ce_file, t2_file, flair_file])

    if not all_uploaded:
        st.info("Please upload all four MRI modalities to begin.")
        st.markdown(
            "**About this project:** This is a deep learning system for automatic "
            "brain tumor segmentation from multi-modal MRI scans. It is part of an "
            "ongoing research project comparing CNN and Transformer-based architectures "
            "on the BraTS 2021 dataset."
        )
        return

    if not run_button:
        st.success("All four modalities uploaded. Click **Run Segmentation** to proceed.")
        return

    # Placeholder for inference (wired up in Day 2)
    with st.spinner("Running segmentation... (placeholder, real inference coming)"):
        st.warning(
            "🚧 Model inference not yet implemented. This is a Day 1 skeleton.\n\n"
            "Day 2 wires this up to the real trained U-Net checkpoint."
        )


if __name__ == "__main__":
    main()