# Second-Semester Research Plan
**Project:** AI-Based Brain Tumor Segmentation Using Deep Learning Models
**Author:** Nazrin Mammadli
**Supervisor:** Dr. Öğr. Üyesi İrem Ülkü
**Date:** May 2026

## Summary

The first semester established a baseline pipeline comparing 3D U-Net and a custom TransUNet-Tiny on the BraTS 2021 dataset. The 3D U-Net reached a validation Dice of 0.92 and outperformed TransUNet-Tiny (0.81), confirming the well-documented finding that CNN architectures generally outperform small Transformer models in low-data medical imaging settings.

While this comparison is a valid baseline, it does not constitute a novel research contribution on its own — the result is consistent with prior literature. The second semester extends the project from a *replication* into a *study*, with a focused research question that goes beyond comparing two architectures.

## Proposed Research Direction

**Primary research question:**
*How does segmentation performance scale with training set size for CNN versus Transformer architectures in 3D brain tumor segmentation, and at what data scale (if any) does the Transformer architecture become competitive?*

This question matters because:

1. **Transformer architectures are known to be data-hungry.** Most published comparisons train all models on the full dataset, which obscures the data-efficiency trade-off. Quantifying this trade-off is genuinely useful information for the field.
2. **Medical imaging datasets are usually small.** Understanding where each architecture performs best is directly relevant to practical deployment in clinical settings, where data is often limited.
3. **The result is not trivially predictable.** While CNNs are expected to win at small data, the *crossover behavior* (if any) is the contribution.

## Experimental Design

The core experiment trains both architectures on multiple training set fractions:

- Training set sizes: 10%, 25%, 50%, 100% of available training data
- Architectures: 3D U-Net and TransUNet-Tiny
- Seeds: 3 random seeds per configuration to estimate variance
- Total runs: 4 sizes × 2 architectures × 3 seeds = 24 training runs

For each run, I will record:
- Validation Dice across epochs
- Final test-set Dice (full sliding-window inference)
- Per-case Dice distribution on the test set
- Training and inference time

## Secondary Analysis: Per-Case Failure Modes

Beyond aggregate Dice scores, I will analyze **where each model fails**:

- Identify the worst-performing 10% of cases for each model
- Examine whether the two models fail on the same cases or different cases
- Investigate whether failure cases share common features (small tumors, specific modalities, scanner artifacts)

This analysis adds qualitative depth to the quantitative results and could motivate ensembling if the models fail on different cases.

## Planned Extension: Multi-Class Segmentation

The first-semester work treated tumor segmentation as a binary problem (tumor vs. background). This significantly underuses the BraTS dataset, which provides three clinically meaningful sub-regions:
- Whole tumor (WT)
- Tumor core (TC)
- Enhancing tumor (ET)

Time permitting, the data-efficiency study will be extended to multi-class segmentation, reporting per-region Dice scores. This is a stretch goal rather than a primary deliverable, conditional on the binary study completing successfully.

## Methodology Improvements over First Semester

In refactoring the project, several methodological issues from the first semester are being corrected:

- **Sliding-window inference** replaces evaluation on randomly cropped patches, ensuring reported Dice reflects full-volume performance.
- **Per-modality intensity normalization** replaces the global 0–4000 intensity scaling, which was applied uniformly to all four modalities despite their differing intensity distributions.
- **Deterministic validation transforms** replace random cropping during validation, eliminating noise in epoch-to-epoch comparisons.
- **Best-checkpoint saving** based on validation Dice replaces last-epoch saving.
- **Reproducibility controls** (seed setting, deterministic mode, configuration logging) are added to support multi-seed experiments.
- **Test-set evaluation** is performed; previously the test set was constructed but unused.
- **Validation-loop bug fix** in the TransUNet training (validation now runs every epoch).

## Engineering Foundation

The project has been refactored from a single Colab notebook into a structured repository with separated data, model, training, and evaluation modules. Configuration is YAML-based to support multi-run experiments. Training runs use Weights & Biases for experiment tracking. The repository is public on GitHub and is intended to remain reproducible for external readers.

## Compute Plan

University HPC is unavailable. Experiments will run on:
- **Kaggle Notebooks** (30 GPU hours/week, T4 ×2) — primary training environment
- **Google Colab Pro** — development and quick iteration

For 24 training runs with estimated 3–4 GPU hours each, the total compute budget is ~80–100 GPU hours, fitting within Kaggle's weekly allocation across approximately 3–4 weeks.

## Target Outcomes

- **Primary deliverable:** A polished public GitHub repository with reproducible experiments, complete results, and clear documentation.
- **Publication target:** A workshop paper submission to a venue such as the MICCAI workshops, MIDL, or a similar venue welcoming student work. Backup target: an arXiv preprint paired with the GitHub repository.

Aiming at publication is a deliberate target rather than aspirational language. I am co-author on a recently accepted Springer LNCS conference proceedings paper (medical, MongoDB-related) and have direct experience with the academic submission process. The publishability outcome for the present project still depends on the strength of the experimental results and is not guaranteed, but the writing and submission stages of the process are within reach.

## Approximate Timeline

- **Weeks 1–2:** Refactor and infrastructure (in progress)
- **Weeks 3–6:** Run data-efficiency experiments (binary segmentation)
- **Weeks 7–8:** Per-case failure analysis and writing
- **Weeks 9–10:** Multi-class extension (stretch)
- **Weeks 11–12:** Final polish, paper draft, submission

## Risks and Mitigations

- **Compute bottleneck:** 30 GPU hours/week may limit experiment iteration. Mitigation: prioritize binary results; treat multi-class as stretch.
- **Variance in small-data runs:** training on 10% of data may produce highly variable results. Mitigation: 3 seeds minimum, report confidence intervals.
- **Publishability uncertainty:** the result of the data-efficiency study may not be novel enough for a paper. Mitigation: target the engineering quality and documentation as the guaranteed outcome; treat publication as upside.
