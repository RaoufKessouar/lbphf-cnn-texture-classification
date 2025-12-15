# Hybrid CNN + LBP-HF Project (KTH-TIPS2b)

This repository implements a texture classification pipeline combining a CNN (ResNet18 pre-trained on ImageNet) and an LBP-HF descriptor (Local Binary Pattern + Fourier) evaluated on the KTH-TIPS2b dataset.

A **lightweight demo version** is provided: the `Demo_Classification.ipynb` notebook runs without raw images thanks to the pre-extracted features cache (`features_all.npz`).

## Main Content
- `Demo_Classification.ipynb`: lightweight demo ready to run (uses pre-extracted features).
- `data/features_all.npz`: cache of fused CNN+LBP-HF descriptors for all images (with `samples` metadata).
- `data/sample_images/`: some sample images for LBP-HF and CNN visualizations.
- `scripts_archive/`: Full pipeline scripts:
  - `01_prepare_dataset_manifest.py`: preprocessing and generation of `manifest.csv`.
  - `02_samplewise_train_test_split.py`: split by sample (sample_a,b,c → train, sample_d → test).
  - `03_extract_hybrid_features.py`: extraction of CNN+LBP-HF features (fixed split).
  - `04_loso_crossval_evaluate.py`: LOSO cross-validation and generation of the `features_all.npz` cache.

## Prerequisites
- Python 3.10+ recommended
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Quick Start (Lightweight version for GitHub)
1. Current project structure:
   ```
   projet_hybride_cnn_lbp/
   ├─ Demo_Classification.ipynb
   ├─ requirements.txt
   ├─ README.md
   ├─ data/
   │  ├─ features_all.npz
   │  ├─ manifest.csv
   │  └─ sample_images/
   │     ├─ lettuce_leaf_sample_b_00164.png
   │     └─ corduroy_sample_a_00032.png
   └─ scripts_archive/
      ├─ 01_prepare_dataset_manifest.py
      ├─ 02_samplewise_train_test_split.py
      ├─ 03_extract_hybrid_features.py
      └─ 04_loso_crossval_evaluate.py
   ```
2. Open and run `Demo_Classification.ipynb`:
   - Uses `data/features_all.npz` directly (no need for raw images).
   - Visualization sections use PNGs from `data/sample_images/`.
   - Automatic download of ResNet18 ImageNet weights on first run (~45 MB).
3. Expected results (LOSO):
   - Average Accuracy ≈ 81.6% ± 2.9%
   - F1-weighted ≈ 80.3% ± 2.5%

## Full Pipeline (if regenerating data)
1. **Preprocessing**: resizes to 128×128 and creates `manifest.csv`.
   ```bash
   python scripts_archive/01_prepare_dataset_manifest.py
   # Input: RAW_KTHTIPS2B/<class>/sample_a|b|c|d/*.png
   # Outputs: data_resized/{rgb,gray}/..., manifest.csv
   ```
2. **Split by sample**: routes sample_a,b,c → train, sample_d → test.
   ```bash
   python scripts_archive/02_samplewise_train_test_split.py
   # Input: manifest.csv
   # Output: dataset_sample_split/{train,test}/{rgb,gray}/<class>/...
   ```
3. **Feature Extraction (fixed split)**: generates train/test NPZ.
   ```bash
   python scripts_archive/03_extract_hybrid_features.py
   # Input: dataset_sample_split/...
   # Outputs: features_npz_sample/features_train.npz, features_test.npz
   # Parameter: USE_AUGMENT=True (in the script) for rotations/flips/contrast
   ```
4. **LOSO Cross-Validation + Global Cache**: builds `features_all.npz` and figures.
   ```bash
   python scripts_archive/04_loso_crossval_evaluate.py
   # Input: dataset_sample_split/...
   # Outputs: features_npz_sample/features_all.npz, resultats/*.png, loso_scores.csv
   # The cache avoids recalculating features on subsequent runs.
   ```

## Notes on files
- `data/features_all.npz` contains:
  - `X` (9504 × 524) concatenating CNN features (512) + LBP-HF (12).
  - `y` labels, `classes` (11 names), `samples` (sample_a|b|c|d) aligned.
- `features_train.npz` / `features_test.npz` are useful for the fixed split but do not contain `samples` metadata.

## Source Data
- KTH-TIPS2b dataset expected in `RAW_KTHTIPS2B/` (11 classes, 4 physical samples a/b/c/d per class).
- Scripts do not automatically download raw data.

## Results and Visualizations
- `resultats/`: confusion matrices per fold, accuracy/F1 barplots, `loso_scores.csv`.
- The notebook shows:
  - LBP-HF visualization and rotation invariance (Fourier spectrum).
  - ResNet18 activation map visualization (orientation sensitivity).
  - LOSO evaluation and average confusion matrix in percentage.

## Author
[RaoufKessouar](https://github.com/RaoufKessouar)
