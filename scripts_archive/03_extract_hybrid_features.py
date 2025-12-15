"""
Extraction des descripteurs hybrides CNN + LBP-HF (split fixe sample_d).

Entrées attendues
-----------------
- ``dataset_sample_split/train/{rgb,gray}/<classe>/*.png``
- ``dataset_sample_split/test/{rgb,gray}/<classe>/*.png``

Sorties produites
-----------------
- ``features_npz_sample/features_train.npz`` : matrice X et labels y du train.
- ``features_npz_sample/features_test.npz``  : matrice X et labels y du test.

Principe
--------
1) Extraction CNN : ResNet18 pré-entraîné (couche fc remplacée par Identity).
2) Extraction LBP-HF : histogramme LBP uniforme, transformée de Fourier,
   magnitude des demi-bandes normalisée.
3) Fusion : concaténation ``[feat_cnn | feat_lbp]``.
4) Option d'augmentation (rotations, flips, contraste) moyennée pour renforcer
   l'invariance empirique.
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps, ImageEnhance
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
import torch
import torchvision.transforms as T
from torchvision import models
from pathlib import Path

# ===========================================================================
# Paramètres
# ===========================================================================
USE_AUGMENT = True  # <<< Mets False pour désactiver l'augmentation
DATA_SPLIT_DIR = "dataset_sample_split"
OUT_DIR = "features_npz_sample"
RADIUS = 1
N_POINTS = 8 * RADIUS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# Modèle CNN
# ===========================================================================
print("Chargement ResNet18 pré-entraîné...")
cnn = models.resnet18(weights="IMAGENET1K_V1")
cnn.fc = torch.nn.Identity()
cnn.eval().to(device)

tfm_cnn = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

# ===========================================================================
# Fonctions de calcul des descripteurs
# ===========================================================================
def extract_lbp_hf(img_gray_np, n_points=8, radius=1):
    lbp = local_binary_pattern(img_gray_np, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), density=True)
    fft = np.fft.fft(hist)
    hf = np.abs(fft[:len(fft)//2]).reshape(1, -1)
    return normalize(hf)[0]

def augment_image_list(img):
    v = [img]
    if USE_AUGMENT:
        for a in [90, 180, 270]:
            v.append(img.rotate(a))
        v.append(ImageOps.mirror(img))
        v.append(ImageOps.flip(img))
        v.append(ImageEnhance.Contrast(img).enhance(1.3))
    return v

def extract_cnn_feat(img_pil):
    feats = []
    with torch.no_grad():
        for im in augment_image_list(img_pil):
            t = tfm_cnn(im).unsqueeze(0).to(device)
            out = cnn(t).cpu().numpy().flatten()
            feats.append(out)
    return np.mean(feats, axis=0)

# ===========================================================================
# Boucle principale : parcours train / test, extraction et sérialisation
# ===========================================================================
os.makedirs(OUT_DIR, exist_ok=True)

for split in ["train", "test"]:
    print(f"\ Extraction features pour split: {split} (USE_AUGMENT={USE_AUGMENT})")
    dir_rgb  = Path(DATA_SPLIT_DIR) / split / "rgb"
    dir_gray = Path(DATA_SPLIT_DIR) / split / "gray"

    classes = sorted([d.name for d in dir_rgb.iterdir() if d.is_dir()])
    X, y = [], []

    for cls_idx, cls in enumerate(classes):
        rgb_cls = dir_rgb / cls
        gray_cls = dir_gray / cls
        if not rgb_cls.is_dir():
            continue

        for img_name in tqdm(os.listdir(rgb_cls), desc=f"{split} | {cls}", leave=False):
            rgb_path  = rgb_cls / img_name
            gray_path = gray_cls / img_name
            if not (rgb_path.exists() and gray_path.exists()):
                continue
            try:
                img_rgb  = Image.open(rgb_path).convert("RGB")
                img_gray = Image.open(gray_path).convert("L")
                feat_cnn = extract_cnn_feat(img_rgb)
                gray_np  = np.array(img_gray, dtype=np.float32) / 255.0
                feat_lbp = extract_lbp_hf(gray_np, N_POINTS, RADIUS)
                X.append(np.concatenate([feat_cnn, feat_lbp]))
                y.append(cls_idx)
            except Exception:
                continue

    if len(X) == 0:
        print(f"Aucun descripteur pour {split}. Vérifier la structure/chemins.")
        continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    np.savez_compressed(f"{OUT_DIR}/features_{split}.npz", X=X, y=y, classes=np.array(classes, dtype=object))
    print(f"Sauvegardé: {OUT_DIR}/features_{split}.npz | {X.shape[0]} images, {X.shape[1]} features")

print("\ Terminé.")