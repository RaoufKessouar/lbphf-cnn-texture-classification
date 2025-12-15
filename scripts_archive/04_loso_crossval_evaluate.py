"""
Validation croisée Leave-One-Sample-Out (LOSO) pour le pipeline CNN + LBP-HF.

Rôle
====
Réutiliser les descripteurs fusionnés pour évaluer la robustesse inter-
échantillon sur KTH-TIPS2b. Chaque fold retient un sous-échantillon (sample_a
à sample_d) pour le test et entraîne un SVM RBF sur les trois autres.

Entrées
-------
- ``dataset_sample_split`` : arborescence train/test RGB et GRAY issue du split.
- ``features_npz_sample/features_all.npz`` : cache optionnel des descripteurs.

Sorties
-------
- Matrices de confusion par fold (PNG) et graphiques de synthèse (accuracy, F1).
- ``loso_scores.csv`` récapitulant accuracy et F1 par fold et les moyennes ± écarts-types.
"""

import os, re, numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torchvision import models

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# ============================================================
# Paramètres globaux
# ============================================================
DATA_ROOT = "dataset_sample_split"
CACHE_NPZ = "features_npz_sample/features_all.npz"
RESULTS_DIR = "resultats"
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_AUGMENT = True     # <<< True/False : activer les rotations/flips/contrastes
RADIUS, N_POINTS = 1, 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Modèle CNN (ResNet18)
# ============================================================
print("Chargement du modèle ResNet18 pré-entraîné...")
cnn = models.resnet18(weights="IMAGENET1K_V1")
cnn.fc = torch.nn.Identity()
cnn.eval().to(device)

tfm_cnn = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ============================================================
# Fonctions utilitaires
# ============================================================
def extract_lbp_hf(img_gray_np, n_points=8, radius=1):
    lbp = local_binary_pattern(img_gray_np, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), density=True)
    fft = np.fft.fft(hist)
    hf = np.abs(fft[:len(fft)//2]).reshape(1, -1)
    return normalize(hf)[0]

def augment_image_list(img):
    v = [img]
    if USE_AUGMENT:
        for a in [90, 180, 270]: v.append(img.rotate(a))
        v.append(ImageOps.mirror(img)); v.append(ImageOps.flip(img))
        v.append(ImageEnhance.Contrast(img).enhance(1.3))
    return v

def extract_cnn_feat(img_pil):
    feats=[]
    with torch.no_grad():
        for im in augment_image_list(img_pil):
            t = tfm_cnn(im).unsqueeze(0).to(device)
            out = cnn(t).cpu().numpy().flatten()
            feats.append(out)
    return np.mean(feats, axis=0)

# ============================================================
# Scan et extraction des features
# ============================================================
def scan_all_images():
    entries=[]
    for split in ["train","test"]:
        for mod in ["rgb","gray"]:
            base = Path(DATA_ROOT)/split/mod
            if not base.exists(): continue
            for cls in sorted([d for d in base.iterdir() if d.is_dir()]):
                for fn in sorted(cls.iterdir()):
                    if fn.suffix.lower() not in [".png",".jpg",".jpeg",".tif",".bmp"]:
                        continue
                    rgbp  = Path(DATA_ROOT)/split/"rgb"/cls.name/fn.name
                    grayp = Path(DATA_ROOT)/split/"gray"/cls.name/fn.name
                    if not (rgbp.exists() and grayp.exists()):
                        continue
                    m = re.search(r"(sample_[abcd])", fn.name.lower())
                    sample = m.group(1) if m else None
                    entries.append((cls.name, str(rgbp), str(grayp), sample))
    return entries

def build_or_load_features():
    os.makedirs(Path(CACHE_NPZ).parent, exist_ok=True)
    if os.path.exists(CACHE_NPZ):
        d = np.load(CACHE_NPZ, allow_pickle=True)
        return d["X"], d["y"], d["classes"], d["samples"]

    entries = scan_all_images()
    if len(entries)==0:
        raise RuntimeError("Aucune image détectée. Vérifie dataset_sample_split/...")

    classes = sorted(list({e[0] for e in entries}))
    cls2idx = {c:i for i,c in enumerate(classes)}

    X, y, samples = [], [], []
    print(f"➡️ Extraction des features (USE_AUGMENT={USE_AUGMENT}) sur {len(entries)} images...")
    for cls, rgb, gray, samp in tqdm(entries):
        try:
            img_rgb  = Image.open(rgb).convert("RGB")
            img_gray = Image.open(gray).convert("L")
            feat_cnn = extract_cnn_feat(img_rgb)
            gray_np  = np.array(img_gray, dtype=np.float32)/255.0
            feat_lbp = extract_lbp_hf(gray_np, N_POINTS, RADIUS)
            X.append(np.concatenate([feat_cnn, feat_lbp]))
            y.append(cls2idx[cls])
            samples.append(samp if samp else "sample_unknown")
        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    samples = np.array(samples, dtype=object)

    np.savez_compressed(CACHE_NPZ, X=X, y=y, classes=np.array(classes, dtype=object), samples=samples)
    print(f"Cache écrit: {CACHE_NPZ} | {X.shape[0]} images, {X.shape[1]} features")
    return X, y, np.array(classes, dtype=object), samples

# ============================================================
# Cross-validation LOSO
# ============================================================
def save_confmat(cm, classes, title, out_png):
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title); plt.xlabel("Prédit"); plt.ylabel("Vrai")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def run_loso():
    X, y, classes, samples = build_or_load_features()
    folds = ["sample_a","sample_b","sample_c","sample_d"]
    fold_acc, fold_f1 = [], []
    rows = [("fold","accuracy","f1_weighted")]

    print("\n===== Validation croisée LOSO (a/b/c/d) =====")
    for held in folds:
        test = (samples == held)
        if test.sum()==0:
            print(f"⚠️ Pas d'images trouvées pour {held}, fold ignoré.")
            continue
        train = ~test
        Xtr, ytr = X[train], y[train]
        Xte, yte = X[test],  y[test]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        clf = SVC(kernel="rbf", C=10, gamma="scale")
        clf.fit(Xtr_s, ytr)
        yhat = clf.predict(Xte_s)

        acc = accuracy_score(yte, yhat)
        f1w = f1_score(yte, yhat, average="weighted")
        fold_acc.append(acc); fold_f1.append(f1w)
        rows.append((held, f"{acc:.4f}", f"{f1w:.4f}"))

        print(f"\n--- Fold: test={held} ---")
        print(f"Accuracy: {acc*100:.2f}% | F1-weighted: {f1w*100:.2f}%")
        print("(extrait du rapport):")
        print(classification_report(yte, yhat, target_names=classes, zero_division=0)[:400], "...")

        cm = confusion_matrix(yte, yhat, labels=range(len(classes)))
        save_confmat(cm, classes, f"LOSO — {held} (Acc={acc*100:.1f}%)", os.path.join(RESULTS_DIR, f"loso_fold_{held}_confmat.png"))

    if fold_acc:
        acc_mean, acc_std = np.mean(fold_acc), np.std(fold_acc)
        f1_mean, f1_std   = np.mean(fold_f1), np.std(fold_f1)
        print("\n===== Résumé LOSO =====")
        print(f"Accuracy moyenne: {acc_mean*100:.2f}%  ± {acc_std*100:.2f}%")
        print(f"F1-weighted moy.: {f1_mean*100:.2f}%  ± {f1_std*100:.2f}%")

        # Graphiques
        plt.figure(figsize=(7,4))
        plt.bar(["a","b","c","d"], [a*100 for a in fold_acc], color="steelblue")
        plt.ylim(0,100); plt.ylabel("Accuracy (%)"); plt.title("LOSO — Accuracy par fold")
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "loso_accuracy_bars.png"), dpi=200); plt.close()

        plt.figure(figsize=(7,4))
        plt.bar(["a","b","c","d"], [f*100 for f in fold_f1], color="orange")
        plt.ylim(0,100); plt.ylabel("F1-weighted (%)"); plt.title("LOSO — F1-score par fold")
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "loso_f1_bars.png"), dpi=200); plt.close()

        with open(os.path.join(RESULTS_DIR, "loso_scores.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerows(rows)
            w.writerow(("mean±std", f"{acc_mean:.4f}±{acc_std:.4f}", f"{f1_mean:.4f}±{f1_std:.4f}"))
        print(f"Figures & CSV sauvegardés dans: {RESULTS_DIR}")
    else:
        print("Aucun fold évalué (samples introuvables).")

if __name__ == "__main__":
    run_loso()
