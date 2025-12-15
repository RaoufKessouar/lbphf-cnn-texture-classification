"""
Prétraitement du dataset KTH-TIPS2b pour la chaîne CNN + LBP-HF.

Objectif
========
Normaliser l'ensemble des images brutes en 128×128 pixels, générer
leurs versions RGB et niveaux de gris, puis produire un manifeste CSV
référençant chaque fichier. Ce manifeste sert d'entrée unique pour les
étapes de split, d'extraction de descripteurs et d'entraînement.

Entrées
-------
- Dossier racine ``RAW_KTHTIPS2B`` structuré par classe et par sous-échantillon
  (sample_a, sample_b, sample_c, sample_d).

Sorties
-------
- ``data_resized/rgb/<classe>/<sample_*>/*.png`` : images redimensionnées en RGB.
- ``data_resized/gray/<classe>/<sample_*>/*.png`` : images redimensionnées en niveaux de gris.
- ``manifest.csv`` : tableau « classe, chemin_rgb, chemin_gray » aligné sur les images.

Étapes principales
------------------
1) Parcours récursif des classes et sous-échantillons.
2) Ouverture, conversion en RGB, redimensionnement bilinéaire 128×128.
3) Export simultané RGB et gris, nommage canonique ``<classe>_<sample>_<id>.png``.
4) Alimentation du manifeste CSV pour traçabilité des fichiers générés.
"""

import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import random

# ===========================================================================
# Paramètres généraux
# ===========================================================================
RAW_DIR = "RAW_KTHTIPS2B"               # dossier des données brutes (avec sample_a…d)
OUT_RGB_DIR  = "data_resized/rgb"
OUT_GRAY_DIR = "data_resized/gray"
IMG_SIZE = (128, 128)
MANIFEST = "manifest.csv"
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".bmp")

# Préparation des dossiers de sortie
os.makedirs(OUT_RGB_DIR,  exist_ok=True)
os.makedirs(OUT_GRAY_DIR, exist_ok=True)

# ===========================================================================
# Parcours du dataset brut et génération des paires RGB/GRAY
# ===========================================================================
classes = [c for c in sorted(os.listdir(RAW_DIR)) if os.path.isdir(os.path.join(RAW_DIR, c))]
rows = []

print(f"Début du prétraitement pour {len(classes)} classes...\n")

for cls in classes:
    src_cls = os.path.join(RAW_DIR, cls)
    count = 0

    # Parcours des sous-dossiers sample_a … sample_d
    for sample_name in sorted(os.listdir(src_cls)):
        sample_path = os.path.join(src_cls, sample_name)
        if not os.path.isdir(sample_path):
            continue

        # Création des sous-dossiers correspondants dans les sorties
        out_rgb_sample  = os.path.join(OUT_RGB_DIR,  cls, sample_name)
        out_gray_sample = os.path.join(OUT_GRAY_DIR, cls, sample_name)
        os.makedirs(out_rgb_sample,  exist_ok=True)
        os.makedirs(out_gray_sample, exist_ok=True)

        for fn in os.listdir(sample_path):
            if not fn.lower().endswith(EXTS):
                continue

            src_path = os.path.join(sample_path, fn)
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Skip {src_path}: {e}")
                continue

            # Redimensionnement et nommage canonique
            img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
            base_name = f"{cls}_{sample_name}_{count:05d}.png"

            dst_rgb  = os.path.join(out_rgb_sample,  base_name)
            dst_gray = os.path.join(out_gray_sample, base_name)

            # Sauvegarde synchronisée RGB et GRAY
            img_resized.save(dst_rgb, format="PNG", optimize=True)
            img_resized.convert("L").save(dst_gray, format="PNG", optimize=True)

            # Indexation pour le manifeste CSV
            rows.append([cls, dst_rgb, dst_gray])
            count += 1

    print(f"{cls}: {count} images traitées.")

# ===========================================================================
# Écriture du manifeste CSV consolidé
# ===========================================================================
with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["class", "path_rgb", "path_gray"])
    w.writerows(rows)

print(f"\nPrétraitement terminé.")
print(f" - Images RGB : {OUT_RGB_DIR}")
print(f" - Images GRAY : {OUT_GRAY_DIR}")
print(f" - Manifest : {MANIFEST} ({len(rows)} lignes)")

# Visualisation rapide d'un échantillon de sorties
sample_classes = random.sample(classes, min(6, len(classes)))
plt.figure(figsize=(12,6))
for i, cls in enumerate(sample_classes):
    folder = os.path.join(OUT_RGB_DIR, cls)
    subfolder = random.choice([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    subpath = os.path.join(folder, subfolder)
    img_name = random.choice([f for f in os.listdir(subpath) if f.lower().endswith('.png')])
    img = Image.open(os.path.join(subpath, img_name))
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"{cls}/{subfolder}")
    plt.axis("off")

plt.suptitle("Exemples d'images redimensionnées (128×128)")
plt.show()
