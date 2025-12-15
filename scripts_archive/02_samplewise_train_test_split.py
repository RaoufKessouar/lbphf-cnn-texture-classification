"""
Découpage inter-échantillon (sample-wise) pour l'évaluation KTH-TIPS2b.

Objectif
========
Recomposer un split entraînement/test en respectant la séparation par sous-
échantillons : ``sample_a``, ``sample_b``, ``sample_c`` pour l'apprentissage,
``sample_d`` pour le test. Les images RGB et GRAY générées au prétraitement
restent strictement synchronisées.

Entrées
-------
- ``manifest.csv`` produit par ``01_preprocess_and_manifest.py`` contenant
  les triples « classe, chemin_rgb, chemin_gray ».

Sorties
-------
- ``dataset_sample_split/train/{rgb,gray}/<classe>/*.png``
- ``dataset_sample_split/test/{rgb,gray}/<classe>/*.png``

Logique
-------
1) Lecture du manifeste pour récupérer les chemins canoniques.
2) Routage de chaque image vers ``train`` ou ``test`` selon le sample détecté
    dans son nom de fichier.
3) Copie simultanée des versions RGB et GRAY afin de conserver l'alignement des
    paires.
"""


import os
import csv
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# ===========================================================================
# Paramètres généraux
# ===========================================================================
MANIFEST = "manifest.csv"                   # Fichier produit par le script 01
OUT_ROOT = "dataset_sample_split"           # Dossier de sortie
SAMPLES_TRAIN = ["sample_a", "sample_b", "sample_c"]
SAMPLES_TEST = ["sample_d"]

# Création de la structure de base (arborescences train/test × rgb/gray)
for split in ["train", "test"]:
    for branch in ["rgb", "gray"]:
        os.makedirs(os.path.join(OUT_ROOT, split, branch), exist_ok=True)

print("\nCréation du split par échantillon (sample_a,b,c → train, sample_d → test)...")

# ===========================================================================
# Lecture du manifeste pour récupérer les chemins source
# ===========================================================================
rows = []
with open(MANIFEST, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# ===========================================================================
# Fonction utilitaire de copie
# ===========================================================================
def copy_item(src, dst_root, split, branch):
    """
    Copie une image dans le dossier approprié du split (train/test) et de la
    branche (rgb/gray), en préservant l'organisation par classe.
    """
    cls = Path(src).parent.parent.name
    dst_dir = os.path.join(dst_root, split, branch, cls)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dst_dir, Path(src).name))

# ===========================================================================
# Répartition des fichiers par sous-échantillon
# ===========================================================================
for row in rows:
    path_rgb, path_gray = row["path_rgb"], row["path_gray"]

    # Détection du sample selon le chemin du fichier
    if any(sample in path_rgb for sample in SAMPLES_TRAIN):
        split = "train"
    elif any(sample in path_rgb for sample in SAMPLES_TEST):
        split = "test"
    else:
        # Sécurité : si le chemin ne contient pas "sample_*"
        split = "train"

    # Copie synchronisée des images RGB et GRAY
    copy_item(path_rgb, OUT_ROOT, split, "rgb")
    copy_item(path_gray, OUT_ROOT, split, "gray")

print("Split inter-échantillon terminé → dossier :", OUT_ROOT)

# Visualisation rapide de la répartition résultante
import os

splits = ["train", "test"]
counts = [sum(len(files) for _,_,files in os.walk(f"{OUT_ROOT}/{s}/rgb")) for s in splits]

plt.figure(figsize=(6,4))
plt.bar(splits, counts, color=["#5cb85c", "#5bc0de"])
plt.title("Répartition des images par split (échantillon)")
plt.ylabel("Nombre d'images")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
