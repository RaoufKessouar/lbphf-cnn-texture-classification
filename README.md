# Projet Hybrid CNN + LBP-HF (KTH-TIPS2b)

Ce dépôt implémente un pipeline de classification de textures combinant un CNN (ResNet18 pré-entraîné ImageNet) et un descripteur LBP-HF (Local Binary Pattern + Fourier) évalué sur le dataset KTH-TIPS2b.

Une **version légère de démonstration** est fournie : le notebook `Demo_Classification.ipynb` se lance sans images brutes grâce au cache de features pré-extraites (`features_all.npz`).

## Contenu principal
- `Demo_Classification.ipynb` : démo légère prête à exécuter (utilise des features pré-extraits).
- `data/features_all.npz` : cache des descripteurs fusionnés CNN+LBP-HF pour l'ensemble des images (avec métadonnées `samples`).
- `data/sample_images/` : quelques images d'exemple pour les visualisations LBP-HF et CNN.
- `scripts_archive/` : Scripts pipeline complet :
  - `01_prepare_dataset_manifest.py` : prétraitement et génération de `manifest.csv`.
  - `02_samplewise_train_test_split.py` : split par échantillon (sample_a,b,c → train, sample_d → test).
  - `03_extract_hybrid_features.py` : extraction des features CNN+LBP-HF (split fixe).
  - `04_loso_crossval_evaluate.py` : validation croisée LOSO et génération du cache `features_all.npz`.

## Prérequis
- Python 3.10+ recommandé
- Installation des dépendances :
  ```bash
  pip install -r requirements.txt
  ```

## Démarrage rapide (version légère pour GitHub)
1. Structure actuelle du projet :
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
2. Ouvrir et exécuter `Demo_Classification.ipynb` :
   - Utilise directement `data/features_all.npz` (pas besoin d'images brutes).
   - Les sections de visualisation utilisent les PNG de `data/sample_images/`.
   - Téléchargement automatique des poids ResNet18 ImageNet lors de la première exécution (~45 Mo).
3. Résultats attendus (LOSO) :
   - Accuracy moyenne ≈ 81.6% ± 2.9%
   - F1-weighted ≈ 80.3% ± 2.5%

## Pipeline complet (si vous régénérez les données)
1. **Prétraitement** : redimensionne en 128×128 et crée `manifest.csv`.
   ```bash
   python scripts_archive/01_prepare_dataset_manifest.py
   # Entrée : RAW_KTHTIPS2B/<classe>/sample_a|b|c|d/*.png
   # Sorties : data_resized/{rgb,gray}/..., manifest.csv
   ```
2. **Split par échantillon** : route sample_a,b,c → train, sample_d → test.
   ```bash
   python scripts_archive/02_samplewise_train_test_split.py
   # Entrée : manifest.csv
   # Sortie : dataset_sample_split/{train,test}/{rgb,gray}/<classe>/...
   ```
3. **Extraction des descripteurs (split fixe)** : génère les NPZ train/test.
   ```bash
   python scripts_archive/03_extract_hybrid_features.py
   # Entrée : dataset_sample_split/...
   # Sorties : features_npz_sample/features_train.npz, features_test.npz
   # Paramètre : USE_AUGMENT=True (dans le script) pour rotations/flips/contraste
   ```
4. **Validation croisée LOSO + cache global** : construit `features_all.npz` et figures.
   ```bash
   python scripts_archive/04_loso_crossval_evaluate.py
   # Entrée : dataset_sample_split/...
   # Sorties : features_npz_sample/features_all.npz, resultats/*.png, loso_scores.csv
   # Le cache évite de recalculer les features aux exécutions suivantes.
   ```

## Notes sur les fichiers
- `data/features_all.npz` contient :
  - `X` (9504 × 524) concaténant features CNN (512) + LBP-HF (12).
  - `y` labels, `classes` (11 noms), `samples` (sample_a|b|c|d) alignés.
- `features_train.npz` / `features_test.npz` sont utiles pour le split fixe mais ne contiennent pas les métadonnées `samples`.

## Données sources
- Dataset KTH-TIPS2b attendu dans `RAW_KTHTIPS2B/` (11 classes, 4 échantillons physiques a/b/c/d par classe).
- Les scripts ne téléchargent pas automatiquement les données brutes.

## Résultats et visualisations
- `resultats/` : matrices de confusion par fold, barplots accuracy/F1, `loso_scores.csv`.
- Le notebook montre :
  - Visualisation LBP-HF et invariance à la rotation (spectre de Fourier).
  - Visualisation des cartes d'activation ResNet18 (sensibilité à l'orientation).
  - Évaluation LOSO et matrice de confusion moyenne en pourcentage.

## Auteur
[RaoufKessouar](https://github.com/RaoufKessouar)