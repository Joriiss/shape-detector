# Projet Data Science & IA

Ce repo contient quatre exercices axés sur la reconnaissance d'images et l'analyse d'images par IA.

## Structure du Projet

```
shapes/
├── main.py              # Exercice 1 : Classification de Formes (CNN)
├── requirements.txt     # Dépendances Python
├── data/               # Données d'entraînement pour la classification de formes
│   ├── circle/
│   ├── square/
│   ├── triangle/
│   └── test/
├── tp2/                # Exercice 2 : Détection de Chiffres Romains
│   ├── main.py         # Implémentation Gemini AI
│   └── data/           # Images d'exemple avec chiffres romains
├── tp3/                # Exercice 3 : Détection de Contours avec OpenCV
│   ├── main.py         # Détection de contours
│   ├── cameraman.tif   # Image d'exemple
│   └── cameraman_contours.png  # Image résultat
├── tp4/                # Exercice 4 : Colorisation d'Images avec CNN
│   ├── main.py         # Conversion N&B et colorisation
│   └── models/         # Fichiers du modèle de colorisation
└── README.md           # Ce fichier
```

---

## Exercice 1 : Classification de Formes avec CNN

**Emplacement :** Répertoire racine (`main.py`)

### Description
Un réseau de neurones convolutif (CNN) construit avec TensorFlow/Keras qui classe des formes géométriques (cercles, carrés et triangles) à partir d'images.

### Fonctionnalités
- Modèle d'apprentissage profond utilisant une architecture CNN
- Entraînement sur des images en niveaux de gris (64x64 pixels)
- Classification des formes en 3 catégories : cercle, carré, triangle
- Capacités d'évaluation et de test du modèle

### Prérequis
- Python 3.7+
- TensorFlow
- NumPy
- scikit-learn
- Pillow (PIL)

### Installation
```bash
pip install -r requirements.txt
```

### Utilisation
```bash
python main.py
```

Le programme va :
1. Charger les données d'entraînement depuis le répertoire `data/`
2. Entraîner un modèle CNN pendant 10 époques
3. Sauvegarder le modèle entraîné sous `shape_model.h5`
4. Tester le modèle sur les images dans `data/test/`

### Architecture du Modèle
- 3 couches convolutives avec MaxPooling
- Couches denses pour la classification
- Sortie : 3 classes (cercle, carré, triangle)

---

## Exercice 2 : Détection de Chiffres Romains avec Gemini AI

**Emplacement :** Répertoire `tp2/`

### Description
Un programme Python qui utilise l'IA Gemini de Google (Gemini 2.0 Flash) pour détecter et identifier les chiffres romains dans les images en utilisant les capacités de vision.

### Fonctionnalités
- Utilise le modèle de vision Gemini 2.0 Flash
- Détecte automatiquement les chiffres romains (I, V, X, L, C, D, M) dans les images
- Interface en ligne de commande simple
- Gestion sécurisée de la clé API avec le fichier `.env`

### Prérequis
- Python 3.7+
- Clé API Google Gemini (avec le billing configuré)
- google-generativeai
- python-dotenv
- Pillow

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1. Obtenez votre clé API Gemini depuis [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Créez un fichier `.env` à la racine du projet :
```
GEMINI_API_KEY=votre_clé_api_ici
```

### Utilisation
```bash
python tp2/main.py tp2/data/numeral-1.png
```

Le programme analyse l'image et affiche tous les chiffres romains trouvés.

---

## Exercice 3 : Détection de Contours avec OpenCV

**Emplacement :** Répertoire `tp3/`

### Description
Un programme Python qui détecte et trace automatiquement les contours des formes principales dans une image en utilisant OpenCV. Ce programme implémente un algorithme de détection de contours basé sur les techniques de traitement d'image classiques.

> **Note :** Ce programme s'est basé sur le tutoriel de [AranaCorp sur la détection de contour avec OpenCV et Python](https://www.aranacorp.com/fr/detection-de-contour-avec-opencv-et-python/#google_vignette).

### Fonctionnalités
- Conversion automatique en niveaux de gris
- Réduction du bruit avec flou gaussien
- Détection des bords avec l'algorithme de Canny
- Détection et traçage des contours avec `findContours()`
- Sauvegarde automatique de l'image avec contours
- Support des images en niveaux de gris et en couleur

### Prérequis
- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Matplotlib (optionnel, pour visualisation)

### Installation
```bash
pip install -r requirements.txt
```

### Utilisation
```bash
python tp3/main.py
```

Le programme va :
1. Charger l'image `cameraman.tif` depuis le répertoire `tp3/`
2. Redimensionner l'image à 512px de largeur
3. Convertir l'image en niveaux de gris
4. Appliquer un flou gaussien pour réduire le bruit
5. Détecter les bords avec l'algorithme de Canny
6. Trouver les contours avec `cv2.findContours()`
7. Dessiner les contours en vert sur l'image originale
8. Afficher les fenêtres avec les résultats intermédiaires
9. Sauvegarder l'image avec contours sous `cameraman_contours.png`

### Algorithme de Détection

Le programme suit les étapes suivantes :

1. **Conversion en niveaux de gris** : `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
2. **Floutage gaussien** : `cv2.GaussianBlur(gray, (5,5), 0)` pour éliminer le bruit
3. **Détection de bords Canny** : `cv2.Canny(blur, 30, 200)` avec deux seuils d'hystérésis
4. **Détection de contours** : `cv2.findContours()` avec :
   - Mode : `cv2.RETR_TREE` (retourne une arborescence complète des contours)
   - Méthode d'approximation : `cv2.CHAIN_APPROX_SIMPLE` (compresse les segments)
5. **Dessin des contours** : `cv2.drawContours()` en vert avec une épaisseur de 3 pixels

### Paramètres

- **Taille du flou** : 5x5 pixels (paramètre `blur_kernel_size`)
- **Seuils Canny** : 30 (seuil bas) et 200 (seuil haut)
- **Mode de récupération** : `RETR_TREE` (tous les contours avec hiérarchie)
- **Méthode d'approximation** : `CHAIN_APPROX_SIMPLE` (optimise le nombre de points)

### Résultat

Le programme génère une image `cameraman_contours.png` dans le répertoire `tp3/` avec tous les contours détectés dessinés en vert sur l'image originale.

---

## Exercice 4 : Colorisation d'Images avec CNN

**Emplacement :** Répertoire `tp4/`

### Description
Un programme Python qui convertit une image couleur en niveaux de gris, puis utilise un modèle CNN pré-entraîné pour reconstruire une image couleur réaliste à partir de l'image en noir et blanc.

> **Note :** Les modèles utilisés proviennent de [mariyakhannn/imagecolorizer](https://github.com/mariyakhannn/imagecolorizer).

### Fonctionnalités
- Conversion d'images couleur en niveaux de gris
- Colorisation automatique d'images N&B avec un modèle CNN pré-entraîné (Caffe)
- Utilisation de l'espace colorimétrique LAB pour la colorisation
- Sauvegarde automatique des résultats

### Prérequis
- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Fichiers du modèle de colorisation (voir Configuration)

### Installation
```bash
pip install -r requirements.txt
```

### Configuration

**⚠️ Important :** Les fichiers du modèle ne sont pas inclus dans ce dépôt car ils sont trop volumineux (le fichier `.caffemodel` fait ~300MB). Vous devez les télécharger manuellement.

1. **Créez le répertoire `models/`** dans `tp4/` :
```bash
mkdir tp4/models
```

2. **Téléchargez les fichiers du modèle** depuis [mariyakhannn/imagecolorizer](https://github.com/mariyakhannn/imagecolorizer) :

   - **`colorization_deploy_v2.prototxt`** : 
     - Lien direct : https://raw.githubusercontent.com/mariyakhannn/imagecolorizer/main/colorization_deploy_v2.prototxt
     - Ou depuis le dépôt : https://github.com/mariyakhannn/imagecolorizer/blob/main/colorization_deploy_v2.prototxt
   
   - **`pts_in_hull.npy`** :
     - Lien direct : https://github.com/mariyakhannn/imagecolorizer/raw/main/pts_in_hull.npy
     - Ou depuis le dépôt : https://github.com/mariyakhannn/imagecolorizer/blob/main/pts_in_hull.npy
   
   - **`colorization_release_v2.caffemodel`** (~300MB) :
     - Lien direct : https://github.com/mariyakhannn/imagecolorizer/raw/main/colorization_release_v2.caffemodel
     - Ou depuis le dépôt : https://github.com/mariyakhannn/imagecolorizer/blob/main/colorization_release_v2.caffemodel
     - **Note :** Ce fichier est volumineux, le téléchargement peut prendre du temps.

3. **Placez les fichiers téléchargés** dans le répertoire `tp4/models/` :
```
tp4/
├── main.py
└── models/
    ├── colorization_deploy_v2.prototxt
    ├── pts_in_hull.npy
    └── colorization_release_v2.caffemodel
```

**Alternative :** Vous pouvez cloner le dépôt [mariyakhannn/imagecolorizer](https://github.com/mariyakhannn/imagecolorizer) et copier les fichiers nécessaires :
```bash
git clone https://github.com/mariyakhannn/imagecolorizer.git
cp imagecolorizer/colorization_deploy_v2.prototxt tp4/models/
cp imagecolorizer/pts_in_hull.npy tp4/models/
cp imagecolorizer/colorization_release_v2.caffemodel tp4/models/
```

### Utilisation

**Convertir une image en noir et blanc :**
```bash
python tp4/main.py bw image.jpg
```

**Coloriser une image N&B :**
```bash
python tp4/main.py colorize image_bw.jpg
```

**Avec options :**
```bash
# Spécifier un autre répertoire pour les modèles
python tp4/main.py colorize image_bw.jpg --model-dir ./models

# Afficher les images (originale et colorisée)
python tp4/main.py colorize image_bw.jpg --display

# Spécifier le fichier de sortie
python tp4/main.py colorize image_bw.jpg output.jpg
```

### Comment ça fonctionne

Le programme utilise un modèle CNN pré-entraîné basé sur le framework Caffe :

1. **Conversion en N&B** : L'image couleur est convertie en niveaux de gris
2. **Prétraitement** : L'image est normalisée et convertie en espace colorimétrique LAB
3. **Prédiction** : Le modèle CNN prédit les canaux de couleur (a et b) à partir du canal de luminosité (L)
4. **Reconstruction** : Les canaux prédits sont combinés avec le canal L original pour créer l'image colorisée

### Résultat

Le programme génère :
- `{nom}_bw.{ext}` : Image convertie en noir et blanc
- `{nom}_colorized.{ext}` : Image colorisée à partir du N&B

---

## Installation (Toutes les Dépendances)

Installez tous les packages requis pour les quatre exercices :

```bash
pip install -r requirements.txt
```

### Dépendances
- `tensorflow` - Framework d'apprentissage profond (Exercice 1)
- `numpy` - Calcul numérique
- `scikit-learn` - Utilitaires d'apprentissage automatique
- `google-generativeai` - API Gemini AI (Exercice 2)
- `Pillow` - Traitement d'images
- `python-dotenv` - Gestion des variables d'environnement (Exercice 2)
- `opencv-python` - Bibliothèque de vision par ordinateur (Exercice 3)

---

## Notes

- Pour l'Exercice 2, assurez-vous que votre fichier `.env` se trouve dans le répertoire racine du projet
- Le fichier `.env` doit être ajouté à `.gitignore` pour garder votre clé API sécurisée
- L'Exercice 1 nécessite des données d'entraînement dans la structure de répertoire `data/`
- L'Exercice 3 nécessite que le fichier `cameraman.tif` soit présent dans le répertoire `tp3/`
- L'Exercice 4 nécessite les fichiers du modèle dans le répertoire `tp4/models/` (voir Configuration ci-dessus). Ces fichiers ne sont pas inclus dans le dépôt et doivent être téléchargés manuellement depuis [mariyakhannn/imagecolorizer](https://github.com/mariyakhannn/imagecolorizer)
- Tous les exercices prennent en charge les formats d'image courants (PNG, JPEG, TIFF, etc.)

## Références

- **Exercice 3** : Basé sur le tutoriel [Détection de contour avec OpenCV et Python](https://www.aranacorp.com/fr/detection-de-contour-avec-opencv-et-python/#google_vignette) par AranaCorp
- **Exercice 4** : Modèles de colorisation provenant de [mariyakhannn/imagecolorizer](https://github.com/mariyakhannn/imagecolorizer)

---
