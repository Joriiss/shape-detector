# Projet Data Science & IA

Ce repo contient cinq exercices axés sur la reconnaissance d'images, l'analyse d'images par IA et le traitement audio.

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
├── tp-audio/           # Exercice 5 : Analyse et Traitement Audio
│   ├── tp1.py          # Analyse audio, visualisation et modification
│   └── hello.mp3       # Fichier audio d'exemple
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

## Exercice 5 : Analyse et Traitement Audio

**Emplacement :** Répertoire `tp-audio/`

### Description
Un programme Python pour analyser, visualiser et traiter des fichiers audio. Le programme permet de charger des fichiers audio (MP3, WAV, etc.), d'afficher des graphiques d'amplitude et des spectrogrammes, et de modifier la vitesse ou supprimer les silences.

> **Note :** Le code pour afficher le spectrogramme et le diagramme d'amplitude s'est basé sur le tutoriel [Visualisation de données audio en Python](https://www.kaggle.com/code/ghazouanihaythem/tmm-visualisation-de-donn-es-audio-en-python#%C3%89tape-3.-Spectre-de-Fr%C3%A9quence-/-Spectrogramme) par ghazouanihaythem sur Kaggle.

### Fonctionnalités
- Chargement de fichiers audio (MP3, WAV, etc.) avec `librosa`
- Visualisation de l'amplitude du signal dans le temps (canaux gauche et droit)
- Génération de spectrogrammes pour analyser les fréquences
- Modification de la vitesse de l'audio sans changer la hauteur
- Suppression automatique des silences au début et à la fin
- Sauvegarde automatique des graphiques en images PNG
- Support des fichiers mono et stéréo

### Prérequis
- Python 3.7+
- librosa (traitement audio)
- soundfile (sauvegarde audio)
- matplotlib (visualisation)
- numpy
- ffmpeg (requis par librosa pour lire les fichiers MP3)

### Installation
```bash
pip install -r requirements.txt
```

**Note :** Pour lire les fichiers MP3, vous devez également installer `ffmpeg` :
- **Windows** : Téléchargez depuis https://ffmpeg.org/download.html et ajoutez-le à votre PATH
- **Linux** : `sudo apt-get install ffmpeg`
- **macOS** : `brew install ffmpeg`

### Utilisation

**Analyse basique d'un fichier audio :**
```bash
python tp-audio/tp1.py hello.mp3
```

**Modifier la vitesse de l'audio :**
```bash
# 2x plus rapide
python tp-audio/tp1.py hello.mp3 --speed 2.0

# 0.5x (2x plus lent)
python tp-audio/tp1.py hello.mp3 --speed 0.5
```

**Supprimer les silences :**
```bash
# Avec le seuil par défaut (20 dB)
python tp-audio/tp1.py hello.mp3 --remove-silence

# Avec un seuil personnalisé (30 dB)
python tp-audio/tp1.py hello.mp3 --remove-silence 30
```

**Combiner les options :**
```bash
# Supprimer les silences puis modifier la vitesse
python tp-audio/tp1.py hello.mp3 --remove-silence --speed 1.5
```

### Résultats générés

Le programme génère automatiquement les fichiers suivants :

- **`canal_gauche.png`** : Graphique d'amplitude du canal gauche en fonction du temps
- **`canal_droit.png`** : Graphique d'amplitude du canal droit en fonction du temps
- **`spectrogramme_canal_gauche.png`** : Spectrogramme du canal gauche (fréquence vs temps)
- **`spectrogramme_canal_droit.png`** : Spectrogramme du canal droit (fréquence vs temps)

Si des modifications sont appliquées (vitesse ou suppression de silence), les fichiers audio modifiés sont également sauvegardés :
- **`{nom}_speed_{facteur}x.wav`** : Fichier audio avec vitesse modifiée
- **`{nom}_no_silence.wav`** : Fichier audio sans silences

### Paramètres

- **`--speed <facteur>`** : Modifie la vitesse de l'audio
  - `1.0` = vitesse normale
  - `2.0` = 2x plus rapide
  - `0.5` = 2x plus lent

- **`--remove-silence [seuil]`** : Supprime les silences
  - Seuil par défaut : 20 dB
  - Plus le seuil est élevé, plus de silence est conservé

### Comment ça fonctionne

1. **Chargement audio** : `librosa.load()` charge le fichier audio et retourne le signal et la fréquence d'échantillonnage
2. **Visualisation d'amplitude** : Les graphiques montrent l'amplitude du signal en fonction du temps pour chaque canal
3. **Spectrogramme** : `plt.specgram()` génère une représentation temps-fréquence montrant l'intensité des différentes fréquences
4. **Modification de vitesse** : `librosa.effects.time_stretch()` modifie la vitesse sans changer la hauteur (pitch)
5. **Suppression de silence** : `librosa.effects.trim()` détecte et supprime automatiquement les silences au début et à la fin

---

## Exercice 5b : Ajout de Bruit et Réduction de Bruit Audio

**Emplacement :** Répertoire `tp-audio/tp2.py`

### Description
Un programme Python pour ajouter du bruit artificiel (bruit blanc gaussien) à un fichier audio, puis tenter de restaurer le son original via un filtre passe-bas. Ce programme permet d'étudier les effets du bruit sur les signaux audio et les techniques de réduction de bruit.

> **Note :** Le code pour ajouter du bruit gaussien s'est basé sur une solution de [Stack Overflow](https://stackoverflow.com/q) par user13832229 (CC BY-SA 4.0).

### Fonctionnalités
- Ajout de bruit gaussien (bruit blanc) à un fichier audio
- Calcul du RMS (Root Mean Square) du signal original
- Réduction de bruit avec un filtre passe-bas Butterworth
- Génération automatique de graphiques (amplitude et spectrogrammes) pour les signaux bruités et filtrés
- Sauvegarde organisée dans des dossiers séparés (`noisy/` et `filtered/`)

### Prérequis
- Python 3.7+
- librosa (traitement audio)
- soundfile (sauvegarde audio)
- matplotlib (visualisation)
- numpy
- scipy (filtres numériques)
- ffmpeg (requis par librosa pour lire les fichiers MP3)

### Installation
```bash
pip install -r requirements.txt
```

### Utilisation

**Ajouter du bruit à un fichier audio :**
```bash
# Avec la déviation standard par défaut (0.05)
python tp-audio/tp2.py add-noise hello.mp3

# Avec une déviation standard personnalisée
python tp-audio/tp2.py add-noise hello.mp3 --std 0.01
```

**Filtrer l'audio bruité avec un filtre passe-bas :**
```bash
# Avec la fréquence de coupure par défaut (3000 Hz)
python tp-audio/tp2.py filter noisy/hello_noisy.wav

# Avec une fréquence de coupure personnalisée
python tp-audio/tp2.py filter noisy/hello_noisy.wav --cutoff 4000
```

### Résultats générés

**Fichiers dans `noisy/` :**
- **`{nom}_noisy.wav`** : Fichier audio avec bruit gaussien ajouté
- **`{nom}_noisy_canal_gauche.png`** : Graphique d'amplitude du canal gauche (bruité)
- **`{nom}_noisy_canal_droit.png`** : Graphique d'amplitude du canal droit (bruité)
- **`{nom}_noisy_spectrogramme_canal_gauche.png`** : Spectrogramme du canal gauche (bruité)
- **`{nom}_noisy_spectrogramme_canal_droit.png`** : Spectrogramme du canal droit (bruité)

**Fichiers dans `filtered/` :**
- **`{nom}_filtered_lowpass_{freq}Hz.wav`** : Fichier audio filtré (passe-bas)
- **`{nom}_filtered_lowpass_{freq}Hz_canal_gauche.png`** : Graphique d'amplitude du canal gauche (filtré)
- **`{nom}_filtered_lowpass_{freq}Hz_canal_droit.png`** : Graphique d'amplitude du canal droit (filtré)
- **`{nom}_filtered_lowpass_{freq}Hz_spectrogramme_canal_gauche.png`** : Spectrogramme du canal gauche (filtré)
- **`{nom}_filtered_lowpass_{freq}Hz_spectrogramme_canal_droit.png`** : Spectrogramme du canal droit (filtré)

### Paramètres

**Mode `add-noise` :**
- **`--std <float>`** : Déviation standard du bruit (défaut: 0.05)
  - Plus la valeur est élevée, plus le bruit est fort
  - Valeurs typiques : 0.001 (faible) à 0.1 (fort)

**Mode `filter` :**
- **`--cutoff <float>`** : Fréquence de coupure en Hz (défaut: 3000.0)
  - Les fréquences au-dessus de cette valeur sont atténuées
  - Valeurs typiques : 2000-8000 Hz selon le type d'audio

### Comment ça fonctionne

1. **Ajout de bruit** :
   - Génération de bruit gaussien avec `np.random.normal(0, std_noise)`
   - Addition du bruit au signal audio original
   - Calcul du RMS pour mesurer l'amplitude du signal

2. **Réduction de bruit (filtre passe-bas)** :
   - Utilisation d'un filtre Butterworth d'ordre 4
   - Le filtre passe-bas supprime les hautes fréquences (où le bruit est souvent présent)
   - Utilisation de `scipy.signal.filtfilt()` pour un filtrage bidirectionnel (pas de décalage de phase)
   - Les basses fréquences (contenu audio principal) sont préservées

### Structure des dossiers

```
tp-audio/
├── noisy/          # Fichiers audio bruités et leurs graphiques
└── filtered/       # Fichiers audio filtrés et leurs graphiques
```

---

## Exercice 5c : Transcription Vocale et Extraction de Mots-clés

**Emplacement :** Répertoire `tp-audio/tp3.py`

### Description
Un programme Python pour transcrire un fichier audio en texte en utilisant la reconnaissance vocale, puis utiliser Gemini AI pour identifier automatiquement 3 mots-clés à partir du texte transcrit.

### Fonctionnalités
- Transcription audio en texte avec Google Speech Recognition
- Comptage automatique du nombre de mots dans la transcription
- Extraction de 3 mots-clés avec Gemini AI
- Support des fichiers audio WAV
- Reconnaissance vocale en anglais (en-US)

### Prérequis
- Python 3.7+
- SpeechRecognition (reconnaissance vocale)
- google-generativeai (API Gemini)
- python-dotenv (gestion des variables d'environnement)
- pydub (traitement audio, requis par SpeechRecognition)
- Clé API Google Gemini (voir Configuration)

### Installation
```bash
pip install -r requirements.txt
```

**Note :** Pour utiliser la transcription, vous devez également installer `ffmpeg` (voir Exercice 5).

### Configuration

1. Obtenez votre clé API Gemini depuis [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Créez ou modifiez le fichier `.env` à la racine du projet :
```
GEMINI_API_KEY=votre_clé_api_ici
```

### Utilisation

**Transcrire un fichier audio et extraire les mots-clés :**
```bash
python tp-audio/tp3.py speech.wav
```

### Résultats

Le programme affiche :
- Le texte transcrit depuis l'audio
- Le nombre de mots trouvés dans la transcription
- 3 mots-clés identifiés par Gemini AI

### Exemple de sortie

```
Transcribing audio file: speech.wav
Language: English (en-US)
--------------------------------------------------
Transcribed text:
Hello world this is a test about artificial intelligence
--------------------------------------------------
Number of words found: 8
--------------------------------------------------
Extracting 3 key words using Gemini...
Key words identified:
  1. artificial intelligence
  2. test
  3. hello
```

### Comment ça fonctionne

1. **Transcription** : `speech_recognition` utilise Google Speech Recognition pour convertir l'audio en texte
2. **Ajustement du bruit** : Le programme ajuste automatiquement le bruit ambiant pour améliorer la reconnaissance
3. **Comptage de mots** : Le texte transcrit est divisé en mots et compté
4. **Extraction de mots-clés** : Gemini AI analyse le texte et identifie les 3 mots-clés les plus représentatifs

### Limitations

- La transcription fonctionne uniquement en anglais (en-US)
- Nécessite une connexion Internet pour Google Speech Recognition et Gemini AI
- Les fichiers audio doivent être en format WAV (ou un format supporté par `speech_recognition`)

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
- `scipy` - Calcul scientifique et filtres numériques (Exercice 5b)
- `google-generativeai` - API Gemini AI (Exercice 2, 5c)
- `Pillow` - Traitement d'images
- `python-dotenv` - Gestion des variables d'environnement (Exercice 2, 5c)
- `opencv-python` - Bibliothèque de vision par ordinateur (Exercice 3)
- `librosa` - Traitement et analyse audio (Exercice 5)
- `soundfile` - Lecture/écriture de fichiers audio (Exercice 5)
- `SpeechRecognition` - Reconnaissance vocale (Exercice 5c)
- `pydub` - Traitement audio (Exercice 5c)

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
- **Exercice 5** : Code de visualisation (spectrogramme et diagramme d'amplitude) basé sur [Visualisation de données audio en Python](https://www.kaggle.com/code/ghazouanihaythem/tmm-visualisation-de-donn-es-audio-en-python#%C3%89tape-3.-Spectre-de-Fr%C3%A9quence-/-Spectrogramme) par ghazouanihaythem sur Kaggle
- **Exercice 5b** : Code pour ajouter du bruit gaussien basé sur une solution de [Stack Overflow](https://stackoverflow.com/questions/63898448/add-noise-to-audio-file-and-reconvert-the-noisy-signal-using-librosa-python) par user13832229 (CC BY-SA 4.0)

---
