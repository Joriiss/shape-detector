# Projet Data Science & IA

Ce repo contient deux exercices axés sur la reconnaissance d'images et l'analyse d'images par IA.

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

## Installation (Toutes les Dépendances)

Installez tous les packages requis pour les deux exercices :

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

---

## Notes

- Pour l'Exercice 2, assurez-vous que votre fichier `.env` se trouve dans le répertoire racine du projet
- Le fichier `.env` doit être ajouté à `.gitignore` pour garder votre clé API sécurisée
- L'Exercice 1 nécessite des données d'entraînement dans la structure de répertoire `data/`
- Les deux exercices prennent en charge les formats d'image courants (PNG, JPEG, etc.)

---
