import numpy as np
import matplotlib.pyplot as plt
import librosa

# Ouvrir le fichier audio MP3
signal_array, sample_freq = librosa.load('hello.mp3', sr=None, mono=False)

# Si le fichier est stéréo, signal_array sera de shape (2, n_samples)
# Si mono, signal_array sera de shape (n_samples,)
if signal_array.ndim == 1:
    n_channels = 1
    n_samples = len(signal_array)
    # Convertir en int16 (librosa charge en float32 normalisé [-1, 1])
    signal_array = (signal_array * 32767).astype(np.int16)
else:
    n_channels = signal_array.shape[0]
    n_samples = signal_array.shape[1]
    # Convertir en int16 pour chaque canal
    signal_array = (signal_array * 32767).astype(np.int16)

# Calculer la durée
t_audio = n_samples / sample_freq

# Afficher des informations de base sur le fichier audio
print(f"Fichier audio chargé: {n_samples} échantillons, {sample_freq} Hz, {n_channels} canal(aux)")

# Afficher les informations extraites
print("Le taux d'échantillonnage du fichier audio est de " + str(sample_freq) + "Hz, soit " + str(sample_freq / 1000) + "kHz")
print("L'audio contient un total de " + str(n_samples) + " images ou échantillons")
print("La durée du fichier audio est de " + str(t_audio) + " secondes")
print("Le fichier audio comporte " + str(n_channels) + " canaux.\n")

# Afficher le nombre total d'échantillons dans le signal
total_samples = signal_array.size
print("Le signal contient un total de " + str(total_samples) + " échantillons.")
print("Si cette valeur est supérieure à " + str(n_samples) + ", c'est en raison de la présence de plusieurs canaux.")
print("Par exemple, Échantillons * Canaux = " + str(n_samples * n_channels))

# Diviser le signal en deux canaux : canal gauche (left) et canal droit (right)
if n_channels == 2:
    # Si stéréo, librosa retourne (2, n_samples), on récupère les canaux
    l_channel = signal_array[0]
    r_channel = signal_array[1]
else:
    # Si mono, on duplique le signal pour les deux canaux
    l_channel = signal_array
    r_channel = signal_array.copy()

# Générer des instants temporels (timestamps) pour chaque échantillon audio
timestamps = np.linspace(0, n_samples / sample_freq, num=n_samples)

# Créer un graphique pour le canal gauche (left channel)
plt.figure(figsize=(10, 5))
plt.plot(timestamps, l_channel)
plt.title('Canal Gauche')
plt.ylabel('Valeur du Signal')
plt.xlabel('Temps (s)')
plt.xlim(0, t_audio)
plt.tight_layout()
plt.savefig('canal_gauche.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graphique du canal gauche sauvegardé: canal_gauche.png")

# Créer un graphique pour le canal droit (right channel)
plt.figure(figsize=(10, 5))
plt.plot(timestamps, r_channel)
plt.title('Canal Droit')
plt.ylabel('Valeur du Signal')
plt.xlabel('Temps (s)')
plt.xlim(0, t_audio)
plt.tight_layout()
plt.savefig('canal_droit.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graphique du canal droit sauvegardé: canal_droit.png")

# Créer un graphique pour le spectrogramme du canal gauche (left channel)
plt.figure(figsize=(10, 5))
# Générer le spectrogramme du canal gauche (l_channel) avec les paramètres spécifiés
plt.specgram(l_channel, Fs=sample_freq, vmin=-20, vmax=50)
# Ajouter un titre au graphique
plt.title('Spectrogramme - Canal Gauche')
# Étiqueter l'axe des y avec "Fréquence (Hz)"
plt.ylabel('Fréquence (Hz)')
# Étiqueter l'axe des x avec "Temps (s)"
plt.xlabel('Temps (s)')
# Définir les limites de l'axe des x pour afficher uniquement la partie de l'audio correspondant à sa durée
plt.xlim(0, t_audio)
# Ajouter une barre de couleur pour indiquer l'intensité du spectre
plt.colorbar()
plt.tight_layout()
plt.savefig('spectrogramme_canal_gauche.png', dpi=150, bbox_inches='tight')
plt.close()
print("Spectrogramme du canal gauche sauvegardé: spectrogramme_canal_gauche.png")

# Répéter les mêmes étapes pour le canal droit (right channel)
plt.figure(figsize=(10, 5))
plt.specgram(r_channel, Fs=sample_freq, vmin=-20, vmax=50)
plt.title('Spectrogramme - Canal Droit')
plt.ylabel('Fréquence (Hz)')
plt.xlabel('Temps (s)')
plt.xlim(0, t_audio)
plt.colorbar()
plt.tight_layout()
plt.savefig('spectrogramme_canal_droit.png', dpi=150, bbox_inches='tight')
plt.close()
print("Spectrogramme du canal droit sauvegardé: spectrogramme_canal_droit.png")