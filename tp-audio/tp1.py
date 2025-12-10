import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import sys
import os

def change_audio_speed(input_file, speed_factor, output_file=None):
    """
    Modifie la vitesse de l'audio sans changer la hauteur.
    
    Args:
        input_file: Chemin vers le fichier audio d'entrée
        speed_factor: Facteur de vitesse (1.0 = normal, 2.0 = 2x plus rapide, 0.5 = 2x plus lent)
        output_file: Chemin du fichier de sortie (optionnel)
    
    Returns:
        Chemin du fichier sauvegardé
    """
    # Charger l'audio
    signal_array, sample_freq = librosa.load(input_file, sr=None, mono=False)
    
    # Modifier la vitesse avec time_stretch (préserve la hauteur)
    if signal_array.ndim == 1:
        # Mono
        signal_stretched = librosa.effects.time_stretch(signal_array, rate=speed_factor)
    else:
        # Stéréo - traiter chaque canal séparément
        signal_stretched = np.array([
            librosa.effects.time_stretch(signal_array[0], rate=speed_factor),
            librosa.effects.time_stretch(signal_array[1], rate=speed_factor)
        ])
    
    # Générer le nom de fichier de sortie
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        extension = '.wav'  # Sauvegarder en WAV
        output_file = f"{base_name}_speed_{speed_factor}x{extension}"
    
    # Sauvegarder le fichier
    if signal_stretched.ndim == 1:
        sf.write(output_file, signal_stretched, sample_freq)
    else:
        # Transposer pour soundfile (canaux en colonnes)
        sf.write(output_file, signal_stretched.T, sample_freq)
    
    return output_file

def remove_silence(input_file, top_db=20, output_file=None):
    """
    Enlève les silences au début et à la fin de l'audio.
    
    Args:
        input_file: Chemin vers le fichier audio d'entrée
        top_db: Seuil en dB pour détecter le silence (défaut: 20)
        output_file: Chemin du fichier de sortie (optionnel)
    
    Returns:
        Chemin du fichier sauvegardé
    """
    # Charger l'audio
    signal_array, sample_freq = librosa.load(input_file, sr=None, mono=False)
    
    # Enlever les silences
    if signal_array.ndim == 1:
        # Mono
        signal_cropped, _ = librosa.effects.trim(signal_array, top_db=top_db)
    else:
        # Stéréo - traiter chaque canal séparément
        signal_cropped_left, _ = librosa.effects.trim(signal_array[0], top_db=top_db)
        signal_cropped_right, _ = librosa.effects.trim(signal_array[1], top_db=top_db)
        # Trouver la longueur minimale pour garder les canaux synchronisés
        min_len = min(len(signal_cropped_left), len(signal_cropped_right))
        signal_cropped = np.array([
            signal_cropped_left[:min_len],
            signal_cropped_right[:min_len]
        ])
    
    # Générer le nom de fichier de sortie
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        extension = '.wav'
        output_file = f"{base_name}_no_silence{extension}"
    
    # Sauvegarder le fichier
    if signal_cropped.ndim == 1:
        sf.write(output_file, signal_cropped, sample_freq)
    else:
        # Transposer pour soundfile (canaux en colonnes)
        sf.write(output_file, signal_cropped.T, sample_freq)
    
    return output_file

def main():
    """Fonction principale."""
    # Parser les arguments
    input_file = 'hello.mp3'
    speed_factor = None
    remove_silence_flag = False
    top_db = 20
    
    # Parser les arguments de ligne de commande
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--speed' and i + 1 < len(sys.argv):
            speed_factor = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--remove-silence':
            remove_silence_flag = True
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                try:
                    top_db = float(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    i += 1
            else:
                i += 1
        elif not sys.argv[i].startswith('--'):
            input_file = sys.argv[i]
            i += 1
        else:
            i += 1
    
    # Si l'option remove-silence est activée, enlever les silences
    if remove_silence_flag:
        print(f"Suppression des silences (seuil: {top_db} dB)...")
        output_file = remove_silence(input_file, top_db=top_db)
        print(f"Fichier audio sans silence sauvegardé: {output_file}")
        # Utiliser le fichier modifié pour le reste du traitement
        input_file = output_file
    
    # Si un facteur de vitesse est spécifié, modifier la vitesse et sauvegarder
    if speed_factor is not None:
        print(f"Modification de la vitesse de l'audio avec un facteur de {speed_factor}x...")
        output_file = change_audio_speed(input_file, speed_factor)
        print(f"Fichier audio modifié sauvegardé: {output_file}")
        # Utiliser le fichier modifié pour le reste du traitement
        input_file = output_file
    
    # Ouvrir le fichier audio
    signal_array, sample_freq = librosa.load(input_file, sr=None, mono=False)

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

if __name__ == '__main__':
    main()