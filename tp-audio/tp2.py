import librosa
import soundfile as sf
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
from scipy import signal

def add_noise(input_file, std_noise=0.05, output_file=None):
    """
    Ajoute du bruit gaussien (bruit blanc) à un fichier audio.
    
    Args:
        input_file: Chemin vers le fichier audio d'entrée
        std_noise: Déviation standard du bruit (défaut: 0.05)
        output_file: Chemin du fichier de sortie (optionnel)
    
    Returns:
        Chemin du fichier sauvegardé
    """
    # Créer le dossier noisy s'il n'existe pas
    script_dir = os.path.dirname(os.path.abspath(input_file)) if os.path.dirname(input_file) else os.getcwd()
    noisy_dir = os.path.join(script_dir, 'noisy')
    os.makedirs(noisy_dir, exist_ok=True)
    
    # Charger le signal audio
    signal, sr = librosa.load(input_file, sr=None, mono=False)
    
    # Calculer le RMS (Root Mean Square) du signal original
    if signal.ndim == 1:
        # Mono
        RMS = math.sqrt(np.mean(signal**2))
        # Générer du bruit gaussien
        noise = np.random.normal(0, std_noise, signal.shape[0])
        # Ajouter le bruit au signal
        signal_noise = signal + noise
    else:
        # Stéréo
        RMS = math.sqrt(np.mean(signal**2))
        # Générer du bruit pour chaque canal
        noise = np.random.normal(0, std_noise, signal.shape)
        # Ajouter le bruit au signal
        signal_noise = signal + noise
    
    # Générer le nom de fichier de sortie dans le dossier noisy
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(noisy_dir, f"{base_name}_noisy.wav")
    else:
        # Si un chemin de sortie est fourni, s'assurer qu'il est dans le dossier noisy
        if not os.path.isabs(output_file):
            output_file = os.path.join(noisy_dir, output_file)
    
    # Sauvegarder le fichier audio bruité
    if signal_noise.ndim == 1:
        sf.write(output_file, signal_noise, sr)
    else:
        sf.write(output_file, signal_noise.T, sr)
    
    print(f"✅ Fichier audio avec bruit sauvegardé: {output_file}")
    print(f"RMS du signal original: {RMS:.6f}")
    print(f"Déviation standard du bruit: {std_noise}")
    print(f"Forme du signal: {signal.shape}")
    print(f"Forme du signal bruité: {signal_noise.shape}")
    
    # Générer les graphiques
    generate_plots(signal_noise, sr, noisy_dir, base_name)
    
    return output_file

def generate_plots(signal_array, sample_freq, output_dir, base_name):
    """
    Génère les graphiques d'amplitude et spectrogrammes pour le signal audio.
    
    Args:
        signal_array: Tableau numpy du signal audio
        sample_freq: Fréquence d'échantillonnage
        output_dir: Dossier de sortie pour les graphiques
        base_name: Nom de base pour les fichiers
    """
    # Convertir en int16 pour les graphiques (comme dans tp1.py)
    if signal_array.ndim == 1:
        n_channels = 1
        n_samples = len(signal_array)
        signal_int16 = (signal_array * 32767).astype(np.int16)
    else:
        n_channels = signal_array.shape[0]
        n_samples = signal_array.shape[1]
        signal_int16 = (signal_array * 32767).astype(np.int16)
    
    # Calculer la durée
    t_audio = n_samples / sample_freq
    
    # Diviser le signal en deux canaux : canal gauche (left) et canal droit (right)
    if n_channels == 2:
        # Si stéréo, librosa retourne (2, n_samples), on récupère les canaux
        l_channel = signal_int16[0]
        r_channel = signal_int16[1]
    else:
        # Si mono, on duplique le signal pour les deux canaux
        l_channel = signal_int16
        r_channel = signal_int16.copy()
    
    # Générer des instants temporels (timestamps) pour chaque échantillon audio
    timestamps = np.linspace(0, n_samples / sample_freq, num=n_samples)
    
    # Créer un graphique pour le canal gauche (left channel)
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, l_channel)
    plt.title('Canal Gauche (avec bruit)')
    plt.ylabel('Valeur du Signal')
    plt.xlabel('Temps (s)')
    plt.xlim(0, t_audio)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{base_name}_noisy_canal_gauche.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphique du canal gauche sauvegardé: {plot_path}")
    
    # Créer un graphique pour le canal droit (right channel)
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, r_channel)
    plt.title('Canal Droit (avec bruit)')
    plt.ylabel('Valeur du Signal')
    plt.xlabel('Temps (s)')
    plt.xlim(0, t_audio)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{base_name}_noisy_canal_droit.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphique du canal droit sauvegardé: {plot_path}")
    
    # Créer un graphique pour le spectrogramme du canal gauche (left channel)
    plt.figure(figsize=(10, 5))
    plt.specgram(l_channel, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Spectrogramme - Canal Gauche (avec bruit)')
    plt.ylabel('Fréquence (Hz)')
    plt.xlabel('Temps (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{base_name}_noisy_spectrogramme_canal_gauche.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spectrogramme du canal gauche sauvegardé: {plot_path}")
    
    # Répéter les mêmes étapes pour le canal droit (right channel)
    plt.figure(figsize=(10, 5))
    plt.specgram(r_channel, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Spectrogramme - Canal Droit (avec bruit)')
    plt.ylabel('Fréquence (Hz)')
    plt.xlabel('Temps (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{base_name}_noisy_spectrogramme_canal_droit.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spectrogramme du canal droit sauvegardé: {plot_path}")

def apply_lowpass_filter(input_file, cutoff_freq=4000.0, output_file=None):
    """
    Applique un filtre passe-bas à un fichier audio pour réduire le bruit.
    
    Args:
        input_file: Chemin vers le fichier audio bruité
        cutoff_freq: Fréquence de coupure en Hz (défaut: 4000.0)
        output_file: Chemin du fichier de sortie (optionnel)
    
    Returns:
        Chemin du fichier filtré
    """
    # Créer le dossier noisy s'il n'existe pas
    script_dir = os.path.dirname(os.path.abspath(input_file)) if os.path.dirname(input_file) else os.getcwd()
    noisy_dir = os.path.join(script_dir, 'noisy')
    os.makedirs(noisy_dir, exist_ok=True)
    
    # Charger l'audio bruité
    noisy_signal, sample_freq = librosa.load(input_file, sr=None, mono=False)
    
    # Créer le filtre passe-bas Butterworth
    nyquist = sample_freq / 2.0
    normal_cutoff = cutoff_freq / nyquist
    
    # Vérifier que la fréquence de coupure est valide
    if normal_cutoff >= 1.0:
        print(f"Attention: La fréquence de coupure ({cutoff_freq} Hz) est trop élevée pour la fréquence d'échantillonnage ({sample_freq} Hz).")
        print(f"Utilisation de {nyquist * 0.95:.1f} Hz comme fréquence de coupure maximale.")
        normal_cutoff = 0.95
    
    # Créer le filtre Butterworth d'ordre 4
    b, a = signal.butter(4, normal_cutoff, btype='low')
    
    # Appliquer le filtre
    if noisy_signal.ndim == 1:
        # Mono
        filtered_signal = signal.filtfilt(b, a, noisy_signal)
    else:
        # Stéréo - filtrer chaque canal séparément
        filtered_signal = np.array([
            signal.filtfilt(b, a, noisy_signal[0]),
            signal.filtfilt(b, a, noisy_signal[1])
        ])
    
    # Générer le nom de fichier de sortie
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # Enlever "_noisy" du nom si présent
        if base_name.endswith('_noisy'):
            base_name = base_name[:-6]
        output_file = os.path.join(noisy_dir, f"{base_name}_filtered_lowpass_{int(cutoff_freq)}Hz.wav")
    else:
        if not os.path.isabs(output_file):
            output_file = os.path.join(noisy_dir, output_file)
    
    # Sauvegarder le fichier filtré
    if filtered_signal.ndim == 1:
        sf.write(output_file, filtered_signal, sample_freq)
    else:
        sf.write(output_file, filtered_signal.T, sample_freq)
    
    print(f"✅ Fichier audio filtré (passe-bas) sauvegardé: {output_file}")
    print(f"Fréquence de coupure: {cutoff_freq} Hz")
    print(f"Fréquence d'échantillonnage: {sample_freq} Hz")
    print(f"Forme du signal filtré: {filtered_signal.shape}")
    
    # Générer les graphiques pour le signal filtré
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    generate_plots(filtered_signal, sample_freq, noisy_dir, base_name)
    
    return output_file

def main():
    """Fonction principale."""
    if len(sys.argv) < 3:
        print("Usage: python tp2.py <mode> <fichier_audio> [options]")
        print("\nModes:")
        print("  add-noise    Ajouter du bruit gaussien au fichier audio")
        print("  filter       Appliquer un filtre passe-bas pour réduire le bruit")
        print("\nOptions pour add-noise:")
        print("  --std <float>     Déviation standard du bruit (défaut: 0.05)")
        print("  --output <file>   Fichier de sortie (optionnel)")
        print("\nOptions pour filter:")
        print("  --cutoff <float>  Fréquence de coupure en Hz (défaut: 4000.0)")
        print("  --output <file>   Fichier de sortie (optionnel)")
        print("\nExemples:")
        print("  python tp2.py add-noise hello.mp3 --std 0.01")
        print("  python tp2.py filter noisy/hello_noisy.wav --cutoff 3000")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    input_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Erreur: Le fichier '{input_file}' n'existe pas.")
        sys.exit(1)
    
    if mode == 'add-noise':
        std_noise = 0.05
        output_file = None
        
        # Parser les arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--std' and i + 1 < len(sys.argv):
                std_noise = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        add_noise(input_file, std_noise, output_file)
    
    elif mode == 'filter':
        cutoff_freq = 4000.0
        output_file = None
        
        # Parser les arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--cutoff' and i + 1 < len(sys.argv):
                cutoff_freq = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        apply_lowpass_filter(input_file, cutoff_freq, output_file)
    
    else:
        print(f"Erreur: Mode invalide '{mode}'. Utilisez 'add-noise' ou 'filter'.")
        sys.exit(1)

if __name__ == '__main__':
    main()

