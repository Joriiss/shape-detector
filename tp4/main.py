import cv2
import sys
import os

def convert_to_black_white(image_path, output_path=None):
    """
    Convertit une image en noir et blanc (niveaux de gris).
    
    Args:
        image_path: Chemin vers l'image à convertir
        output_path: Chemin de sortie (optionnel, généré automatiquement si non fourni)
    
    Returns:
        Chemin de l'image sauvegardée
    """
    # Charger l'image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Convertir en niveaux de gris (noir et blanc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Générer le chemin de sortie si non fourni
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        extension = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_bw{extension}"
    
    # Sauvegarder l'image en noir et blanc
    cv2.imwrite(output_path, gray)
    
    return output_path


def main():
    """Fonction principale."""
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <chemin_image> [chemin_sortie]")
        print("\nExemple:")
        print("  python main.py image.jpg")
        print("  python main.py image.jpg output_bw.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Convertir l'image
        result_path = convert_to_black_white(image_path, output_path)
        print(f"Image convertie en noir et blanc: {result_path}")
        
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

