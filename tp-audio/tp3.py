import speech_recognition as sr
import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio(input_file):
    # Créer un objet Recognizer
    r = sr.Recognizer()
    
    # Charger le fichier audio
    with sr.AudioFile(input_file) as source:
        # Ajuster le bruit ambiant
        r.adjust_for_ambient_noise(source)
        # Lire le fichier audio
        audio = r.record(source)
    
    try:
        text = r.recognize_google(audio, language='en-US')
        return text
    except sr.UnknownValueError:
        return "Error: Could not understand the audio"
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

def extract_keywords(text, api_key=None):
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        return ["Error: GEMINI_API_KEY not found in .env file"]
    
    if text.startswith("Error"):
        return ["Error: Cannot extract keywords from failed transcription"]
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""Analyze the following transcribed text and identify exactly 3 key words that best summarize or represent the main topics or themes.

Text: {text}

Respond with ONLY the 3 key words, separated by commas. Do not include any explanations, numbers, or additional text. Just the 3 words separated by commas.

Example format: word1, word2, word3"""
        
        response = model.generate_content(prompt)
        keywords_text = response.text.strip()
        
        # Extraire les mots-clés (enlever les numéros de liste si présents)
        keywords = [kw.strip() for kw in keywords_text.replace(',', ',').split(',')]
        keywords = [kw for kw in keywords if kw and not kw[0].isdigit()]
        
        # Prendre les 3 premiers mots-clés valides
        keywords = keywords[:3]
        
        # Si on n'a pas 3 mots-clés, compléter avec des valeurs par défaut
        while len(keywords) < 3:
            keywords.append("N/A")
        
        return keywords[:3]
    
    except Exception as e:
        return [f"Error extracting keywords: {str(e)}"]

def main():
    """Fonction principale."""
    if len(sys.argv) < 2:
        print("Usage: python tp3.py <fichier_audio>")
        print("\nExample:")
        print("  python tp3.py hello.wav")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)
    
    print(f"Transcribing audio file: {input_file}")
    print("Language: English (en-US)")
    print("-" * 50)
    
    text = transcribe_audio(input_file)
    
    print("Transcribed text:")
    print(text)
    print("-" * 50)
    
    # Compter le nombre de mots
    if text and not text.startswith("Error"):
        words = text.split()
        word_count = len(words)
        print(f"Number of words found: {word_count}")
        print("-" * 50)
        
        # Extraire 3 mots-clés avec Gemini
        print("Extracting 3 key words using Gemini...")
        keywords = extract_keywords(text)
        
        if keywords and not keywords[0].startswith("Error"):
            print("Key words identified:")
            for i, keyword in enumerate(keywords, 1):
                print(f"  {i}. {keyword}")
        else:
            print(f"Error: {keywords[0] if keywords else 'Failed to extract keywords'}")
    else:
        print("Number of words found: 0 (transcription failed)")
    
    return text

if __name__ == '__main__':
    main()
