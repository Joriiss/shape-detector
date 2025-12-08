import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

def load_test_results(json_path):
    """Load the correct labels from results.json"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Create a dictionary mapping filename to correct label
    correct_labels = {}
    for shape, filenames in results.items():
        for filename in filenames:
            correct_labels[filename] = shape
    
    return correct_labels

def test_model(model_path, test_dir, results_json_path):
    """Test the model on all test images and compare with correct labels"""
    
    # Load the saved model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load correct labels
    print(f"Loading correct labels from {results_json_path}...")
    correct_labels = load_test_results(results_json_path)
    
    # Get all test images
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    shape_names = ['circle', 'square', 'triangle']
    
    print(f"\nTesting {len(test_images)} images...\n")
    
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        
        # Load and prepare image
        img = Image.open(img_path).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 64, 64, 1)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_shape = shape_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        # Get correct label
        correct_shape = correct_labels.get(img_file, "unknown")
        
        # Check if prediction is correct
        is_correct = (predicted_shape == correct_shape)
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Store result
        results.append({
            'filename': img_file,
            'predicted': predicted_shape,
            'correct': correct_shape,
            'confidence': confidence,
            'correct': is_correct
        })
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{status} {img_file}: predicted={predicted_shape}, correct={correct_shape}, confidence={confidence:.2f}")
    
    # Calculate and print accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"Total images: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Wrong predictions: {total_predictions - correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")
    
    return results, accuracy

if __name__ == '__main__':
    model_path = 'shape_model.h5'
    test_dir = 'data/test'
    results_json_path = 'data/test/results.json'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running main.py")
    elif not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
    elif not os.path.exists(results_json_path):
        print(f"Error: Results file '{results_json_path}' not found!")
    else:
        test_model(model_path, test_dir, results_json_path)