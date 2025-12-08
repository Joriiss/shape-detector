import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def create_model():
    # initialisation du modèle 
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax') 
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# chargement des données
def load_data(data_dir):
    shapes = ['circle', 'square', 'triangle']
    images = []
    labels = []
    
    for label_idx, shape in enumerate(shapes):
        shape_dir = os.path.join(data_dir, shape)
        if not os.path.exists(shape_dir):
            continue
        
        print(f"loading {shape} images...")
        image_files = [f for f in os.listdir(shape_dir) if f.endswith('.png')]
        
        for img_file in image_files:
            img_path = os.path.join(shape_dir, img_file)
            try:
                img = Image.open(img_path).convert('L')  # conversion en noir et blanc
                img = img.resize((64, 64)) # redimensionne en 64x64
                img_array = np.array(img) / 255.0  
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                print(f"error loading {img_path}: {e}")
                continue
    
    return np.array(images), np.array(labels)


def main():
    data_dir = 'data'
    
    print("loading data...")
    images, labels = load_data(data_dir)
    
    images = images.reshape(-1, 64, 64, 1)
    
    print(f"loaded {len(images)} images")
    print(f"image shape: {images.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    print("\ncreating model...")
    model = create_model()
    model.summary()
    
    print("\ntraining model...")
    history = model.fit(X_train, y_train,
                       epochs=10,
                       batch_size=32,
                       validation_data=(X_val, y_val),
                       verbose=1)
    
    model.save('shape_model.h5')
    print("\nmodel saved as 'shape_model.h5'")
    
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        shape_names = ['circle', 'square', 'triangle']
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        
        print("\ntesting on test images:")
        for img_file in test_images[:10]:
            img_path = os.path.join(test_dir, img_file)
            img = Image.open(img_path).convert('L')
            img = img.resize((64, 64))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 64, 64, 1)
            
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            print(f"{img_file}: {shape_names[predicted_class]} ({confidence:.2f})")

if __name__ == '__main__':
    main()