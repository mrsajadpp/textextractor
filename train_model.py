import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
import json

def load_data(data_dir):
    images = []
    labels = []
    label_map = {}
    label_counter = 0
    max_label_length = 0
    
    # First pass to get max length and build label map
    for filename in os.listdir(data_dir):
        if filename.endswith((".jpg", ".png")):
            label_path = os.path.splitext(os.path.join(data_dir, filename))[0] + ".txt"
            with open(label_path, 'r') as f:
                label = f.read().strip()
                max_label_length = max(max_label_length, len(label))
                for char in label:
                    if char not in label_map:
                        label_map[char] = label_counter
                        label_counter += 1
    
    # Second pass to load and pad data
    for filename in os.listdir(data_dir):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(data_dir, filename)
            label_path = os.path.splitext(image_path)[0] + ".txt"
            
            with open(label_path, 'r') as f:
                label = f.read().strip()
            
            # Pad labels
            label_encoded = [label_map[char] for char in label]
            label_encoded.extend([0] * (max_label_length - len(label_encoded)))
            
            image = Image.open(image_path).convert('L')
            image = image.resize((128, 32))
            image = np.array(image) / 255.0
            
            images.append(image)
            labels.append(label_encoded)
    
    return np.expand_dims(np.array(images), axis=-1), np.array(labels), label_map

def create_model(input_shape, num_classes, max_text_length):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Reshape((-1, 64)),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dense(max_text_length * num_classes),
        layers.Reshape((max_text_length, num_classes)),
        layers.Activation('softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = 'newdata'  # Replace with your data directory
    images, labels, label_map = load_data(data_dir)
    input_shape = (32, 128, 1)
    num_classes = len(label_map)

    max_text_length = labels.shape[1]
    model = create_model(input_shape, num_classes, max_text_length)
    model.fit(images, labels, epochs=10, validation_split=0.2)
    model.save('ocr_model.h5')
    
    # Save the label map to a JSON file
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f)
