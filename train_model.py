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
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(data_dir, filename)
            label_path = os.path.splitext(image_path)[0] + ".txt"
            with open(label_path, 'r') as f:
                label = f.read().strip()
            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1
            image = Image.open(image_path).convert('L')
            image = image.resize((128, 32))
            image = np.array(image) / 255.0
            images.append(image)
            labels.append(label_map[label])
    images = np.expand_dims(np.array(images), axis=-1)
    labels = np.array(labels)
    return images, labels, label_map

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = 'data'  # Replace with your data directory
    images, labels, label_map = load_data(data_dir)
    input_shape = (32, 128, 1)
    num_classes = len(label_map)
    
    model = create_model(input_shape, num_classes)
    model.fit(images, labels, epochs=10, validation_split=0.2)
    model.save('ocr_model.h5')
    
    # Save the label map to a JSON file
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f)
