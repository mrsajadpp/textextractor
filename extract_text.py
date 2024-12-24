import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((128, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image

def extract_text_from_image(model, image_path, label_map):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction, axis=1)[0]
    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map_inv[predicted_label]

if __name__ == "__main__":
    model_path = 'ocr_model.h5'  # Replace with your model path
    image_path = 'Anita-Updated-Scan-Mobile-e16589.jpg'  # Replace with your image path
    label_map_path = 'label_map.json'  # Replace with your label map path
    
    model = load_model(model_path)
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    extracted_text = extract_text_from_image(model, image_path, label_map)
    print(extracted_text)
