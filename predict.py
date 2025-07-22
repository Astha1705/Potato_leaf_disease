# src/predict.py

import os
import tensorflow as tf
from utils import predict_image
import sys

# Configuration
MODEL_PATH = "models/model_v1.keras"
IMAGE_PATH = "sample_leaf.jpg"  # Replace this with your image path

# Load model
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names from training data structure
DATA_DIR = "data/PlantVillage"
class_names = sorted(os.listdir(DATA_DIR))

# Predict
predicted_class, confidence = predict_image(model, IMAGE_PATH, class_names)

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence}%")
