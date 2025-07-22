# src/utils.py

import tensorflow as tf
import numpy as np

def load_and_prepare_image(image_path, image_size=256):
    """Load and preprocess a single image."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_size, image_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict_image(model, image_path, class_names):
    """Predict the class of an image."""
    img_array = load_and_prepare_image(image_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence
