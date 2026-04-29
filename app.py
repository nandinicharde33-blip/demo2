import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os

# Load the saved Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('image_classification_model.h5')
    return model

# Load the label encoder
@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

model = load_model()
label_encoder = load_label_encoder()
class_names = label_encoder.classes_

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((250, 250))  # Resize to model's expected input size
    img_array = np.array(img)

    # Ensure image has 3 channels (RGB). Convert if grayscale or has alpha channel.
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    img_array = img_array.astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app layout
st.title('Image Classification App')
st.write('Upload an image to get a prediction!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = predictions[0][predicted_class_index] * 100

    st.success(f'Prediction: {predicted_class_label}')
    st.info(f'Confidence: {confidence:.2f}%')

    # Display all class probabilities (optional)
    st.write("--- All Probabilities ---")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
