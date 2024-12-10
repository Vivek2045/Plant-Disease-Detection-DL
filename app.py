import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Define the class names (Replace with your own class names)
class_names = ['healthy', 'unhealthy']

# Streamlit web app
st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf, and the model will predict the disease.")

# Image upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image file
    image = Image.open(uploaded_image)
    
    # Preprocess the image
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display the image and prediction result
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Disease: {predicted_class} with confidence {np.max(prediction)*100:.2f}%")
