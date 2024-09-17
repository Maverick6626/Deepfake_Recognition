import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import os
import warnings

# ignore warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model and predict function
model = keras.models.load_model("my_model.h5")

def predict_image(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_input = img_resized.reshape((1, 256, 256, 3))

    prediction = model.predict(img_input)[0][0]
    label = 'Fake' if prediction == 0 else 'Real'
    return label


# Web side frontend and backend
st.title("Deepfake Recognition")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict_image(image)
    st.write(f"Prediction: {label}")
