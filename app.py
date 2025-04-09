import streamlit as st
from PIL import Image
import torch
import os
import numpy as np

MODEL_DIR = 'saved_model'
MODEL_NAME = 'vehicle_fraud_resnet50_pytorch.pth'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Must match the output from train_pipeline.py! Check training logs.
CLASS_NAMES = ['NotFraud', 'Fraud'] 
NUM_CLASSES = len(CLASS_NAMES)


try:
    from modeltrain import IMAGE_SIZE, device
    from predict import load_prediction_model, make_prediction, preprocess_single_image
except ImportError as e:
    st.error(f"Error importing necessary modules: {e}")
    st.stop()


@st.cache_resource
def load_model(model_path, num_classes):
    try:
        model = load_prediction_model(model_path, num_classes)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first by running train_pipeline.py.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None


model = load_model(MODEL_PATH, NUM_CLASSES)


st.title("Vehicle Insurance Fraud Detection (PyTorch)")
st.write("Upload an image of a vehicle to classify it as potentially fraudulent or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if model is None:
    st.warning("Model could not be loaded. Prediction is unavailable.")
    st.stop()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        image_bytes = uploaded_file.getvalue()
        processed_image_tensor = preprocess_single_image(image_bytes, IMAGE_SIZE)

        predicted_class, confidence = make_prediction(model, processed_image_tensor, CLASS_NAMES)

        st.subheader("Prediction Result:")
        
        fraud_class_index = -1
        if "Fraud" in CLASS_NAMES:
            fraud_class_index = CLASS_NAMES.index("Fraud")

        if predicted_class == CLASS_NAMES[fraud_class_index] and fraud_class_index != -1:
             st.error(f"Predicted Class: **{predicted_class}**")
        else:
             st.success(f"Predicted Class: **{predicted_class}**")

        st.metric(label="Confidence", value=f"{confidence:.2%}")


    except Exception as e:
        st.error(f"An error occurred during processing or prediction: {e}")

else:
    st.info("Please upload an image file.")

st.sidebar.info(
    "This is a demonstration using a PyTorch ResNet50 model. "
    
)