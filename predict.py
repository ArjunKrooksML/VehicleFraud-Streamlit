import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import io

try:
    from modeltrain import IMAGE_SIZE, device, build_model, preprocess_single_image
except ImportError:
     print("Make sure train_pipeline.py is in the same directory or Python path.")
     IMAGE_SIZE = 224
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     def build_model(num_classes): raise NotImplementedError("build_model not imported")
     def preprocess_single_image(img, size): raise NotImplementedError("preprocess not imported")


def load_prediction_model(model_path, num_classes):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = build_model(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        raise

def make_prediction(model, processed_image_tensor, class_names):
    if model is None:
        raise ValueError("Model is not loaded.")
    if processed_image_tensor is None:
         raise ValueError("Processed image tensor is None.")

    model.eval()
    with torch.no_grad():
        outputs = model(processed_image_tensor.to(device))

        if len(class_names) == 2:
            probs = torch.sigmoid(outputs)
            score = probs.item()
            threshold = 0.5
            if score >= threshold:
                predicted_class_index = 1
                confidence = score
            else:
                predicted_class_index = 0
                confidence = 1 - score
        else:
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_class_index_tensor = torch.max(probs, 1)
            predicted_class_index = predicted_class_index_tensor.item()
            confidence = confidence.item()


    if class_names and len(class_names) > predicted_class_index:
        predicted_class_name = class_names[predicted_class_index]
    else:
        predicted_class_name = f"Class {predicted_class_index}"

    print(f"Raw output score/probs: {outputs.cpu().numpy()}")
    print(f"Predicted class: {predicted_class_name}, Confidence: {confidence:.4f}")

    return predicted_class_name, confidence