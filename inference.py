import torch
import torch.nn as nn
from torchvision import models
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2
import os

# =============== CONFIG ===============
MODEL_PATH = "models/E6_albumentations.pth"   # your trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Grade 0 (Normal)",
    "Grade 1 (Doubtful)",
    "Grade 2 (Mild)",
    "Grade 3 (Moderate)",
    "Grade 4 (Severe)"
]

# =============== TRANSFORM (same as training) ===============
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])


# =============== LOAD MODEL ===============
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("✔ Model loaded:", MODEL_PATH)
    return model


# =============== PREPROCESS IMAGE ===============
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    transformed = transform(image=img)["image"]
    transformed = transformed.unsqueeze(0)  # add batch dim

    return transformed.to(DEVICE)


# =============== INFERENCE FUNCTION ===============
def predict(img_path):
    model = load_model()

    img_tensor = preprocess_image(img_path)

    with torch.no_grad():
        with autocast("cuda"):
            output = model(img_tensor)

    probs = torch.softmax(output, dim=1)[0]
    pred_class = torch.argmax(probs).item()

    # Print prediction summary
    print("\n=== Prediction Result ===")
    print("Image:", img_path)
    print("Predicted Class:", CLASS_NAMES[pred_class])
    print("\nConfidence Scores:")
    for i, score in enumerate(probs):
        print(f"{CLASS_NAMES[i]}: {score:.4f}")

    return pred_class, probs


# =============== MAIN (Run directly) ===============
if __name__ == "__main__":
    print("Enter image path:")
    path = input("> ")

    if not os.path.exists(path):
        print("❌ ERROR: File not found:", path)
    else:
        predict(path)
