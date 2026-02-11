import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import ImageTk, Image
import torch
import torch.nn as nn
from torchvision import models
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os

MODEL_PATH = "models/E6_albumentations.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Grade 0 (Normal)",
    "Grade 1 (Doubtful)",
    "Grade 2 (Mild)",
    "Grade 3 (Moderate)",
    "Grade 4 (Severe)"
]

# Albumentations Transform
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# -----------------------------------------------------------------------------
# Load AI Model
# -----------------------------------------------------------------------------
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()


# -----------------------------------------------------------------------------
# Grad-CAM Function
# -----------------------------------------------------------------------------
def generate_gradcam(image_path):
    img_org = Image.open(image_path).convert("RGB")
    img_np = np.array(img_org)
    orig_h, orig_w = img_np.shape[:2]

    # Preprocess
    tensor_img = transform(image=img_np)["image"]
    tensor_img = tensor_img.unsqueeze(0).to(DEVICE)

    # Hooks
    feature_maps = []
    gradients = []

    def forward_hook(m, i, o):
        feature_maps.append(o)

    def backward_hook(m, gi, go):
        gradients.append(go[0])

    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward
    with autocast("cuda"):
        logits = model(tensor_img)

    pred_class = logits.argmax().item()

    # Backward
    model.zero_grad()
    logits[0, pred_class].backward(retain_graph=True)

    fmap = feature_maps[0][0].detach().cpu().numpy()
    grads = gradients[0].detach().cpu().numpy()

    weights = grads.mean(axis=(1, 2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (orig_w, orig_h))

    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    save_path = "gradcam_output.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return pred_class, save_path


# -----------------------------------------------------------------------------
# GUI Functions
# -----------------------------------------------------------------------------
def open_image():
    global file_path, img_label
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if file_path:
        img = Image.open(file_path)
        img = img.resize((350, 350))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        result_label.config(text="Image loaded. Click Predict.")


def predict_image():
    global file_path

    if not file_path:
        result_label.config(text="Please select an image first.")
        return

    result_label.config(text="Processing...")

    pred_class, cam_path = generate_gradcam(file_path)

    result_label.config(
        text=f"PREDICTION: {CLASS_NAMES[pred_class]}"
    )

    # Show Grad-CAM
    heat = Image.open(cam_path)
    heat = heat.resize((350, 350))
    heat_tk = ImageTk.PhotoImage(heat)
    heatmap_label.config(image=heat_tk)
    heatmap_label.image = heat_tk


# -----------------------------------------------------------------------------
# GUI Window Setup
# -----------------------------------------------------------------------------
root = tk.Tk()
root.title("Osteoarthritis AI – Knee X-ray Classifier")
root.geometry("800x600")
root.configure(bg="#1e1e1e")

# Title
title_label = Label(
    root,
    text="Osteoarthritis Detection AI",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="#1e1e1e"
)
title_label.pack(pady=10)

# Image preview
img_label = Label(root, bg="#1e1e1e")
img_label.pack(pady=10)

# Buttons
Button(
    root,
    text="Select X-ray",
    command=open_image,
    width=20,
    height=2,
    bg="#353535",
    fg="white",
).pack(pady=5)

Button(
    root,
    text="Predict Severity",
    command=predict_image,
    width=20,
    height=2,
    bg="#007acc",
    fg="white",
).pack(pady=5)

# Result text
result_label = Label(
    root,
    text="Upload an X-ray to begin.",
    font=("Arial", 14),
    fg="white",
    bg="#1e1e1e"
)
result_label.pack(pady=10)

# Heatmap display
heatmap_label = Label(root, bg="#1e1e1e")
heatmap_label.pack(pady=10)

# Start GUI
file_path = None
root.mainloop()
