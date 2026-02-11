import torch
import torch.nn as nn
from torchvision import models
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2
import os

# ==============================================================
# CONFIG
# ==============================================================
MODEL_PATH = "../models/E6_albumentations.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Grade 0 (Normal)",
    "Grade 1 (Doubtful)",
    "Grade 2 (Mild)",
    "Grade 3 (Moderate)",
    "Grade 4 (Severe)"
]

# ==============================================================
# TRANSFORMS (same as E6)
# ==============================================================
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])


# ==============================================================
# LOAD MODEL
# ==============================================================
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ==============================================================
# HOOKS
# ==============================================================
feature_maps = None
gradients = None

def forward_hook(module, inp, out):
    global feature_maps
    feature_maps = out

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# ==============================================================
# GENERATE GRAD-CAM
# ==============================================================
def generate_heatmap(image_path):

    # Load image
    img_org = Image.open(image_path).convert("RGB")
    img_np = np.array(img_org)

    # Keep original resolution for resize
    orig_h, orig_w = img_np.shape[:2]

    # Apply transform
    tensor_img = transform(image=img_np)["image"]
    tensor_img = tensor_img.unsqueeze(0).to(DEVICE)

    # Load model
    model = load_model()

    # Hook the last ResNet block
    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass **WITH gradients**
    with autocast("cuda"):
        logits = model(tensor_img)

    pred_class = logits.argmax().item()

    # Backward pass
    model.zero_grad()
    logits[0, pred_class].backward(retain_graph=True)

    # Extract maps
    fmap = feature_maps[0].detach().cpu().numpy()
    grads = gradients.detach().cpu().numpy()

    # Compute weights
    weights = grads.mean(axis=(1, 2))

    # Generate CAM
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w_ in enumerate(weights):
        cam += w_ * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize using ORIGINAL image size
    cam = cv2.resize(cam, (orig_w, orig_h))

    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Save
    save_path = "gradcam_output.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("\n==============================")
    print("Prediction:", CLASS_NAMES[pred_class])
    print("Saved → gradcam_output.jpg")
    print("==============================")

    return save_path


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print("Enter image path:")
    path = input("> ").strip('"')

    if not os.path.exists(path):
        print("❌ ERROR: File not found:", path)
    else:
        generate_heatmap(path)
