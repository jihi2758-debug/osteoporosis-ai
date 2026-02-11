import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report
import numpy as np
from PIL import Image
import os


# =========================
# PATHS
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "E6_albumentations.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
BATCH_SIZE = 16


# =========================
# TRANSFORMS
# =========================
tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])


# =========================
# DATASET
# =========================
class AlbDataset(Dataset):
    def __init__(self, folder):
        self.paths = []
        self.labels = []

        for label in sorted(os.listdir(folder)):
            class_dir = os.path.join(folder, label)
            for img in os.listdir(class_dir):
                self.paths.append(os.path.join(class_dir, img))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        img = tf(image=img)["image"]
        return img, self.labels[idx]


# =========================
# LOAD DATA
# =========================
val_ds = AlbDataset(os.path.join(DATA_DIR, "val"))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# LOAD MODEL
# =========================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# =========================
# GET PREDICTIONS
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(
    all_labels,
    all_preds,
    target_names=[f"Class {i}" for i in range(NUM_CLASSES)]
)

print("\n===== CLASSIFICATION REPORT =====\n")
print(report)


# save to file
with open("classification_report.txt", "w") as f:
    f.write(report)

print("✅ classification_report.txt saved")
print("✅ Model not modified (evaluation only)")
