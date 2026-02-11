import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
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

print("Using device:", DEVICE)


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
# COLLECT PROBABILITIES
# =========================
all_probs = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)

        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

all_probs = np.vstack(all_probs)
all_labels = np.hstack(all_labels)


# =========================
# BINARIZE LABELS
# =========================
y_true = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))


# =========================
# ROC CURVE
# =========================
plt.figure(figsize=(7,6))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

# random baseline
plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multi-class)")
plt.legend()
plt.grid()

plt.savefig("roc_auc_curve.png")
plt.show()

print("✅ roc_auc_curve.png saved")
print("✅ Model not modified (evaluation only)")
