import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os

# =========================
# PATHS & CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

print("Using Device:", DEVICE)
print("Dataset Path:", DATA_DIR)


# =========================
# ALBUMENTATIONS TRANSFORMS
# =========================
train_tf = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.04,
        scale_limit=0.10,
        rotate_limit=10,
        p=0.7
    ),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])


# =========================
# CUSTOM DATASET
# =========================
class AlbDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(folder))

        for label in classes:
            class_dir = os.path.join(folder, label)
            for img in os.listdir(class_dir):
                self.paths.append(os.path.join(class_dir, img))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, self.labels[idx]


# =========================
# LOAD DATASETS
# =========================
train_ds = AlbDataset(os.path.join(DATA_DIR, "train"), train_tf)
val_ds   = AlbDataset(os.path.join(DATA_DIR, "val"),   val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))


# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_ds.labels),
    y=train_ds.labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print("Class Weights:", class_weights)


# =========================
# MODEL (RESNET50)
# =========================
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler("cuda")  # AMP


# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast("cuda"):
            out = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Val Acc: {val_acc:.2f}%")


# =========================
# SAVE MODEL
# =========================
os.makedirs(os.path.join(SCRIPT_DIR, "..", "models"), exist_ok=True)
save_path = os.path.join(SCRIPT_DIR, "..", "models", "E6_albumentations.pth")

torch.save(model.state_dict(), save_path)
print("\n✔ Albumentations Training Completed!")
print("✔ Model saved to:", save_path)
