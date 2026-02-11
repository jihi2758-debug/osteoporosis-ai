import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os

# ------------------------------
# CONFIG
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../data"     # train/val/test folders are inside this
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 5          # folders 0,1,2,3,4

# ------------------------------
# TRANSFORMS (FOR RGB IMAGES)
# ------------------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ------------------------------
# LOAD DATA
# ------------------------------
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# MODEL (ResNet18 for RGB)
# ------------------------------
model = models.resnet18(weights='IMAGENET1K_V1')

# ❗ IMPORTANT: DO NOT modify conv1 — this keeps 3-channel RGB support
# model.conv1 = ...

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------
# TRAINING LOOP
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # --------------------------
    # VALIDATION
    # --------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.2f}%")

# ------------------------------
# SAVE MODEL
# ------------------------------
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/model.pth")
print("🎉 Training complete! Model saved as models/model.pth")
