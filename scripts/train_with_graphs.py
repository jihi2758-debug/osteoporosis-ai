import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

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


train_ds = AlbDataset(os.path.join(DATA_DIR, "train"))
val_ds   = AlbDataset(os.path.join(DATA_DIR, "val"))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)


# =========================
# MODEL
# =========================
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =========================
# LISTS FOR GRAPH
# =========================
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []


# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)


    # ===== VALIDATION =====
    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total

    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | "
          f"Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f}")


# =========================
# PLOTS
# =========================
epochs = range(1, EPOCHS+1)

# Accuracy graph
plt.figure()
plt.plot(epochs, train_acc_list, label="Train Acc")
plt.plot(epochs, val_acc_list, label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig("accuracy_graph.png")

# Loss graph
plt.figure()
plt.plot(epochs, train_loss_list, label="Train Loss")
plt.plot(epochs, val_loss_list, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_graph.png")

plt.show()


# =========================
# SAVE MODEL (new file only)
# =========================
torch.save(model.state_dict(), "new_model_with_graphs.pth")

print("\n✅ accuracy_graph.png saved")
print("✅ loss_graph.png saved")
print("✅ model saved separately (safe)")
