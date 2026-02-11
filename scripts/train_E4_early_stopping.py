import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import autocast, GradScaler
import numpy as np
import os

# =========================
# PATHS & CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 40            # allow long training, early stop will handle it
LR = 1e-4
PATIENCE = 4           # stop if no improvement for 4 epochs

print("Using Device:", DEVICE)
print("Dataset Path:", DATA_DIR)


# =========================
# TRANSFORMS (RGB)
# =========================
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


# =========================
# DATASETS
# =========================
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))


# =========================
# CLASS WEIGHTS
# =========================
targets = train_ds.targets
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(targets),
    y=targets
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print("Class weights:", class_weights)


# =========================
# MODEL
# =========================
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scaler = GradScaler("cuda")


# =========================
# EARLY STOPPING VARIABLES
# =========================
best_acc = 0.0
wait = 0  # count epochs without improvement


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
    # EARLY STOPPING LOGIC
    # =========================
    if val_acc > best_acc:
        best_acc = val_acc
        wait = 0

        # Save best model
        best_path = os.path.join(SCRIPT_DIR, "..", "models", "E4_best_model.pth")
        torch.save(model.state_dict(), best_path)
        print("✔ New Best Model Saved!")
        print("Best Accuracy:", best_acc)

    else:
        wait += 1
        print(f"No improvement. Patience step {wait}/{PATIENCE}")

        if wait >= PATIENCE:
            print("⛔ Early Stopping Triggered!")
            break


# =========================
# SAVE FINAL MODEL
# =========================
final_path = os.path.join(SCRIPT_DIR, "..", "models", "E4_final_model.pth")
torch.save(model.state_dict(), final_path)

print("\n✔ Training Finished!")
print("✔ Best model saved to:", best_path)
print("✔ Final model saved to:", final_path)
