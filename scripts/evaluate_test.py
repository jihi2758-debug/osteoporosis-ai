# scripts/evaluate_test.py
import torch, os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "model.pth")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "test")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH = 32

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

ds = datasets.ImageFolder(DATA_DIR, transform=tf)
loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        preds = out.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true); y_pred = np.array(y_pred)
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))
print("\nWeighted Cohen kappa:", cohen_kappa_score(y_true, y_pred, weights="quadratic"))
