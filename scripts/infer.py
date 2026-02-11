# scripts/infer.py
import torch, os
from torchvision import transforms, models
from PIL import Image
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# load model architecture
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # change if not 5 classes
model = model.to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(out.argmax(1).cpu().numpy()[0])
    return pred, probs

if __name__ == "__main__":
    # change this to a file you want to test
    sample = os.path.join(SCRIPT_DIR, "..", "data", "test", "0", os.listdir(os.path.join(SCRIPT_DIR,"..","data","test","0"))[0])
    p, probs = predict(sample)
    print("pred:", p, "probs:", probs)
