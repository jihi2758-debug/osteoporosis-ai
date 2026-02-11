# scripts/gradcam.py
import torch, os, cv2, numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

def get_heatmap(model, img_tensor, target_layer):
    activations = None; grads = None
    def forward_hook(module, inp, out):
        nonlocal activations; activations = out.detach()
    def backward_hook(module, grad_in, grad_out):
        nonlocal grads; grads = grad_out[0].detach()
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    out = model(img_tensor.unsqueeze(0).to(DEVICE))
    pred = out.argmax(1).item()
    loss = out[0, pred]
    model.zero_grad()
    loss.backward()

    pooled = torch.mean(grads, dim=[0,2,3])
    cam = (activations[0] * pooled[..., None, None]).sum(0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-9)
    h1.remove(); h2.remove()
    return cam, pred

def overlay_heatmap(orig_img_path):
    img = Image.open(orig_img_path).convert("RGB")
    x = transform(img)
    cam, pred = get_heatmap(model, x, model.layer4)
    cam = cv2.resize(cam, (img.width, img.height))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    img_np = np.array(img)[:,:,::-1]  # RGB->BGR for cv2
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    out_path = os.path.join(SCRIPT_DIR, "..", "models", "gradcam_out.png")
    cv2.imwrite(out_path, overlay)
    print("Saved overlay to:", out_path)

if __name__=="__main__":
    sample = os.path.join(SCRIPT_DIR,"..","data","test","0", os.listdir(os.path.join(SCRIPT_DIR,"..","data","test","0"))[0])
    overlay_heatmap(sample)
