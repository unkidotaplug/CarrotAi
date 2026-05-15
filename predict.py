"""
Usage: python3 predict.py <image_path>
Returns: carrot / other + confidence
"""

import sys
from pathlib import Path
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from PIL import Image

MODEL_PATH = Path(__file__).parent / "carrot_model.pth"

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, len(class_names)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names

def predict(image_path: str):
    model, class_names = load_model()
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    label = class_names[probs.argmax().item()]
    conf  = probs.max().item()
    print(f"Prediction : {label.upper()}")
    print(f"Confidence : {conf:.1%}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs[i]:.1%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
