"""
Carrot classifier — transfer learning on MobileNetV2.
Dataset: images in CarrotAi/dataset/carrot + CarrotAi/dataset/other
Output: /Users/macboss/CarrotAi/carrot_model.pth
"""

import os, sys, random, subprocess
from pathlib import Path

BASE_DIR   = Path("/Users/macboss/CarrotAi")
DATA_DIR   = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "carrot_model.pth"
CLASS_NAMES = ["other", "carrot"]

# ── 1. Install deps ───────────────────────────────────────────────────────────
PACKAGES = {
    "icrawler":    "icrawler",
    "torch":       "torch",
    "torchvision": "torchvision",
    "PIL":         "Pillow",
    "tqdm":        "tqdm",
    "sklearn":     "scikit-learn",
}

print("==> Checking dependencies...")
for module, pkg in PACKAGES.items():
    try:
        __import__(module)
    except ImportError:
        print(f"    installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── 2. Download images (skip if already present) ──────────────────────────────
from icrawler.builtin import BingImageCrawler

SEARCHES = {
    "carrot": ["carrot vegetable", "fresh carrot", "orange carrot",
               "bunch of carrots", "raw carrot", "carrot root"],
    "other":  ["tomato vegetable", "cucumber vegetable", "broccoli",
               "potato", "banana fruit", "onion vegetable",
               "cabbage vegetable", "apple fruit"],
}
PER_QUERY = 70

print("\n==> Checking dataset...")
for label, queries in SEARCHES.items():
    dest = DATA_DIR / label
    dest.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest.glob("*.jpg")))
    need = PER_QUERY * len(queries)
    if existing >= int(need * 0.7):
        print(f"    {label}: {existing} images — OK")
        continue
    print(f"    {label}: downloading (have {existing}, want ~{need})...")
    for q in queries:
        crawler = BingImageCrawler(
            storage={"root_dir": str(dest)},
            feeder_threads=1, parser_threads=1, downloader_threads=4,
        )
        crawler.crawl(keyword=q, max_num=PER_QUERY,
                      min_size=(100, 100), file_idx_offset="auto")
    print(f"    {label}: {len(list(dest.glob('*.jpg')))} images")

carrot_n = len(list((DATA_DIR / "carrot").glob("*.jpg")))
other_n  = len(list((DATA_DIR / "other").glob("*.jpg")))
print(f"\n    Total — carrot: {carrot_n}  other: {other_n}")
if carrot_n < 50:
    sys.exit("ERROR: not enough carrot images. Check internet connection.")

# ── 3. Dataset ────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.metrics import classification_report

print(f"NumPy: ", end=""); import numpy as np; print(np.__version__)
print(f"Torch: {torch.__version__}")

NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def make_transforms(augment=False):
    ops = [T.Resize((224, 224))]
    if augment:
        ops += [T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)]
    ops += [T.ToTensor(), NORMALIZE]
    return T.Compose(ops)

class CarrotDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (224, 224))
        return self.transform(img), label

# Collect all samples
all_samples = []
for label_idx, label in enumerate(CLASS_NAMES):
    for p in (DATA_DIR / label).glob("*.jpg"):
        all_samples.append((p, label_idx))
random.shuffle(all_samples)

n_val   = max(1, int(0.2 * len(all_samples)))
n_train = len(all_samples) - n_val
train_s, val_s = all_samples[:n_train], all_samples[n_train:]

train_ds = CarrotDataset(train_s, make_transforms(augment=True))
val_ds   = CarrotDataset(val_s,   make_transforms(augment=False))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
print(f"\n==> Dataset: {n_train} train / {n_val} val")

# ── 4. Model ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"==> Device: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for p in model.features.parameters():
    p.requires_grad = False
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2),
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── 5. Train ──────────────────────────────────────────────────────────────────
EPOCHS = 15
best_val_acc = 0.0
print(f"\n==> Training {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    scheduler.step()

    model.eval()
    vc, vt, preds_all, labels_all = 0, 0, [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            vc += (preds == labels).sum().item()
            vt += labels.size(0)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    val_acc = vc / vt
    print(f"  Epoch {epoch:2d}/{EPOCHS} | loss {loss_sum/total:.4f} | "
          f"train {correct/total:.3f} | val {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "class_names": CLASS_NAMES,
            "input_size": 224,
        }, MODEL_PATH)
        print(f"    ✓ Saved best model (val_acc={val_acc:.3f})")

print("\n==> Classification report:")
print(classification_report(labels_all, preds_all, target_names=CLASS_NAMES))
(BASE_DIR / "class_names.txt").write_text("\n".join(CLASS_NAMES))
print(f"\n✅ Done! Model → {MODEL_PATH}")
print(f"   Best val accuracy: {best_val_acc:.1%}")
