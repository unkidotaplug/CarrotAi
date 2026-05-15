"""
Retrain CarrotAI v2 — fixes false positives on orange objects.
Changes vs v1:
  - Added orange non-carrot negatives (furniture, clothes, fruits, etc.)
  - Unfreeze last 3 blocks of MobileNetV2 backbone (shape/texture learning)
  - Stronger augmentation
  - More epochs with cosine LR schedule
"""

import os, sys, random, subprocess
from pathlib import Path

BASE_DIR    = Path("/Users/macboss/CarrotAi")
DATA_DIR    = BASE_DIR / "dataset"
MODEL_PATH  = BASE_DIR / "carrot_model.pth"
CLASS_NAMES = ["other", "carrot"]

print("==> Checking dependencies...")
for module, pkg in {"torch":"torch","torchvision":"torchvision",
                    "PIL":"Pillow","tqdm":"tqdm","sklearn":"scikit-learn",
                    "icrawler":"icrawler"}.items():
    try: __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

from icrawler.builtin import BingImageCrawler

# ── Download extra negatives: orange objects that are NOT carrots ─────────────
EXTRA_NEGATIVES = [
    "orange sofa furniture",
    "orange chair furniture",
    "orange jacket clothing",
    "orange pumpkin vegetable",
    "orange fruit citrus",
    "orange traffic cone",
    "orange basketball",
    "orange flower",
    "orange pencil",
    "orange cat",
]

print("\n==> Downloading extra orange-object negatives...")
other_dir = DATA_DIR / "other"
other_dir.mkdir(parents=True, exist_ok=True)

for query in EXTRA_NEGATIVES:
    before = len(list(other_dir.glob("*.jpg")))
    crawler = BingImageCrawler(
        storage={"root_dir": str(other_dir)},
        feeder_threads=1, parser_threads=1, downloader_threads=4,
    )
    crawler.crawl(keyword=query, max_num=60, min_size=(100,100), file_idx_offset="auto")
    after = len(list(other_dir.glob("*.jpg")))
    print(f"    '{query}': +{after-before} images")

carrot_n = len(list((DATA_DIR/"carrot").glob("*.jpg")))
other_n  = len(list(other_dir.glob("*.jpg")))
print(f"\n    Total — carrot: {carrot_n}  other: {other_n}")

# ── Dataset ───────────────────────────────────────────────────────────────────
import torch, torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

class CarrotDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (224,224))
        return self.transform(img), label

NORMALIZE = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

train_tf = T.Compose([
    T.Resize((256,256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.1),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    NORMALIZE,
])
val_tf = T.Compose([T.Resize((224,224)), T.ToTensor(), NORMALIZE])

all_samples = []
for idx, label in enumerate(CLASS_NAMES):
    for p in (DATA_DIR/label).glob("*.jpg"):
        all_samples.append((p, idx))
random.shuffle(all_samples)

n_val   = max(1, int(0.2 * len(all_samples)))
n_train = len(all_samples) - n_val
train_s, val_s = all_samples[:n_train], all_samples[n_train:]

train_ds = CarrotDataset(train_s, train_tf)
val_ds   = CarrotDataset(val_s,   val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
print(f"\n==> Dataset: {n_train} train / {n_val} val")

# ── Model: unfreeze last 3 blocks for shape/texture learning ─────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"==> Device: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze all first, then unfreeze last 3 InvertedResidual blocks + classifier
for p in model.parameters():
    p.requires_grad = False

# MobileNetV2 features: 0..17 blocks + 18 conv
# Unfreeze blocks 15,16,17,18 (deep texture/shape layers)
for i in range(15, 19):
    for p in model.features[i].parameters():
        p.requires_grad = True

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
)
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"==> Trainable params: {trainable:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4, weight_decay=1e-4
)
EPOCHS = 25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ── Train ─────────────────────────────────────────────────────────────────────
best_val_acc = 0.0
print(f"\n==> Training {EPOCHS} epochs...")

for epoch in range(1, EPOCHS+1):
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
    lr = scheduler.get_last_lr()[0]
    print(f"  Epoch {epoch:2d}/{EPOCHS} | loss {loss_sum/total:.4f} | "
          f"train {correct/total:.3f} | val {val_acc:.3f} | lr {lr:.2e}")

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
(BASE_DIR/"class_names.txt").write_text("\n".join(CLASS_NAMES))
print(f"\n✅ Done! Model → {MODEL_PATH}")
print(f"   Best val accuracy: {best_val_acc:.1%}")
