# CarrotAI

Binary image classifier: **carrot vs other**.

- Architecture: MobileNetV2 (transfer learning, ImageNet weights)
- Dataset: ~362 carrot + ~531 non-carrot images (Bing Image Search)
- Accuracy: **97%** (val), best val acc: **98.3%**
- Trained on: Apple MPS (Apple Silicon)

## Files

| File | Description |
|------|-------------|
| `carrot_model.pth` | Trained model checkpoint |
| `class_names.txt` | Class labels: `other`, `carrot` |
| `train.py` | Full training pipeline (downloads dataset + trains) |
| `predict.py` | Run inference on a single image |

## Usage

**Predict:**
```bash
python3 predict.py /path/to/image.jpg
```

**Retrain from scratch:**
```bash
python3 train.py
```

## Requirements

```
torch
torchvision
icrawler
Pillow
tqdm
scikit-learn
numpy<2
```
