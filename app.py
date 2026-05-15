from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
import torch, torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import io, base64

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "carrot_model.pth"

app = Flask(__name__)

# Load model once at startup
ckpt        = torch.load(MODEL_PATH, map_location="cpu")
CLASS_NAMES = ckpt["class_names"]

model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, len(CLASS_NAMES)),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CarrotAI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f0f0f;
    color: #f0f0f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  h1 { font-size: 2rem; margin-bottom: 6px; letter-spacing: -0.5px; }
  .sub { color: #888; margin-bottom: 36px; font-size: 0.95rem; }

  .drop-zone {
    width: 100%;
    max-width: 480px;
    border: 2px dashed #333;
    border-radius: 20px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: #161616;
  }
  .drop-zone.drag-over { border-color: #f97316; background: #1a1100; }
  .drop-zone input { display: none; }
  .drop-icon { font-size: 3rem; margin-bottom: 12px; }
  .drop-text { color: #888; font-size: 0.95rem; }
  .drop-text span { color: #f97316; text-decoration: underline; cursor: pointer; }

  #preview-wrap {
    width: 100%;
    max-width: 480px;
    margin-top: 24px;
    display: none;
  }
  #preview {
    width: 100%;
    border-radius: 16px;
    max-height: 320px;
    object-fit: contain;
    background: #161616;
  }

  #result {
    width: 100%;
    max-width: 480px;
    margin-top: 20px;
    border-radius: 16px;
    padding: 24px;
    background: #161616;
    display: none;
  }
  .label {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 12px;
  }
  .label.carrot { color: #f97316; }
  .label.other  { color: #60a5fa; }

  .bars { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: flex; align-items: center; gap: 12px; }
  .bar-name { width: 60px; font-size: 0.85rem; color: #aaa; }
  .bar-track {
    flex: 1; height: 10px; background: #2a2a2a; border-radius: 99px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 99px; transition: width 0.6s cubic-bezier(.4,0,.2,1);
  }
  .bar-fill.carrot { background: #f97316; }
  .bar-fill.other  { background: #60a5fa; }
  .bar-pct { font-size: 0.85rem; color: #ccc; width: 44px; text-align: right; }

  .spinner {
    display: none;
    width: 36px; height: 36px;
    border: 3px solid #333;
    border-top-color: #f97316;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 20px auto 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .try-again {
    margin-top: 16px;
    background: none;
    border: 1px solid #333;
    color: #888;
    padding: 8px 20px;
    border-radius: 99px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: border-color 0.2s, color 0.2s;
  }
  .try-again:hover { border-color: #f97316; color: #f97316; }
</style>
</head>
<body>

<h1>🥕 CarrotAI</h1>
<p class="sub">Загрузи фото — нейросеть скажет, морковь это или нет</p>

<div class="drop-zone" id="dropZone">
  <input type="file" id="fileInput" accept="image/*">
  <div class="drop-icon">📷</div>
  <div class="drop-text">Перетащи фото сюда или <span onclick="document.getElementById('fileInput').click()">выбери файл</span></div>
</div>

<div id="preview-wrap">
  <img id="preview" src="" alt="preview">
</div>

<div class="spinner" id="spinner"></div>

<div id="result">
  <div class="label" id="labelText"></div>
  <div class="bars">
    <div class="bar-row">
      <div class="bar-name">Морковь</div>
      <div class="bar-track"><div class="bar-fill carrot" id="barCarrot" style="width:0%"></div></div>
      <div class="bar-pct" id="pctCarrot">0%</div>
    </div>
    <div class="bar-row">
      <div class="bar-name">Другое</div>
      <div class="bar-track"><div class="bar-fill other" id="barOther" style="width:0%"></div></div>
      <div class="bar-pct" id="pctOther">0%</div>
    </div>
  </div>
  <button class="try-again" onclick="reset()">Загрузить другое фото</button>
</div>

<script>
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
dropZone.addEventListener('click', () => fileInput.click());

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview').src = e.target.result;
    document.getElementById('preview-wrap').style.display = 'block';
    dropZone.style.display = 'none';
    predict(file);
  };
  reader.readAsDataURL(file);
}

async function predict(file) {
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('result').style.display = 'none';

  const fd = new FormData();
  fd.append('image', file);

  const res  = await fetch('/predict', { method: 'POST', body: fd });
  const data = await res.json();

  document.getElementById('spinner').style.display = 'none';

  const carrotPct = (data.probs.carrot * 100).toFixed(1);
  const otherPct  = (data.probs.other  * 100).toFixed(1);

  const labelEl = document.getElementById('labelText');
  if (data.label === 'carrot') {
    labelEl.textContent = '🥕 Это морковь!';
    labelEl.className = 'label carrot';
  } else {
    labelEl.textContent = '❌ Не морковь';
    labelEl.className = 'label other';
  }

  document.getElementById('barCarrot').style.width = carrotPct + '%';
  document.getElementById('barOther').style.width  = otherPct  + '%';
  document.getElementById('pctCarrot').textContent = carrotPct + '%';
  document.getElementById('pctOther').textContent  = otherPct  + '%';

  document.getElementById('result').style.display = 'block';
}

function reset() {
  document.getElementById('dropZone').style.display = 'block';
  document.getElementById('preview-wrap').style.display = 'none';
  document.getElementById('result').style.display = 'none';
  document.getElementById('spinner').style.display = 'none';
  fileInput.value = '';
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400

    img    = Image.open(io.BytesIO(file.read())).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    result = {
        "label": CLASS_NAMES[probs.argmax().item()],
        "probs": {name: round(probs[i].item(), 4) for i, name in enumerate(CLASS_NAMES)},
    }
    return jsonify(result)

if __name__ == "__main__":
    print("🥕 CarrotAI запущен → http://localhost:5000")
    app.run(debug=False, port=5000)
