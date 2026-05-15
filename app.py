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

ckpt        = torch.load(MODEL_PATH, map_location="cpu")
CLASS_NAMES = ckpt["class_names"]

model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(CLASS_NAMES)),
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
    background: #0f0f0f; color: #f0f0f0;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 24px;
  }
  h1 { font-size: 2rem; margin-bottom: 6px; letter-spacing: -0.5px; margin-top: 16px; }
  .sub { color: #888; margin-bottom: 28px; font-size: 0.95rem; }

  /* Tabs */
  .tabs { display: flex; gap: 8px; margin-bottom: 28px; }
  .tab {
    padding: 8px 24px; border-radius: 99px; cursor: pointer; font-size: 0.95rem;
    border: 1px solid #333; background: none; color: #888;
    transition: all 0.2s;
  }
  .tab.active { background: #f97316; border-color: #f97316; color: #fff; font-weight: 600; }

  .panel { display: none; width: 100%; max-width: 520px; flex-direction: column; align-items: center; }
  .panel.active { display: flex; }

  /* Upload panel */
  .drop-zone {
    width: 100%; border: 2px dashed #333; border-radius: 20px;
    padding: 48px 24px; text-align: center; cursor: pointer;
    transition: border-color 0.2s, background 0.2s; background: #161616;
  }
  .drop-zone.drag-over { border-color: #f97316; background: #1a1100; }
  .drop-zone input { display: none; }
  .drop-icon { font-size: 3rem; margin-bottom: 12px; }
  .drop-text { color: #888; font-size: 0.95rem; }
  .drop-text span { color: #f97316; text-decoration: underline; cursor: pointer; }
  #preview-wrap { width: 100%; margin-top: 20px; display: none; }
  #preview { width: 100%; border-radius: 16px; max-height: 320px; object-fit: contain; background: #161616; }

  /* Camera panel */
  .cam-wrap {
    width: 100%; border-radius: 20px; overflow: hidden;
    position: relative; background: #161616;
  }
  #video { width: 100%; display: block; border-radius: 20px; transform: scaleX(-1); }
  .cam-overlay {
    position: absolute; top: 12px; left: 12px; right: 12px;
    display: flex; justify-content: space-between; align-items: flex-start;
    pointer-events: none;
  }
  .cam-badge {
    padding: 6px 14px; border-radius: 99px; font-weight: 700;
    font-size: 1rem; backdrop-filter: blur(8px);
    background: rgba(0,0,0,0.55); transition: all 0.2s;
  }
  .cam-badge.carrot { background: rgba(249,115,22,0.85); color: #fff; }
  .cam-badge.other  { background: rgba(30,30,30,0.85);   color: #aaa; }
  .cam-conf {
    padding: 6px 14px; border-radius: 99px; font-size: 0.85rem;
    backdrop-filter: blur(8px); background: rgba(0,0,0,0.55); color: #ccc;
  }
  #startCam {
    margin-top: 16px; padding: 12px 32px; border-radius: 99px;
    background: #f97316; border: none; color: #fff; font-size: 1rem;
    font-weight: 600; cursor: pointer; transition: opacity 0.2s;
  }
  #startCam:hover { opacity: 0.85; }
  #stopCam {
    display: none; margin-top: 16px; padding: 12px 32px; border-radius: 99px;
    background: none; border: 1px solid #555; color: #aaa; font-size: 0.95rem;
    cursor: pointer; transition: border-color 0.2s, color 0.2s;
  }
  #stopCam:hover { border-color: #f97316; color: #f97316; }
  canvas { display: none; }

  /* Result block (shared) */
  .result-box {
    width: 100%; border-radius: 16px; padding: 24px;
    background: #161616; display: none; margin-top: 20px;
  }
  .label { font-size: 1.8rem; font-weight: 700; margin-bottom: 12px; }
  .label.carrot { color: #f97316; }
  .label.other  { color: #60a5fa; }
  .bars { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: flex; align-items: center; gap: 12px; }
  .bar-name { width: 60px; font-size: 0.85rem; color: #aaa; }
  .bar-track { flex: 1; height: 10px; background: #2a2a2a; border-radius: 99px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 99px; transition: width 0.4s cubic-bezier(.4,0,.2,1); }
  .bar-fill.carrot { background: #f97316; }
  .bar-fill.other  { background: #60a5fa; }
  .bar-pct { font-size: 0.85rem; color: #ccc; width: 44px; text-align: right; }
  .try-again {
    margin-top: 16px; background: none; border: 1px solid #333;
    color: #888; padding: 8px 20px; border-radius: 99px; cursor: pointer;
    font-size: 0.9rem; transition: border-color 0.2s, color 0.2s;
  }
  .try-again:hover { border-color: #f97316; color: #f97316; }
  .spinner {
    display: none; width: 36px; height: 36px;
    border: 3px solid #333; border-top-color: #f97316;
    border-radius: 50%; animation: spin 0.7s linear infinite; margin: 20px auto 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<h1>🥕 CarrotAI</h1>
<p class="sub">Распознавание моркови с помощью нейросети</p>

<div class="tabs">
  <button class="tab active" onclick="switchTab('upload')">📁 Фото</button>
  <button class="tab"        onclick="switchTab('camera')">📷 Камера</button>
</div>

<!-- UPLOAD PANEL -->
<div class="panel active" id="panel-upload">
  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept="image/*">
    <div class="drop-icon">🖼️</div>
    <div class="drop-text">Перетащи фото сюда или <span onclick="event.stopPropagation();document.getElementById('fileInput').click()">выбери файл</span></div>
  </div>
  <div id="preview-wrap"><img id="preview" src="" alt="preview"></div>
  <div class="spinner" id="spinner-upload"></div>
  <div class="result-box" id="result-upload">
    <div class="label" id="label-upload"></div>
    <div class="bars">
      <div class="bar-row">
        <div class="bar-name">Морковь</div>
        <div class="bar-track"><div class="bar-fill carrot" id="bar-u-carrot" style="width:0%"></div></div>
        <div class="bar-pct" id="pct-u-carrot">0%</div>
      </div>
      <div class="bar-row">
        <div class="bar-name">Другое</div>
        <div class="bar-track"><div class="bar-fill other" id="bar-u-other" style="width:0%"></div></div>
        <div class="bar-pct" id="pct-u-other">0%</div>
      </div>
    </div>
    <button class="try-again" onclick="resetUpload()">Загрузить другое фото</button>
  </div>
</div>

<!-- CAMERA PANEL -->
<div class="panel" id="panel-camera">
  <div class="cam-wrap">
    <video id="video" autoplay playsinline muted></video>
    <div class="cam-overlay">
      <div class="cam-badge other" id="camBadge">Нет сигнала</div>
      <div class="cam-conf" id="camConf"></div>
    </div>
  </div>
  <canvas id="canvas"></canvas>
  <button id="startCam" onclick="startCamera()">Включить камеру</button>
  <button id="stopCam"  onclick="stopCamera()">Выключить камеру</button>
</div>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['upload','camera'][i]===name));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  if (name !== 'camera') stopCamera();
}

// ── Upload ────────────────────────────────────────────────────────────────
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
dropZone.addEventListener('click', () => fileInput.click());

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview').src = e.target.result;
    document.getElementById('preview-wrap').style.display = 'block';
    dropZone.style.display = 'none';
    predictFile(file);
  };
  reader.readAsDataURL(file);
}

async function predictFile(file) {
  document.getElementById('spinner-upload').style.display = 'block';
  document.getElementById('result-upload').style.display  = 'none';
  const fd = new FormData(); fd.append('image', file);
  const data = await (await fetch('/predict', { method: 'POST', body: fd })).json();
  document.getElementById('spinner-upload').style.display = 'none';
  showUploadResult(data);
}

function showUploadResult(data) {
  const cp = (data.probs.carrot * 100).toFixed(1);
  const op = (data.probs.other  * 100).toFixed(1);
  const lbl = document.getElementById('label-upload');
  lbl.textContent = data.label === 'carrot' ? '🥕 Это морковь!' : '❌ Не морковь';
  lbl.className   = 'label ' + data.label;
  document.getElementById('bar-u-carrot').style.width = cp + '%';
  document.getElementById('bar-u-other').style.width  = op + '%';
  document.getElementById('pct-u-carrot').textContent = cp + '%';
  document.getElementById('pct-u-other').textContent  = op + '%';
  document.getElementById('result-upload').style.display = 'block';
}

function resetUpload() {
  dropZone.style.display = 'block';
  document.getElementById('preview-wrap').style.display = 'none';
  document.getElementById('result-upload').style.display = 'none';
  fileInput.value = '';
}

// ── Camera ────────────────────────────────────────────────────────────────
let stream = null, camInterval = null;
const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = stream;
    document.getElementById('startCam').style.display = 'none';
    document.getElementById('stopCam').style.display  = 'inline-block';
    camInterval = setInterval(captureAndPredict, 800);
  } catch(e) {
    alert('Не удалось получить доступ к камере: ' + e.message);
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  clearInterval(camInterval); camInterval = null;
  video.srcObject = null;
  document.getElementById('startCam').style.display  = 'inline-block';
  document.getElementById('stopCam').style.display   = 'none';
  document.getElementById('camBadge').textContent    = 'Нет сигнала';
  document.getElementById('camBadge').className      = 'cam-badge other';
  document.getElementById('camConf').textContent     = '';
}

let predicting = false;
async function captureAndPredict() {
  if (predicting || !stream || video.readyState < 2) return;
  predicting = true;
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  // mirror the capture to match the mirrored video display
  ctx.save(); ctx.scale(-1, 1); ctx.drawImage(video, -canvas.width, 0); ctx.restore();

  canvas.toBlob(async blob => {
    try {
      const fd = new FormData(); fd.append('image', blob, 'frame.jpg');
      const data = await (await fetch('/predict', { method: 'POST', body: fd })).json();
      const cp = (data.probs.carrot * 100).toFixed(0);
      const badge = document.getElementById('camBadge');
      const conf  = document.getElementById('camConf');
      if (data.label === 'carrot') {
        badge.textContent = '🥕 Морковь!';
        badge.className   = 'cam-badge carrot';
      } else {
        badge.textContent = '❌ Не морковь';
        badge.className   = 'cam-badge other';
      }
      conf.textContent = cp + '%';
    } finally { predicting = false; }
  }, 'image/jpeg', 0.85);
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
    return jsonify({
        "label": CLASS_NAMES[probs.argmax().item()],
        "probs": {name: round(probs[i].item(), 4) for i, name in enumerate(CLASS_NAMES)},
    })

if __name__ == "__main__":
    print("🥕 CarrotAI запущен → http://localhost:8080")
    app.run(debug=False, host="127.0.0.1", port=8080)
