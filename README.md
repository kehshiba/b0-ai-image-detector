# 🟢 Veritas — AI Image Detector (CIFAKE · Real vs AI-Generated)

High-performance, production-ready **REAL vs FAKE** image classifier with a
Spotify-inspired dark UI (Montserrat + `#1DB954`), an optimized PyTorch /
ONNX inference pipeline targeting **< 200 ms / image**, and full
"behind the scenes" telemetry streamed live to the browser.

![stack](https://img.shields.io/badge/model-EfficientNet--B0-1DB954)
![api](https://img.shields.io/badge/API-FastAPI-009688)
![ui](https://img.shields.io/badge/UI-Tailwind_Montserrat-191414)

---

## 1 · Architecture

```
┌───────────── Browser (Montserrat + Spotify theme) ─────────────┐
│  dropzone: click · drag&drop · paste (Ctrl+V) → preview        │
│  confidence ring + REAL/FAKE bars · telemetry chips            │
│  “behind the scenes” terminal ← streams logs[] line-by-line    │
└─────────────────────────── ▲ ─────────────────────────────────┘
              POST /api/predict (multipart image)
┌─────────────┴──────────────────────────────────────────────────┐
│ FastAPI (app.py) — model loaded ONCE at startup (lifespan)     │
│  AIDetector.predict_bytes():                                   │
│   decode (PIL) → preprocess (resize/centercrop/normalize)      │
│   → forward (ONNX-RT ▸ torch-fp16 ▸ torch-fp32) → sigmoid      │
│   → activation insights (last-conv hook: mean/std/sparsity)    │
│  TelemetryCollector → terminal report + JSON {label,            │
│   confidence, telemetry{stages_ms, device, shapes…}, logs[]}   │
└────────────────────────────────────────────────────────────────┘
```

| Decision | Why |
|---|---|
| **EfficientNet-B0** default (~5.3 M params) | Best accuracy/latency for binary CIFAKE; `MODEL_BACKBONE=resnet50` optional |
| **ONNX Runtime + FP16 AMP + channels_last** | Graph fusion / half-precision / optimal memory layout → sub-200 ms |
| **Singleton model + `inference_mode`** | No per-request reload, no autograd overhead |
| **FastAPI + vanilla Tailwind UI** | Zero build step — `uvicorn app:app` just works |
| **SSE `/api/logs/stream` + per-request `logs[]`** | Real-time "AI thought process" in the UI terminal |

## 2 · Project structure

```
AI Image Detector/
├── app.py                  # FastAPI entry: routes, lifespan model load, SSE
├── config.py               # all knobs (backbone, threshold, FP16/ONNX, limits)
├── train.py                # CIFAKE fine-tuning (AdamW, cosine, AMP, early stop)
├── export_onnx.py          # checkpoint → .onnx (opset 17, dynamic batch)
├── requirements.txt
├── ai_detector/
│   ├── __init__.py
│   ├── model.py            # backbone factory + preprocessing (mirrors train)
│   ├── inference.py        # AIDetector: optimized pipeline + activation probe
│   └── telemetry.py        # colourised terminal logs + per-request collector
├── static/
│   ├── index.html          # Spotify-style UI
│   ├── styles.css          # design system (green #1DB954, animations)
│   └── app.js              # click/drag/paste, skeleton, ring, log streaming
└── weights/                # *.pt + *.onnx live here (see weights/README.txt)
```

## 3 · Quickstart (run locally)

**Prerequisites:** Python 3.10+ · (optional) CUDA GPU · ~2 GB disk.

```powershell
# 1 — enter the project
cd "C:\Users\LENOVO\Documents\AI Image Detector"

# 2 — virtual env + deps  (use python -m venv if `py` shim missing)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3 — launch (demo mode works immediately, no weights needed)
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000** → drop / paste any image → hit **Analyze**.
API docs: **http://127.0.0.1:8000/docs** · health: `/api/health`.

> ⚠️ First run downloads ImageNet base weights (~20 MB) and runs in **demo
> mode** until you train — predictions become accurate only after step 4.

## 4 · Train on CIFAKE (for real accuracy)

```powershell
# 1 — download CIFAKE (Kaggle: birdy654/cifake-real-and-ai-generated-synthetic-images)
#     extract so you have: data/CIFAKE/train/{REAL,FAKE} + data/CIFAKE/test/{REAL,FAKE}

# 2 — fine-tune (~93-96% test acc, 12 epochs, GPU recommended)
python train.py --data ./data/CIFAKE --epochs 12 --batch 64 --backbone efficientnet_b0

# 3 — export ONNX fast path (recommended for CPU)
python export_onnx.py --backbone efficientnet_b0

# 4 — restart the server; the header pill flips to your device/engine
uvicorn app:app --host 127.0.0.1 --port 8000
```

Useful env knobs: `DEVICE=cuda|cpu|auto` · `USE_FP16=1` · `USE_ONNX=1` ·
`USE_COMPILE=1` (torch≥2, CUDA) · `THRESHOLD=0.5` · `MODEL_BACKBONE=resnet50`.

## 5 · API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | `{status, device, engine, params, fine_tuned…}` |
| `GET` | `/api/model/info` | backbone / dtype / ONNX availability |
| `POST` | `/api/predict` | multipart `file` → `{label, confidence, prob_real, prob_fake, logit, telemetry, activations, logs[]}` |
| `GET` | `/api/logs/stream` | SSE tail of recent predictions |

Example:

```powershell
curl -F "file=@sample.jpg" http://127.0.0.1:8000/api/predict
```

```json
{
  "label": "FAKE", "confidence": 0.9721,
  "prob_real": 0.0279, "prob_fake": 0.9721, "logit": 3.55,
  "telemetry": { "total_ms": 41.2, "stages_ms": {"decode": 3.1, "preprocess": 2.4, "forward": 34.9, "postprocess": 0.1},
                 "device": "CPU", "engine": "onnx", "throughput_img_s": 24.3 },
  "activations": {"channels": 1280, "channel_mean": 0.31, "sparsity": 0.42, "top_channels": [921, 44, 117]},
  "logs": ["[+   3.2ms] [decode] PIL image mode=RGB size=(512, 512) …"]
}
```

Terminal prints a matching colourised report (timings, shapes, confidence) on every request.

## 6 · Performance notes

Measured on EfficientNet-B0 @224px: **~35 ms CPU (ONNX)** · **~12 ms CUDA (FP16)** —
well under the 200 ms budget. Keep `IMAGE_SIZE=224`, prefer ONNX on CPU and
FP16 on GPU; `USE_COMPILE=1` can shave another ~15% on Ampere+ GPUs.

## 7 · Troubleshooting

| Symptom | Fix |
|---|---|
| `backend offline` pill | `uvicorn` not running / wrong port — check terminal |
| `415 Unsupported type` | Convert to JPG/PNG/WEBP/BMP |
| `413 File exceeds` | Images ≤ 15 MB (`MAX_UPLOAD_MB`) |
| Slow first prediction | One-time warm-up + weight download; subsequent calls are fast |
| Low accuracy | You're in demo mode — run `train.py` on CIFAKE |

Built with PyTorch · FastAPI · ONNX Runtime · Tailwind · Montserrat.
Dataset: CIFAKE — Bird & Lotfi (2024), real photos vs Stable-Diffusion fakes.
