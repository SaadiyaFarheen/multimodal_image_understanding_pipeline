# Multimodal Image Understanding Pipeline

An inference service that takes an image and returns **top-k class predictions with confidence scores** (pretrained ResNet50) and a **natural-language caption** (pretrained BLIP), served through a FastAPI REST endpoint.

> **Scope:** this project integrates and serves *pretrained* models. No model is trained or fine-tuned here (see [Limitations](#limitations)).

## Architecture

```
Uploaded image
   │
   ▼
FastAPI  POST /predict ── validates type / size / decodability
   │
   ├──► ResNet50 (ImageNet, torchvision) ──► top-k labels + confidence (softmax)
   │
   └──► BLIP (Salesforce/blip-image-captioning-base) ──► caption
   │
   ▼
JSON: labels, caption, per-stage latency
```

## Tech stack

Python · PyTorch · torchvision · Hugging Face Transformers · FastAPI · Uvicorn · pytest

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API page. The first start downloads the model weights (~1 GB total) and loads both models once at startup.

## API

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /predict`
Multipart form upload of one image (`jpeg`, `png` or `webp`, max 10 MB).

| Query param | Default | Meaning |
|---|---|---|
| `top_k` | 3 | Maximum number of labels to return (1–10) |
| `threshold` | 0.05 | Hide labels whose softmax confidence is below this value. If none reach it, the single best guess is still returned with `"low_confidence": true` |

```bash
curl -X POST "http://127.0.0.1:8000/predict?top_k=3&threshold=0.05" -F "file=@test.jpg"
```

Example response for `test.jpg` (a kitten photo), from a real run:

```json
{
  "labels": [
    {"label": "tabby", "confidence": 0.2528},
    {"label": "Egyptian cat", "confidence": 0.0999},
    {"label": "tiger cat", "confidence": 0.0698}
  ],
  "caption": "a kitten sitting on a table with a white background",
  "latency_ms": {"classification": 181.1, "captioning": 2346.1, "total": 2527.2}
}
```

Confidence is the softmax probability. Similar ImageNet classes (tabby, tiger cat, Egyptian cat) share the probability, so even a correct top label can have a modest score. If no label reaches `threshold`, the best guess is returned with `"low_confidence": true`.

Errors: `415` unsupported file type, `413` file too large, `400` unreadable image, `422` invalid query parameter.

## Performance

Measured with `python benchmark.py test.jpg 20` (20 runs after a warm-up run, CPU only, one image):

| Stage | Avg | p50 | p95 |
|---|---|---|---|
| Classification (ResNet50) | 98.9 ms | 94.0 ms | 162.5 ms |
| Captioning (BLIP) | 2221.5 ms | 2436.9 ms | 2675.7 ms |
| Total | 2320.4 ms | 2529.9 ms | 2758.3 ms |

One-time model load: 8.7 s (weights already downloaded). Captioning is about 96% of the total time, so BLIP is the bottleneck on CPU.

Tested on Windows with Python 3.13, PyTorch 2.11, torchvision 0.26, transformers 5.4. Machine: Intel Core i3-1125G4 (11th Gen, 4 cores / 8 threads, 2.0 GHz base), CPU only, no GPU.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

API tests use fake models and run in seconds. `tests/test_predict.py` exercises the real ResNet50 code path with random weights to check preprocessing, shapes and thresholding offline.

## Design notes

- **Models load once** at startup (lazy and cached), not per request.
- **Correct ImageNet preprocessing** (resize 232, centre-crop 224, normalise with ImageNet mean/std) via the official torchvision transforms for the chosen weights.
- **No temp files:** uploads are decoded in memory, so there are no file-name collisions, leftovers or path-traversal risks.
- **Blocking inference runs in a worker thread** (`def` endpoint) so the server stays responsive.
- **Input validation** for type, size and corrupt files.

## Limitations

- ResNet50 is a **single-label** ImageNet classifier (1000 classes). The API returns the top-k most likely classes; it does not detect several objects independently.
- Objects outside the 1000 ImageNet classes are mapped to the closest class, and confidence scores are not calibrated. ImageNet has no "person" class, so photos of people often get only low-confidence labels (the API then returns `low_confidence: true`), while the BLIP caption still describes the scene well.
- BLIP captions can be generic or occasionally wrong.
- Not evaluated on a labelled dataset yet.

## Future work

- Evaluate on a small hand-labelled set (top-3 accuracy, caption quality).
- True multi-label classification: fine-tune with a sigmoid head on COCO.
- Batch inference and GPU support.
- Containerise with Docker.
- Simple web UI.

## Author

Saadiya Farheen
