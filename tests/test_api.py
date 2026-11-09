"""API tests. The ML models are replaced with fakes, so these run in seconds
and need no model downloads."""
import io
import os

os.environ["PRELOAD_MODELS"] = "0"

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main

client = TestClient(main.app)


def make_image_bytes(fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(
        main.predict, "predict_image",
        lambda image, top_k=3, threshold=0.05: [{"label": "tabby", "confidence": 0.61}][:top_k],
    )
    monkeypatch.setattr(main.caption, "generate_caption", lambda image: "a cat on a table")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_predict_ok():
    r = client.post("/predict", files={"file": ("a.png", make_image_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["labels"] == [{"label": "tabby", "confidence": 0.61}]
    assert body["caption"] == "a cat on a table"
    assert set(body["latency_ms"]) == {"classification", "captioning", "total"}


def test_rejects_non_image_content_type():
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image():
    r = client.post("/predict", files={"file": ("a.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_rejects_too_large(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/predict", files={"file": ("a.png", make_image_bytes(), "image/png")})
    assert r.status_code == 413


def test_query_param_validation():
    r = client.post("/predict?top_k=0", files={"file": ("a.png", make_image_bytes(), "image/png")})
    assert r.status_code == 422


def test_no_temp_files_left_behind():
    before = set(os.listdir("."))
    client.post("/predict", files={"file": ("../evil.png", make_image_bytes(), "image/png")})
    assert set(os.listdir(".")) == before
