"""Runs the real ResNet50 code path with randomly initialised weights
(pretrained=False), so it tests preprocessing / shapes / thresholding offline."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image

import predict


def test_returns_top_k_with_valid_labels():
    img = Image.new("RGB", (100, 100), "green")
    result = predict.predict_image(img, top_k=3, threshold=0.0, pretrained=False)
    assert len(result) == 3
    for r in result:
        assert isinstance(r["label"], str) and r["label"]
        assert 0.0 <= r["confidence"] <= 1.0


def test_low_confidence_fallback_when_nothing_passes_threshold():
    img = Image.new("RGB", (100, 100), "green")
    result = predict.predict_image(img, threshold=1.01, pretrained=False)
    assert len(result) == 1
    assert result[0]["low_confidence"] is True


def test_no_low_confidence_flag_when_labels_pass_threshold():
    img = Image.new("RGB", (100, 100), "green")
    result = predict.predict_image(img, threshold=0.0, pretrained=False)
    assert len(result) >= 1
    assert all("low_confidence" not in r for r in result)


def test_handles_non_rgb_input():
    img = Image.new("L", (100, 100), 128)  # grayscale
    result = predict.predict_image(img, pretrained=False)
    assert len(result) >= 1
