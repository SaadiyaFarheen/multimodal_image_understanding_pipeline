"""Image classification with a pretrained ResNet50 (ImageNet, 1000 classes).

Note: ResNet50 here is a *single-label* classifier (softmax over 1000 classes).
We return the top-k most likely classes that pass a confidence threshold.
It is NOT a multi-label model.
"""
from functools import lru_cache

import torch
from PIL import Image
from torchvision import models


@lru_cache(maxsize=1)
def get_classifier(pretrained: bool = True):
    """Load ResNet50 once and reuse it (lazy: nothing is loaded at import time)."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.eval()

    # The official preprocessing for these weights: resize 232 -> center-crop 224
    # -> convert to tensor -> normalise with ImageNet mean/std.
    preprocess = models.ResNet50_Weights.DEFAULT.transforms()
    # Class names ship with torchvision, so no network call is needed.
    classes = models.ResNet50_Weights.DEFAULT.meta["categories"]
    return model, preprocess, classes


def predict_image(image: Image.Image, top_k: int = 3, threshold: float = 0.05,
                  pretrained: bool = True):
    """Return up to `top_k` [{"label", "confidence"}] with confidence >= threshold.

    If NOTHING reaches the threshold (e.g. a photo of a person: ImageNet has no
    "person" class, so the probability is spread over many unrelated classes),
    return the single best guess marked "low_confidence": True instead of an
    empty list, so the caller still gets an answer and knows not to trust it.
    """
    model, preprocess, classes = get_classifier(pretrained)

    batch = preprocess(image.convert("RGB")).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(batch)

    probs = torch.softmax(logits[0], dim=0)
    top = torch.topk(probs, top_k)

    pairs = list(zip(top.values.tolist(), top.indices.tolist()))
    results = [{"label": classes[idx], "confidence": round(p, 4)}
               for p, idx in pairs if p >= threshold]
    if not results:
        p, idx = pairs[0]
        results = [{"label": classes[idx], "confidence": round(p, 4), "low_confidence": True}]
    return results
