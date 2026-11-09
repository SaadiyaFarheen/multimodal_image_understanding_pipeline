"""Image captioning with BLIP (Salesforce/blip-image-captioning-base)."""
from functools import lru_cache

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def get_captioner():
    """Load BLIP once and reuse it (lazy: nothing is loaded at import time)."""
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def generate_caption(image: Image.Image, max_new_tokens: int = 30) -> str:
    processor, model = get_captioner()

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True)
