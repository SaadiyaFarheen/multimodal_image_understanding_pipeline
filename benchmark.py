"""Measure inference latency on your machine.

Usage:  python benchmark.py [image_path] [runs]
Prints average / p50 / p95 per stage, so you can put REAL numbers in the README.
"""
import statistics
import sys
import time

from PIL import Image

import caption
import predict


def timed(fn, *args):
    t = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t) * 1000


def summarize(name, values):
    values = sorted(values)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    print(f"{name:15s} avg {statistics.mean(values):7.1f} ms | "
          f"p50 {statistics.median(values):7.1f} ms | p95 {p95:7.1f} ms")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    image = Image.open(path).convert("RGB")

    t = time.perf_counter()
    predict.get_classifier()
    caption.get_captioner()
    print(f"Model load (one-time): {time.perf_counter() - t:.1f} s")

    predict.predict_image(image)       # warm-up runs, not counted
    caption.generate_caption(image)

    cls = [timed(predict.predict_image, image) for _ in range(runs)]
    cap = [timed(caption.generate_caption, image) for _ in range(runs)]
    print(f"\n{runs} runs on {path}")
    summarize("classification", cls)
    summarize("captioning", cap)
    summarize("total", [a + b for a, b in zip(cls, cap)])
