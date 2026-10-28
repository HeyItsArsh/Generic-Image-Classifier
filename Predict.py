"""Classify one image using the project’s saved model."""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python Predict.py path/to/image")
    image_path = Path(sys.argv[1]).expanduser()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    classes = json.loads((ROOT / "classes.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(ROOT / "image_classifier.keras")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB").resize((int(model.input_shape[2]), int(model.input_shape[1])), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)
    scores = model.predict(pixels[None, ...], verbose=0)[0]
    if len(scores) != len(classes):
        raise ValueError("The model and category order do not match.")
    for i in np.argsort(scores)[::-1][:min(5, len(classes))]:
        print(f"{classes[int(i)]:<18} {scores[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()
