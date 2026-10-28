"""Batch-evaluate the saved classifier using matching folders in test_dataset/."""
import csv
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "test_dataset"


def main():
    classes = json.loads((ROOT / "classes.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(ROOT / "image_classifier.keras")
    if int(model.output_shape[-1]) != len(classes):
        raise ValueError("The model and classes.json do not match.")
    if not TEST_DIR.is_dir():
        raise ValueError("test_dataset/ was not found.")
    paths=[(p,i) for i,name in enumerate(classes) for p in sorted((TEST_DIR/name).rglob('*')) if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.gif'}]
    def image_batches():
        size=(int(model.input_shape[2]),int(model.input_shape[1]))
        for start in range(0,len(paths),32):
            images=[]; labels=[]
            for path,label in paths[start:start+32]:
                with Image.open(path) as source:
                    images.append(np.asarray(ImageOps.exif_transpose(source).convert('RGB').resize(size,Image.Resampling.BILINEAR),dtype=np.float32))
                labels.append(label)
            yield np.stack(images),np.array(labels,dtype=int)
    dataset=image_batches()
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    seen = 0
    for images, labels in dataset:
        predictions = np.argmax(model.predict(images, verbose=0), axis=1)
        actual = np.asarray(labels).astype(int)
        np.add.at(confusion, (actual, predictions), 1)
        seen += len(actual)
        if seen % 512 < len(actual):
            print(f"Evaluated {seen} images…")
    total = int(confusion.sum())
    if not total:
        raise ValueError("No images could be evaluated.")
    correct = int(np.trace(confusion))
    print(f"\nOverall accuracy: {correct / total:.2%} ({correct}/{total})")
    rows, f1s = [], []
    for i, name in enumerate(classes):
        tp, actual_n, predicted_n = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        precision = tp / predicted_n if predicted_n else 0.0
        recall = tp / actual_n if actual_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if actual_n:
            f1s.append(f1)
        rows.append([name, actual_n, tp, precision, recall, f1])
        print(f"{name}: accuracy {recall:.2%} | precision {precision:.2%} | recall {recall:.2%} | F1 {f1:.2%} | {actual_n} images")
    if f1s:
        print(f"Macro F1: {np.mean(f1s):.2%}")
    with (ROOT / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted", *classes])
        writer.writerows([[name, *map(int, row)] for name, row in zip(classes, confusion)])
    with (ROOT / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "images", "correct", "precision", "recall", "f1"])
        writer.writerows(rows)
    print("Saved confusion_matrix.csv and evaluation_summary.csv.")


if __name__ == "__main__":
    main()
