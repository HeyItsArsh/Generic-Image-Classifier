# Generic Image Classifier — Image Desk

A local image-classification application with a browser interface, command-line prediction, and batch evaluation. The included trained model recognizes **Apple, Banana, Grape, Mango, Strawberry, cats, and dogs**.

## Quick start on Windows

Install Python 3.12, then run in PowerShell:

```powershell
git clone https://github.com/HeyItsArsh/Generic-Image-Classifier.git
Set-Location Generic-Image-Classifier
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py --open
```

After setup, double-click **Start UI.bat**. Keep the launcher running. The browser opens after the model loads. The server listens only on `127.0.0.1`, chooses an available port from 8000 through 8009, and uses a temporary session key. Use the full URL printed by the launcher if the browser does not open.

On Linux/macOS, create a Python 3.12 virtual environment and run `python -m pip install -r requirements.txt`, followed by `python app.py --open` inside that environment.

The trained model and labels are included. Datasets, environments, session keys, feature caches, and local backups are excluded from Git. No image dataset is required to launch the application.

## Interface

Select or drop an image, inspect its preview, and request classification. The interface shows the highest-ranked category, alternative matches, model status, and recent results. Uploads are limited to 8 MB. Images are not automatically saved to training folders or used for learning; recent history is held in the browser session.

## Model

- ImageNet-pretrained MobileNetV2 backbone, frozen during training.
- 224 × 224 RGB input, EXIF orientation correction and bilinear resizing.
- Pixel scaling to [-1, 1] embedded in the saved model.
- Pooled features, training-fitted normalization, dropout 0.25, L2-regularized softmax head.
- Seven category outputs ordered by `classes.json`.

`image_classifier.keras` and `classes.json` are a matched pair. Keep them together. The model always selects among known categories: unknown-object rejection, object localization, and automatic category expansion are not implemented. Confidence is a softmax score, not a calibrated probability of correctness. Sequential retention after adding categories has not been demonstrated.

## Recorded experiment

| Split | Images | Accuracy |
|---|---:|---:|
| Training | 7,500 | Not used as the reported generalization score |
| Validation | 700 | 90.57% |
| Internal holdout | 1,050 | 90.86% |

The head used batch size 64 and at most 60 epochs; early stopping ended the recorded run after 20 epochs and restored the validation-loss checkpoint. Feature extraction used batch size 32. The holdout is an internal split, not a fresh external benchmark. Exact-file deduplication does not eliminate all near-duplicates. The existing 1,797-image test collection was separately evaluated and has historical source overlap; it is not an independent benchmark.

See `MODEL_CARD.md`, `upgrade/audit.json`, `upgrade/comparison.json`, and the evaluation CSVs for recorded evidence. These records describe the saved experiment and do not imply a new training run on every checkout.

## Predict and evaluate

```powershell
.\venv\Scripts\python.exe Predict.py path\to\image.jpg
.\venv\Scripts\python.exe Evaluate.py
```

For evaluation, supply your own held-out images under `test_dataset/<category>/`, using exact names from `classes.json`. Evaluate.py writes `evaluation_summary.csv` and `confusion_matrix.csv` and overwrites existing results. CLI, app, and evaluation read image dimensions from the saved model and use consistent preprocessing.

## Train on additional images

Provide labeled images in `dataset/`, `dataset_fruit/`, or `combined_dataset/`. The trainer also reads `test_dataset_Combined/` as an animal **source pool**, mapping `cat` and `dog` to the saved labels. Despite the folder name, that pool is not treated as an untouched test set. Put independent evaluation images in `test_dataset/` instead.

Example:

```text
dataset/
  Apple/
  Banana/
  Grape/
  Mango/
  Strawberry/
  cats/
  dogs/
test_dataset/
  Apple/
  ...
```

Provide sufficient readable examples of **all seven categories**, plus a nonempty labeled test collection. Exact hashes from test_dataset are excluded before training split selection. Source duplicates are collapsed and conflicting labels are excluded. The trainer selects up to 900 training images per fruit class and 1,500 per animal class, with separate validation/holdout assignments. Adding images does not guarantee that every image is selected.

```powershell
.\venv\Scripts\python.exe Main.py
```

The pipeline builds a **fresh head**, writes a candidate under `upgrade/`, and compares it against the active model on the same selected images. It does not automatically replace the active model. It writes a local `split_manifest.json` with file paths/hashes and updates audit/comparison records; preserve experiment records before retraining. The manifest and caches are excluded from Git. Pretrained backbone weights download on first training and cache locally.

Do not use `--reuse-split` after changing dataset files: that option reuses a saved local manifest rather than rescanning sources. New category names are ignored unless class handling is explicitly redesigned; changing classes.json alone is insufficient.

After inspecting newly generated comparison results:

```powershell
.\venv\Scripts\python.exe upgrade\install_upgrade.py
```

Installation requires improved holdout mean class recall and unchanged label order, and creates a local backup of the active model/labels. Restart the app afterward. Do not delete the active model before training: the comparison stage needs it. Repeated development against one holdout set reduces its independence; use fresh external evaluation for final validation.

## Development and CI

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest tests/ -q
```

GitHub Actions configures syntax checks, tests, and an application artifact containing the source, UI, trained model, labels, documentation, and training scripts. The artifact excludes datasets and does not create a virtual environment; install requirements before running it. CI checks software behavior, not model accuracy or continual-learning retention.
