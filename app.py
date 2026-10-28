from __future__ import annotations

import hmac, io, json, logging, secrets, threading, sys, tempfile, webbrowser
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
RUNTIME_TEMP = ROOT / ".runtime_temp"
RUNTIME_TEMP.mkdir(exist_ok=True)
tempfile.tempdir = str(RUNTIME_TEMP)
MODEL_PATH, CLASSES_PATH, STATIC_PATH = ROOT / "image_classifier.keras", ROOT / "classes.json", ROOT / "static"
PORT, IMAGE_SIZE, MAX_UPLOAD, MAX_PIXELS = 8000, (128, 128), 8 * 1024 * 1024, 32_000_000
ACCESS_KEY = secrets.token_urlsafe(24)
MODEL, CLASS_NAMES = None, []
PREDICTION_LOCK = threading.Lock()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


HOST = "127.0.0.1"


def load_model():
    global MODEL, CLASS_NAMES, IMAGE_SIZE
    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        raise FileNotFoundError("The saved model or its category list is missing.")
    names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError("classes.json must contain an ordered list of category names.")
    model = tf.keras.models.load_model(MODEL_PATH)
    if int(model.output_shape[-1]) != len(names):
        raise ValueError("The model and category list do not match. Keep them from the same training run.")
    MODEL, CLASS_NAMES = model, names
    IMAGE_SIZE = (int(model.input_shape[2]), int(model.input_shape[1]))
    logging.info("Loaded the saved classifier with %d categories.", len(names))


def json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def extract_image(body, content_type):
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Choose an image file and try again.")
    envelope = (b"Content-Type: " + content_type.encode("ascii", errors="strict")
                + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    message = BytesParser(policy=default).parsebytes(envelope)
    for part in message.iter_parts():
        if part.get_content_disposition() == "form-data" and part.get_param("name", header="content-disposition") == "image":
            data = part.get_payload(decode=True)
            if data:
                return data
    raise ValueError("No image was included in the upload.")


def classify(data):
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("That image is too large to process.")
            image = ImageOps.exif_transpose(source).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError):
        raise ValueError("That file could not be read as an image.") from None
    with PREDICTION_LOCK:
        scores = MODEL.predict(np.expand_dims(np.asarray(image, dtype=np.float32), 0), verbose=0)[0]
    if len(scores) != len(CLASS_NAMES):
        raise RuntimeError("The model and category order no longer match.")
    order = np.argsort(scores)[::-1][:min(3, len(CLASS_NAMES))]
    return {"label": CLASS_NAMES[int(order[0])], "confidence": float(scores[int(order[0])]),
            "matches": [{"label": CLASS_NAMES[int(i)], "confidence": float(scores[int(i)])} for i in order]}


class SingleInstanceHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    server_version = "IndexImageDesk/2.0"

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/api/status", "/health"):
            json_response(self, 200, {"ready": MODEL is not None, "categories": CLASS_NAMES,
                                      "categoryCount": len(CLASS_NAMES), "imageSize": list(IMAGE_SIZE)})
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/static/app.css": ("app.css", "text/css; charset=utf-8"),
                  "/static/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if path not in assets:
            json_response(self, 404, {"error": "That page was not found."})
            return
        name, mime = assets[path]
        try:
            body = (STATIC_PATH / name).read_bytes()
        except OSError:
            json_response(self, 500, {"error": "The image desk files could not be loaded."})
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlparse(self.path).path not in ("/api/predict", "/predict"):
            json_response(self, 404, {"error": "That page was not found."})
            return
        if not hmac.compare_digest(self.headers.get("X-Access-Key", ""), ACCESS_KEY):
            json_response(self, 403, {"error": "This session key expired. Open the current link in the launcher window."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            json_response(self, 400, {"error": "The upload could not be read."})
            return
        if length <= 0:
            json_response(self, 400, {"error": "Choose an image before classifying it."})
            return
        if length > MAX_UPLOAD + 32_768:
            json_response(self, 413, {"error": "Choose an image under 8 MB."})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("The upload ended before it was received.")
            data = extract_image(body, self.headers.get("Content-Type", ""))
            if len(data) > MAX_UPLOAD:
                json_response(self, 413, {"error": "Choose an image under 8 MB."})
                return
            result = classify(data)
        except ValueError as exc:
            json_response(self, 400, {"error": str(exc)})
            return
        except Exception:
            logging.exception("Classification failed")
            json_response(self, 500, {"error": "The image could not be classified. Try another."})
            return
        json_response(self, 200, result)

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)


def main():
    global PORT
    load_model()
    host = HOST
    for candidate in range(8000, 8010):
        try:
            server = SingleInstanceHTTPServer((host, candidate), Handler)
            PORT = candidate
            break
        except OSError as exc:
            if getattr(exc, "errno", None) not in (98, 10048):
                raise
    else:
        raise RuntimeError("Ports 8000 through 8009 are busy. Close an old image desk and try again.")
    url = f"http://{host}:{PORT}/?key={ACCESS_KEY}"
    (ROOT / ".image-desk-session.json").write_text(json.dumps({"url": url, "port": PORT}), encoding="utf-8")
    logging.info("Index Image Desk is ready at %s", url)
    if "--open" in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    logging.info("This temporary session key changes each time the image desk starts.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Closing the image desk.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
