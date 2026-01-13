from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import io
import tempfile

app = FastAPI(title="Forgery Detection API")

MODEL_PATH = Path("../../models/production_model.keras")

model = None

def load_model():
    global model
    if model is None:
        print("📦 Loading production model from:", MODEL_PATH.resolve())
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def ela_transform(image: Image.Image, quality=90):
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, "orig.jpg")
        comp_path = os.path.join(tmpdir, "comp.jpg")

        image.save(orig_path, "JPEG", quality=100)
        image.save(comp_path, "JPEG", quality=quality)

        original = Image.open(orig_path).convert("RGB")
        compressed = Image.open(comp_path).convert("RGB")

        ela = ImageChops.difference(original, compressed)

        extrema = ela.getextrema()
        max_diff = max([e[1] for e in extrema]) or 1

        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)

        return ela


@app.get("/")
def root():
    return {"status": "Forgery Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    ela = ela_transform(image)
    ela = ela.resize((224, 224))

    arr = np.array(ela) / 255.0
    arr = np.expand_dims(arr, axis=0)

    model = load_model()
    pred = model.predict(arr)[0][0]

    label = "Forged" if pred < 0.5 else "Original"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return {
        "prediction": label,
        "confidence": (confidence * 100)
    }
