from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import io
import traceback

app = FastAPI(title="Forgery Detection API")

WEIGHTS_PATH = Path("production_model.weights.h5")
model = None
load_error = None


def build_model():
    """Rebuild the exact model architecture used during training"""
    # MobileNetV2 base
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # We'll load our trained weights
    )
    
    # Add custom classification head
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    return model


def load_model():
    global model, load_error
    if model is None and load_error is None:
        try:
            print(f"Building model architecture...")
            model = build_model()
            print(f"Loading weights from: {WEIGHTS_PATH}")
            print(f"Weights file exists: {WEIGHTS_PATH.exists()}")
            model.load_weights(WEIGHTS_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            load_error = str(e)
            print(f"Error loading model: {e}")
            traceback.print_exc()
    return model


def ela_transform(image: Image.Image, quality=90):
    """Generate Error Level Analysis image"""
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


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Forgery Detection API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #333; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Forgery Detection API</h1>
            <p>Detect whether an image has been forged or tampered with using Error Level Analysis (ELA) and Deep Learning.</p>
            
            <div class="endpoint">
                <h3>POST /predict</h3>
                <p>Upload an image to check if it's original or forged.</p>
                <p>Request: <code>multipart/form-data</code> with field <code>file</code></p>
                <p>Response: <code>{"prediction": "Original|Forged", "confidence": float}</code></p>
            </div>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Check API health status.</p>
            </div>
            
            <p><a href="/docs">Interactive API Documentation (Swagger UI)</a></p>
        </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "weights_path": str(WEIGHTS_PATH),
        "weights_exists": WEIGHTS_PATH.exists(),
        "load_error": load_error
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load model
        loaded_model = load_model()
        if loaded_model is None:
            raise HTTPException(
                status_code=500,
                detail=f"Model failed to load: {load_error}"
            )

        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Generate ELA
        ela = ela_transform(image)
        ela = ela.resize((224, 224))

        # Prepare for prediction
        arr = np.array(ela) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        pred = loaded_model.predict(arr, verbose=0)[0][0]

        label = "Forged" if pred < 0.5 else "Original"
        confidence = float(pred if pred > 0.5 else 1 - pred)

        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
