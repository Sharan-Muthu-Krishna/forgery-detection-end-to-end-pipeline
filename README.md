# 🕵️ End-to-End Image Forgery Detection Pipeline

An automated MLOps pipeline that detects forged/tampered images using **Error Level Analysis (ELA)** and **Deep Learning**. One command trains, evaluates, and deploys the model — fully automated from data to production.

## 🎯 Overview

When someone edits an image (copy-paste, splice, retouch), the tampered region has different compression artifacts compared to the rest of the image. This system:

1. Applies **Error Level Analysis (ELA)** to highlight tampered regions
2. Feeds ELA images into a fine-tuned **MobileNetV2** model
3. Classifies images as **Original** or **Forged** with a confidence score

### Results

| Metric | Score |
|--------|-------|
| Training Accuracy | **99.53%** |
| Test Accuracy | **92.37%** |
| F1 Score | **92.81%** |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     AUTOMATED PIPELINE                        │
│                                                                │
│   Raw Images ──► ELA Transform ──► MobileNetV2 ──► Prediction │
│                                                                │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌───────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐  │
│   │   Train    │──►│  Evaluate  │──►│ Compare  │──►│ Deploy │  │
│   │   Model    │   │   Model    │   │ F1 Score │   │ if     │  │
│   │           │   │           │   │          │   │ better │  │
│   └───────────┘   └───────────┘   └──────────┘   └────────┘  │
│                                                                │
├──────────────────────────────────────────────────────────────┤
│                      SERVING LAYER                             │
│                                                                │
│   FastAPI REST API  ──►  Hugging Face Spaces (Cloud)          │
│   Streamlit UI      ──►  Local Web Interface                  │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/Sharan-Muthu-Krishna/forgery-detection-end-to-end-pipeline.git
cd forgery-detection-end-to-end-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_train_eval_deploy.py
```

This single command will:
1. ✅ Train the model on ELA images
2. ✅ Evaluate on test data
3. ✅ Compare F1 score with production model
4. ✅ Auto-deploy to Hugging Face if better

### View Experiment Tracking

```bash
mlflow ui
```
Open http://localhost:5000 to see all training runs, metrics, and model comparisons.

### Run Local Streamlit UI

```bash
streamlit run serving/ui/app.py
```

---

## 🌐 Live API

The model is deployed on Hugging Face Spaces:

| Link | Description |
|------|-------------|
| [API Endpoint](https://sharanmk-forgery-detection.hf.space) | Production REST API |

### API Usage

```bash
curl -X POST "https://sharanmk-forgery-detection.hf.space/predict" \
  -F "file=@your_image.jpg"
```

Response:
```json
{
  "prediction": "Forged",
  "confidence": 87.45
}
```

---

## 🔧 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Deep Learning** | TensorFlow / Keras | Model training & inference |
| **Model** | MobileNetV2 | Lightweight CNN with Transfer Learning |
| **Pipeline** | ZenML | MLOps orchestration & step caching |
| **Tracking** | MLflow | Experiment tracking & metric logging |
| **API** | FastAPI | REST API for predictions |
| **UI** | Streamlit | Web interface for image upload |
| **Container** | Docker | Reproducible deployment environment |
| **Cloud** | Hugging Face Spaces | Free model hosting & serving |
---

## 🔬 How ELA Works

**Error Level Analysis** detects tampering by exploiting JPEG compression artifacts:

1. **Compress** — Re-save the image at 90% JPEG quality
2. **Compare** — Calculate pixel-by-pixel difference between original and compressed
3. **Highlight** — Scale up differences to make tampered regions visible

Untampered images show **uniform** error levels. Tampered regions show **significantly different** error levels because they were added at a different compression level.

---

## 📊 MLOps Pipeline Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   TRAIN MODEL   │────►│ PREPARE TEST    │────►│ EVALUATE MODEL  │────►│ DEPLOY IF       │
│                 │     │ DATA            │     │                 │     │ BETTER          │
│ • Load base     │     │ • Clean images  │     │ • Predict on    │     │ • Compare F1    │
│   model         │     │ • Generate ELA  │     │   test set      │     │ • Export weights│
│ • Fine-tune     │     │ • Normalize     │     │ • Calc accuracy │     │ • Upload to HF  │
│ • Log to MLflow │     │                 │     │ • Calc F1 score │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🧩 Design Patterns

- **Factory Pattern** — `DataIngestorFactory` creates the appropriate data ingestor based on file type
- **Abstract Base Class** — `DataIngestor` defines a contract for all data ingestors
- **Three-Layer Architecture** — Core logic (`src/`) → Steps (`steps/`) → Pipelines (`pipelines/`)
- **Performance-Based Deployment** — Auto-deploy only when new model outperforms production

---

## 📝 License

This project is for educational and portfolio purposes.
