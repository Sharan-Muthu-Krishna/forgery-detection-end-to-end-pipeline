# -*- coding: utf-8 -*-
"""
Cloud deployment module for Hugging Face Spaces.
Automatically uploads the production model to a Hugging Face Space.
"""

from pathlib import Path
from huggingface_hub import HfApi
import shutil
import tempfile
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configuration
HF_USERNAME = "SharanMK"
SPACE_NAME = "forgery-detection"
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

# Local paths
HF_SPACE_DIR = Path(__file__).parent.parent.parent / "serving" / "hf_space"


def deploy_to_huggingface(model_path: Path) -> str:
    """
    Deploy the production model to Hugging Face Spaces.
    
    Args:
        model_path: Path to the production_model.keras file
        
    Returns:
        URL of the deployed Space
    """
    api = HfApi()
    
    print("\n" + "="*50)
    print("[DEPLOY] DEPLOYING TO HUGGING FACE SPACES")
    print("="*50)
    
    # Step 1: Create the Space if it doesn't exist
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
        print(f"[OK] Space ready: {REPO_ID}")
    except Exception as e:
        print(f"[NOTE] Space creation note: {e}")
    
    # Step 2: Upload all Space files
    # Use weights file for cross-version compatibility
    weights_path = model_path.parent / "production_model.weights.h5"
    
    files_to_upload = [
        (HF_SPACE_DIR / "app.py", "app.py"),
        (HF_SPACE_DIR / "requirements.txt", "requirements.txt"),
        (HF_SPACE_DIR / "Dockerfile", "Dockerfile"),
        (HF_SPACE_DIR / "README.md", "README.md"),
        (weights_path, "production_model.weights.h5"),
    ]
    
    print(f"[INFO] HF_SPACE_DIR: {HF_SPACE_DIR}")
    print(f"[INFO] HF_SPACE_DIR exists: {HF_SPACE_DIR.exists()}")
    
    for local_path, repo_path in files_to_upload:
        if local_path.exists():
            print(f"[UPLOAD] Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="space"
            )
            print(f"[OK] Uploaded {repo_path}")
        else:
            print(f"[WARN] File not found: {local_path}")
    
    space_url = f"https://huggingface.co/spaces/{REPO_ID}"
    api_url = f"https://{HF_USERNAME.lower()}-{SPACE_NAME}.hf.space"
    
    print("\n" + "="*50)
    print("[SUCCESS] DEPLOYMENT COMPLETE!")
    print("="*50)
    print(f"[URL] Space URL: {space_url}")
    print(f"[URL] API URL: {api_url}")
    print(f"[URL] Docs: {api_url}/docs")
    print("="*50 + "\n")
    
    return api_url


if __name__ == "__main__":
    # Test deployment
    model_path = Path("models/production_model.keras")
    if model_path.exists():
        deploy_to_huggingface(model_path)
    else:
        print(f"Model not found at {model_path}")
