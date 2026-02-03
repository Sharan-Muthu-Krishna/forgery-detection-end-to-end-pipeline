from zenml import step
import shutil
from pathlib import Path
import json
import tensorflow as tf
from src.deployment.cloud_deployer import deploy_to_huggingface


def export_weights(model_path: Path) -> Path:
    """Export model weights for cross-version compatibility"""
    weights_path = model_path.parent / "production_model.weights.h5"
    
    print(f"[EXPORT] Loading model to export weights...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"[EXPORT] Saving weights to: {weights_path}")
    model.save_weights(weights_path)
    
    print(f"[EXPORT] Weights file size: {weights_path.stat().st_size / 1024 / 1024:.2f} MB")
    return weights_path


@step
def deploy_model_if_better(metrics: dict, candidate_model_path: str) -> str:
    prod_path = Path("models/production_model.keras")
    prod_metrics_path = Path("models/production_metrics.json")
    candidate_path = Path(candidate_model_path)

    candidate_f1 = metrics["f1"]

    print("\n====== DEPLOYMENT DECISION ======")
    print(f"Candidate F1: {candidate_f1}")

    should_deploy = False

    if not prod_path.exists():
        print("No production model found -> deploying first model")
        prod_path.parent.mkdir(exist_ok=True)
        shutil.copy(candidate_path, prod_path)
        prod_metrics_path.write_text(json.dumps(metrics, indent=2))
        should_deploy = True
        result = "DEPLOYED_FIRST_MODEL"
    else:
        prod_metrics = json.loads(prod_metrics_path.read_text())
        prod_f1 = prod_metrics["f1"]

        print(f"Production F1: {prod_f1}")

        if candidate_f1 > prod_f1:
            print("Candidate is BETTER -> replacing production model")
            shutil.copy(candidate_path, prod_path)
            prod_metrics_path.write_text(json.dumps(metrics, indent=2))
            should_deploy = True
            result = "DEPLOYED_NEW_MODEL"
        else:
            print("Candidate is WORSE -> keeping old production model")
            result = "KEPT_OLD_MODEL"

    # Deploy to cloud if we have a new/better model
    if should_deploy:
        try:
            # Step 1: Export weights for cross-version compatibility
            print("\n[CLOUD DEPLOY] Step 1: Exporting weights...")
            export_weights(prod_path)
            
            # Step 2: Upload to Hugging Face
            print("\n[CLOUD DEPLOY] Step 2: Uploading to Hugging Face...")
            deploy_to_huggingface(prod_path)
            
        except Exception as e:
            print(f"[WARNING] Cloud deployment failed: {e}")
            print("Local deployment succeeded, but cloud deployment failed.")

    return result
