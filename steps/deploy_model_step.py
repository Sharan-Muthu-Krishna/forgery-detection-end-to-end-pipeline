from zenml import step, get_step_context
import shutil
from pathlib import Path
import json

@step
def deploy_model_if_better(metrics: dict, candidate_model_path: str) -> str:
    prod_path = Path("models/production_model.keras")
    prod_metrics_path = Path("models/production_metrics.json")
    candidate_path = Path(candidate_model_path)

    candidate_f1 = metrics["f1"]

    print("\n====== DEPLOYMENT DECISION ======")
    print(f"Candidate F1: {candidate_f1}")

    if not prod_path.exists():
        print("No production model found → deploying first model")
        prod_path.parent.mkdir(exist_ok=True)
        shutil.copy(candidate_path, prod_path)
        prod_metrics_path.write_text(json.dumps(metrics, indent=2))
        return "DEPLOYED_FIRST_MODEL"

    prod_metrics = json.loads(prod_metrics_path.read_text())
    prod_f1 = prod_metrics["f1"]

    print(f"Production F1: {prod_f1}")

    if candidate_f1 > prod_f1:
        print("Candidate is BETTER → replacing production model")
        shutil.copy(candidate_path, prod_path)
        prod_metrics_path.write_text(json.dumps(metrics, indent=2))
        return "DEPLOYED_NEW_MODEL"
    else:
        print("Candidate is WORSE → keeping old production model")
        return "KEPT_OLD_MODEL"
