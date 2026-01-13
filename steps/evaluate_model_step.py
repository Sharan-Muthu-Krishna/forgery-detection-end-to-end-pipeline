from zenml import step
import mlflow
from src.evaluation.evaluator import Evaluator

@step
def evaluate_model(model_path: str, X_test, y_test) -> dict:
    print(f"🔍 Loading model from: {model_path}")

    evaluator = Evaluator()
    metrics = evaluator.evaluate(model_path, X_test, y_test)

    print(f"📊 Evaluation metrics: {metrics}")

    mlflow.log_metric("test_accuracy", metrics["accuracy"])
    mlflow.log_metric("test_f1", metrics["f1"])

    return metrics
