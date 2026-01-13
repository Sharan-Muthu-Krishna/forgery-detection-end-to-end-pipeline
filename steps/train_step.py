from zenml import step
from pathlib import Path
from src.training.model_trainer import ModelTrainer
import mlflow

@step(enable_cache=False)
def train_model(ela_data_path: str, base_model_path: str) -> str:
    trainer = ModelTrainer(Path(base_model_path))

    mlflow.tensorflow.autolog()

    model_path = trainer.train_and_save(Path(ela_data_path))

    mlflow.log_param("model_path", model_path)

    return model_path
