from zenml import pipeline
from steps.train_step import train_model

@pipeline
def training_pipeline(
    ela_data_path: str,
    base_model_path: str
):
    train_model(
        ela_data_path=ela_data_path,
        base_model_path=base_model_path
    )
