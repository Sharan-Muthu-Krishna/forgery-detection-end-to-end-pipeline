from zenml import pipeline
from steps.train_step import train_model
from steps.prepare_test_data import prepare_test_data
from steps.evaluate_model_step import evaluate_model
from steps.deploy_model_step import deploy_model_if_better

@pipeline
def train_eval_deploy_pipeline(ela_data_path, base_model_path, test_dir, clean_dir, ela_dir):
    model_path = train_model(ela_data_path, base_model_path)
    X, y = prepare_test_data(test_dir, clean_dir, ela_dir)
    metrics = evaluate_model(model_path, X, y)
    deploy_model_if_better(metrics, model_path)
