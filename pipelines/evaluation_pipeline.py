from zenml import pipeline
from steps.load_model import load_latest_model
from steps.prepare_test_data import prepare_test_data
from steps.evaluate_model import evaluate_model

@pipeline
def evaluation_pipeline(test_dir, clean_dir, ela_dir):
    model = load_latest_model()
    X, y = prepare_test_data(test_dir, clean_dir, ela_dir)
    evaluate_model(model, X, y)
