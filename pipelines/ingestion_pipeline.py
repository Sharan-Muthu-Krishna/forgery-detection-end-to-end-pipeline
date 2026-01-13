from zenml import pipeline
from steps.ingest_data_step import ingest_data
from steps.clean_images_step import clean_images
from steps.ela_step import generate_ela


@pipeline
def ingestion_pipeline(zip_path: str, extract_path: str, dataset_root: str, clean_path: str, ela_path: str):
    raw_data = ingest_data(zip_path, extract_path)
    clean_data = clean_images(raw_data, dataset_root, clean_path)
    generate_ela(clean_data, ela_path)
