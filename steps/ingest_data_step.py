from zenml import step
from pathlib import Path
from src.data_ingestion.data_ingestor import DataIngestorFactory


@step
def ingest_data(zip_file_path: str, extract_dir: str) -> str:
    """
    ZenML step that ingests (unzips if needed) the dataset
    and returns the path to the extracted data.
    """
    zip_path = Path(zip_file_path)
    output_dir = Path(extract_dir)

    ingestor = DataIngestorFactory.create(zip_path, output_dir)
    dataset_path = ingestor.ingest()

    return str(dataset_path)
