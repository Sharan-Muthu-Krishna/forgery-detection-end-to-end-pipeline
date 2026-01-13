from zenml import step
from pathlib import Path
from src.preprocessing.ela_processor import ELAProcessor


@step
def generate_ela(clean_dataset_path: str, ela_dataset_path: str) -> str:
    """
    Converts clean images into ELA images.
    """
    input_dir = Path(clean_dataset_path)
    output_dir = Path(ela_dataset_path)

    ela = ELAProcessor(input_dir, output_dir)
    ela_path = ela.process()

    return str(ela_path)
