from zenml import step
from pathlib import Path
from src.preprocessing.image_cleaner import ImageCleaner


@step
def clean_images(raw_dataset_path: str, dataset_root_folder: str, clean_dataset_path: str) -> str:
    """
    Cleans the image dataset.
    """
    base_path = Path(raw_dataset_path)
    actual_data = base_path / dataset_root_folder 

    output_dir = Path(clean_dataset_path)

    cleaner = ImageCleaner(actual_data, output_dir)
    cleaned_path = cleaner.clean()

    return str(cleaned_path)
