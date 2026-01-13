from pathlib import Path
from PIL import Image
import shutil


class ImageCleaner:
    """
    Cleans and validates image datasets.
    Converts all images to RGB JPEG.
    Removes corrupted files.
    """

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def _prepare_folders(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for class_dir in self.input_dir.iterdir():
            if class_dir.is_dir():
                (self.output_dir / class_dir.name).mkdir(exist_ok=True)

    def _process_image(self, image_path: Path, save_path: Path) -> bool:
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            img.save(save_path, "JPEG", quality=95)
            return True
        except Exception:
            return False

    def clean(self) -> Path:
        self._prepare_folders()
        removed = 0
        processed = 0

        for class_dir in self.input_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for img_file in class_dir.iterdir():
                if not img_file.is_file():
                    continue

                target = self.output_dir / class_dir.name / (img_file.stem + ".jpg")

                success = self._process_image(img_file, target)
                if success:
                    processed += 1
                else:
                    removed += 1

        print(f"--Processed images: {processed}")
        print(f"--Removed corrupted: {removed}")

        return self.output_dir
