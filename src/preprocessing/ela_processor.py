from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance


class ELAProcessor:
    """
    Generates Error Level Analysis (ELA) images for forgery detection.
    """

    def __init__(self, input_dir: Path, output_dir: Path, quality: int = 90):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.quality = quality

    def _prepare_folders(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for class_dir in self.input_dir.iterdir():
            if class_dir.is_dir():
                (self.output_dir / class_dir.name).mkdir(exist_ok=True)

    def _generate_ela(self, image_path: Path, save_path: Path):
        original = Image.open(image_path).convert("RGB")

        temp_path = save_path.parent / "temp.jpg"
        original.save(temp_path, "JPEG", quality=self.quality)

        compressed = Image.open(temp_path)

        ela = ImageChops.difference(original, compressed)

        extrema = ela.getextrema()
        max_diff = max([e[1] for e in extrema]) or 1

        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)

        ela.save(save_path, "JPEG")
        temp_path.unlink()

    def process(self) -> Path:
        self._prepare_folders()

        count = 0
        for class_dir in self.input_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for img_file in class_dir.iterdir():
                if not img_file.is_file():
                    continue

                out_file = self.output_dir / class_dir.name / img_file.name
                self._generate_ela(img_file, out_file)
                count += 1

        print(f"--Generated ELA images: {count}")
        return self.output_dir
