from abc import ABC, abstractmethod
from pathlib import Path
import zipfile



class DataIngestor(ABC):
    """Abstract interface for all data ingestors."""

    @abstractmethod
    def ingest(self) -> Path:
        pass



class ZipDataIngestor(DataIngestor):
    """
    Handles ingestion of zipped datasets.
    Extracts only once (idempotent behavior).
    """

    def __init__(self, zip_path: Path, extract_dir: Path):
        self.zip_path = zip_path
        self.extract_dir = extract_dir

    def _is_already_extracted(self) -> bool:
        return self.extract_dir.exists() and any(self.extract_dir.iterdir())

    def ingest(self) -> Path:
        if self._is_already_extracted():
            print(f"--Dataset already exists at {self.extract_dir}")
            return self.extract_dir

        print(f"--Extracting dataset from {self.zip_path}...")

        self.extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.extract_dir)

        print(f"--Dataset extracted to {self.extract_dir}")
        return self.extract_dir



class DataIngestorFactory:
    """Factory to create the correct DataIngestor."""

    @staticmethod
    def create(data_path: Path, extract_dir: Path) -> DataIngestor:
        if data_path.suffix == ".zip":
            return ZipDataIngestor(data_path, extract_dir)

        raise ValueError(f"No ingestor available for {data_path}")
