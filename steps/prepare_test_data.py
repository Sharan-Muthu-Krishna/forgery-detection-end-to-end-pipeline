from zenml import step
from pathlib import Path
import tensorflow as tf
from src.preprocessing.image_cleaner import ImageCleaner
from src.preprocessing.ela_processor import ELAProcessor
from typing import Tuple

@step
def prepare_test_data(
    test_dir: str,
    clean_dir: str,
    ela_dir: str
) -> Tuple[tf.Tensor, tf.Tensor]:

    test_dir = Path(test_dir)
    clean_dir = Path(clean_dir)
    ela_dir = Path(ela_dir)

    cleaner = ImageCleaner(test_dir, clean_dir)
    clean_path = cleaner.clean()

    ela = ELAProcessor(clean_path, ela_dir)
    ela_path = ela.process()

    data = tf.keras.utils.image_dataset_from_directory(
        ela_path,
        image_size=(224,224),
        batch_size=32,
        shuffle=False
    )

    X, y = [], []
    for images, labels in data:
        X.append(images)
        y.append(labels)

    X = tf.concat(X, axis=0) / 255.0
    y = tf.concat(y, axis=0)

    return X, y
