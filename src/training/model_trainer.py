from pathlib import Path
import tensorflow as tf
from keras.optimizers import Adam
from datetime import datetime

class ModelTrainer:
    def __init__(self, base_model_path: Path, img_size=(224,224), batch_size=16):
        self.base_model_path = base_model_path
        self.img_size = img_size
        self.batch_size = batch_size

    def _load_data(self, data_dir: Path):
        data = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="binary"
        )
        return data.map(lambda x, y: (x/255.0, y))

    def _freeze_layers(self, model, trainable_layers=30):
        for layer in model.layers[:-trainable_layers]:
            layer.trainable = False

    def train_and_save(self, ela_data_path: Path, output_dir="models", epochs=32, lr=1e-5):
        model = tf.keras.models.load_model(self.base_model_path)
        self._freeze_layers(model)

        model.compile(
            optimizer=Adam(lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        train_data = self._load_data(ela_data_path)
        model.fit(train_data, epochs=epochs)

        Path(output_dir).mkdir(exist_ok=True)
        model_path = Path(output_dir) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        model.save(model_path)

        return str(model_path)
