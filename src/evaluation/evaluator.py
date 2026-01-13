import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

class Evaluator:
    def evaluate(self, model_path, X, y):
        model = tf.keras.models.load_model(model_path)

        preds = model.predict(X)
        preds = (preds > 0.5).astype(int).reshape(-1)
        y = y.numpy().reshape(-1)

        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)

        return {
            "accuracy": float(acc),
            "f1": float(f1)
        }
