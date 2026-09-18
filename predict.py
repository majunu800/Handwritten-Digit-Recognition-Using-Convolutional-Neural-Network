from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.preprocessing import preprocess_image_for_model

MODEL_PATH = Path(__file__).resolve().parent / "model" / "digit_cnn.keras"


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the saved CNN model from disk."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first using python train_model.py."
        )

    try:
        import tensorflow as tf

        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        raise RuntimeError("The model could not be loaded. Please check the saved model file.") from exc


def predict_digit(image_input, model=None):
    """Predict a digit from a PIL image, file path, NumPy array, or uploaded file."""
    if model is None:
        model = load_model()

    image_batch = preprocess_image_for_model(image_input)
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit] * 100)
    return predicted_digit, confidence, probabilities
