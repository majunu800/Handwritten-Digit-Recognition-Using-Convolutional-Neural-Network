from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from utils.visualization import save_confusion_matrix, save_training_history


def build_model():
    """Create a beginner-friendly CNN for MNIST digit recognition."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1), padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_metrics(history, y_true, y_pred, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
    }

    with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    with (model_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    with (model_dir / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(classification_report(y_true, y_pred, digits=4))

    save_training_history(history, model_dir)
    save_confusion_matrix(y_true, y_pred, list(range(10)), model_dir)


def main():
    print("Loading MNIST dataset...")
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model_dir = Path(__file__).resolve().parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Building CNN model...")
    model = build_model()

    callback = None
    try:
        from tensorflow.keras.callbacks import EarlyStopping

        callback = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    except Exception:
        callback = []

    print("Training CNN model...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=8,
        batch_size=128,
        callbacks=callback,
        verbose=1,
    )

    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    model_path = model_dir / "digit_cnn.keras"
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    save_metrics(history, y_test, y_pred, model_dir)

    with (model_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump({key: [float(v) for v in values] for key, values in history.history.items()}, file, indent=4)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
