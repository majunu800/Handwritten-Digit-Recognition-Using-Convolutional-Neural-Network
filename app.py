from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from predict import load_model, predict_digit
from utils.preprocessing import preprocess_image_for_model

MODEL_DIR = Path(__file__).resolve().parent / "model"


def get_model_status():
    if not (MODEL_DIR / "digit_cnn.keras").exists():
        return False, "The trained model is not available yet. Please train the model first."
    return True, "Model ready."


def load_training_metrics():
    metrics_path = MODEL_DIR / "metrics.json"
    history_path = MODEL_DIR / "training_history.json"

    if metrics_path.exists():
        import json

        with metrics_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    if history_path.exists():
        import json

        with history_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return {
                "accuracy": data.get("accuracy", [0])[-1],
                "val_accuracy": data.get("val_accuracy", [0])[-1],
                "loss": data.get("loss", [0])[-1],
                "val_loss": data.get("val_loss", [0])[-1],
            }

    return None


st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Recognize Digit", "Upload Image", "Model Performance", "About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.header("Handwritten Digit Recognition")

if menu == "Home":
    st.title("Handwritten Digit Recognition using CNN")
    st.markdown(
        "This project uses a Convolutional Neural Network trained on the MNIST dataset to recognize handwritten digits from 0 to 9."
    )

    st.markdown(
        """
        ### Project Overview
        - Build a CNN model that learns from handwritten digit images
        - Use the MNIST dataset for training and evaluation
        - Accept user-drawn or uploaded digit images for prediction
        - Display prediction confidence and model performance metrics
        """
    )

    st.info("Use the sidebar to navigate between digit recognition, image upload, performance metrics, and project information.")

elif menu == "Recognize Digit":
    st.title("Recognize a Digit")

    model_ready, model_message = get_model_status()
    if not model_ready:
        st.warning(model_message)
        st.stop()

    model = load_model()

    st.write("Draw a digit in the box below and click Predict.")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1.0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="digit_canvas",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("digit_canvas", None)
            st.rerun()
    with col2:
        predict_clicked = st.button("Predict", use_container_width=True)

    if canvas_result is not None:
        image_data = canvas_result.image_data
        if image_data is not None and image_data.size > 0:
            rgb = image_data[:, :, :3]
            grayscale = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            pil_image = Image.fromarray(grayscale, mode="L")
            st.image(pil_image, caption="Drawn digit", width=150)

            if predict_clicked:
                try:
                    digit, confidence, probabilities = predict_digit(pil_image, model=model)
                    st.success(f"The model predicts this digit as {digit}.")
                    st.metric("Predicted Digit", digit)
                    st.metric("Confidence", f"{confidence:.2f}%")

                    probability_df = pd.DataFrame({"Digit": list(range(10)), "Probability": probabilities})
                    st.bar_chart(probability_df.set_index("Digit"))
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error("Unable to predict the digit. Please redraw the digit clearly and try again.")

elif menu == "Upload Image":
    st.title("Upload an Image")

    model_ready, model_message = get_model_status()
    if not model_ready:
        st.warning(model_message)
        st.stop()

    model = load_model()
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Uploaded Image", width=220)

            processed = preprocess_image_for_model(image)
            processed_image = processed[0, :, :, 0].reshape(28, 28)
            st.image(processed_image, caption="Processed 28x28 image", width=220, clamp=True)

            digit, confidence, probabilities = predict_digit(image, model=model)
            st.metric("Predicted Digit", digit)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.bar_chart(pd.Series(probabilities, index=list(range(10))))
        except ValueError as exc:
            st.warning(str(exc))
        except Exception:
            st.error("Invalid image format or prediction failed. Please upload a clear handwritten digit image.")

elif menu == "Model Performance":
    st.title("Model Performance")

    metrics = load_training_metrics()
    if metrics is None:
        st.warning("Model metrics are not available. Please train the model first using the training script.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    col2.metric("Validation Accuracy", f"{metrics.get('val_accuracy', 0) * 100:.2f}%")
    col3.metric("Training Loss", f"{metrics.get('loss', 0):.4f}")
    col4.metric("Validation Loss", f"{metrics.get('val_loss', 0):.4f}")

    history_path = MODEL_DIR / "training_history.json"
    if history_path.exists():
        import json

        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)

        st.subheader("Accuracy and Loss Curves")
        acc_df = pd.DataFrame({"Epoch": range(1, len(history.get("accuracy", [])) + 1), "Training Accuracy": history.get("accuracy", []), "Validation Accuracy": history.get("val_accuracy", [])})
        loss_df = pd.DataFrame({"Epoch": range(1, len(history.get("loss", [])) + 1), "Training Loss": history.get("loss", []), "Validation Loss": history.get("val_loss", [])})
        st.line_chart(acc_df.set_index("Epoch"))
        st.line_chart(loss_df.set_index("Epoch"))

    if (MODEL_DIR / "confusion_matrix.png").exists():
        st.subheader("Confusion Matrix")
        st.image(str(MODEL_DIR / "confusion_matrix.png"), use_container_width=True)

    if (MODEL_DIR / "classification_report.txt").exists():
        st.subheader("Classification Report")
        with (MODEL_DIR / "classification_report.txt").open("r", encoding="utf-8") as file:
            report = file.read()
        st.code(report)

elif menu == "About":
    st.title("About the Project")
    st.markdown(
        """
        ## Problem Statement
        Handwritten digit recognition is a classic machine learning problem in which a system must classify images of digits from 0 to 9.

        ## Objective
        To build a simple and effective CNN model that learns patterns from the MNIST dataset and predicts digits from handwritten inputs.

        ## Dataset
        The MNIST dataset is a standard dataset containing 70,000 grayscale images of handwritten digits, commonly used for image classification tasks.

        ## CNN
        A Convolutional Neural Network uses filters to detect edges, shapes, and patterns. It then combines those features to recognize digits.

        ## How the System Works
        1. Train the CNN on the MNIST dataset.
        2. Preprocess the user input to 28x28 grayscale format.
        3. Feed the processed image into the trained model.
        4. Predict the digit and show the confidence score.

        ## Applications
        - Digit recognition systems
        - Bank check processing
        - Form processing
        - Educational projects and demonstrations

        ## Limitations
        - The model performs best with clean, centered handwriting.
        - It may struggle with noisy or distorted digits.

        ## Future Improvements
        - Improve preprocessing and centering
        - Add real-time webcam input
        - Build a larger custom dataset for more robustness
        """
    )
