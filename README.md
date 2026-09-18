# Handwritten Digit Recognition using CNN

## Introduction
This project is a simple and beginner-friendly implementation of handwritten digit recognition using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and can predict digits from 0 to 9.

## Objective
The main objective is to build a CNN-based application that can:
- Train on the MNIST dataset
- Recognize handwritten digits
- Predict digit probabilities and confidence
- Display results through a user-friendly Streamlit web app

## Features
- CNN model trained on MNIST
- User drawing canvas for digit recognition
- Image upload support for handwritten digits
- Model performance dashboard
- Accuracy and loss graphs
- Confusion matrix and classification report
- Simple and academic project layout

## Technologies Used
- Python 3.10+
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

## Dataset
The project uses the built-in MNIST dataset from TensorFlow Keras:

from tensorflow.keras.datasets import mnist

It contains 28x28 grayscale images of handwritten digits from 0 to 9.

## CNN Architecture
The model is a simple CNN:

1. Input: 28x28x1
2. Conv2D with 32 filters, 3x3 kernel, ReLU
3. MaxPooling2D with 2x2 pool
4. Conv2D with 64 filters, 3x3 kernel, ReLU
5. MaxPooling2D with 2x2 pool
6. Flatten
7. Dense(128, ReLU)
8. Dropout(0.5)
9. Dense(10, Softmax)

## Project Structure
handwritten_digit_recognition/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── model/
│   └── digit_cnn.keras
├── utils/
│   ├── preprocessing.py
│   └── visualization.py
└── assets/
    └── sample_images/

## File Purpose
- app.py: Streamlit user interface for home, drawing, upload, and metrics
- train_model.py: trains and evaluates the CNN model
- predict.py: loads the model and predicts digits from new images
- preprocessing.py: prepares handwritten images for CNN input
- visualization.py: saves training plots and confusion matrix
- requirements.txt: project dependencies
- README.md: project documentation

## Installation
Create a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## How to Train the Model
Run:

python train_model.py

This script will:
- Load the MNIST dataset
- Preprocess the images
- Train the CNN model
- Save the trained model as model/digit_cnn.keras
- Generate accuracy/loss graphs and evaluation reports

## How to Run the Application
Start the Streamlit app:

streamlit run app.py

## How Prediction Works
1. The user draws a digit or uploads an image.
2. The image is converted to grayscale and resized to 28x28.
3. The pixel values are normalized.
4. The image is reshaped to the CNN input format.
5. The trained CNN predicts the most likely digit.
6. The output includes the predicted digit and confidence score.

## Results
The model should achieve a good accuracy on the MNIST test dataset, often above 98% for a simple CNN implementation.

## Applications
- Automated form processing
- Banking digit recognition
- Educational projects
- Basic OCR systems

## Limitations
- Works best on clean, centered digits
- Less accurate on noisy or slanted handwriting
- Not designed for large-scale production use

## Future Enhancements
- Add webcam-based real-time recognition
- Improve preprocessing for better robustness
- Add more performance visualization
- Expand to more custom handwritten datasets

## Final Note
This project is designed to be simple, beginner-friendly, and easy to demonstrate in a college mini project environment.

# Handwritten-Digit-Recognition-Using-Convolutional-Neural-Network
