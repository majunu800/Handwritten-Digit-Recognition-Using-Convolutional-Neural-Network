# Mini Project Report

## Handwritten Digit Recognition Using Convolutional Neural Network

**Student Name:** ____________________  
**Roll Number:** ____________________  
**Course / Semester:** ____________________  
**Institution:** ____________________  
**Academic Year:** 2025-2026  

---

## 1. Problem Statement

Recognizing handwritten digits from images is an important problem in computer vision and pattern recognition. Handwritten digits can have different shapes, sizes, writing styles, thicknesses, and orientations, which makes manual or rule-based recognition difficult.

The problem addressed in this project is to develop an automated system that accepts an image containing a handwritten digit and classifies it as one of the ten digits from 0 to 9. The system should be able to process a user-drawn digit or an uploaded image and display the predicted digit along with the model confidence.

---

## 2. Objective

The objectives of this mini project are:

1. To understand the basic working of a Convolutional Neural Network.
2. To train a CNN model using the MNIST handwritten digit dataset.
3. To preprocess handwritten digit images into the format expected by the model.
4. To classify handwritten digits from 0 to 9.
5. To build a simple interactive web application using Streamlit.
6. To display prediction confidence and model performance information.
7. To provide a complete and reproducible machine learning application.

---

## 3. Existing System

In a traditional system, handwritten digits may be identified manually or by using fixed image-processing rules. Such systems usually depend on manually selected features such as edges, curves, loops, and pixel positions.

### Limitations of the Existing System

- Manual recognition is slow for a large number of images.
- Rule-based systems are sensitive to writing style and image noise.
- Hand-crafted features may not represent all possible digit shapes.
- The system may require separate rules for different writing styles.
- It is difficult to improve the system without redesigning the rules.

Machine learning and deep learning methods overcome many of these limitations by learning useful features directly from labeled image data.

---

## 4. Novelty

The novelty of this project is the combination of a trained CNN model with an interactive user interface. The application is designed as a complete working system rather than only a training script.

The main novel features are:

- A CNN automatically learns important visual features from digit images.
- Users can draw a digit directly on an interactive canvas.
- Users can also upload a PNG, JPG, or JPEG image.
- Input images are converted to grayscale and resized to the MNIST format of 28 x 28 pixels.
- The preprocessing step detects the digit, crops unnecessary background, centers the digit, and normalizes pixel values.
- The application displays the predicted digit, confidence percentage, and class probabilities.
- Training accuracy, validation accuracy, loss curves, confusion matrix, and classification report are available in the application.
- The project can be trained and executed locally using Python and Streamlit.

---

## 5. Code Implementation

### 5.1 Technologies Used

- **Programming language:** Python 3.11
- **Deep learning framework:** TensorFlow and Keras
- **Dataset:** MNIST handwritten digit dataset
- **Web framework:** Streamlit
- **Image processing:** Pillow and NumPy
- **Visualization:** Matplotlib and Streamlit charts
- **Evaluation:** Scikit-learn

### 5.2 Dataset

The project uses the MNIST dataset provided through TensorFlow Keras. The dataset contains grayscale images of handwritten digits from 0 to 9.

- Training images: 60,000
- Test images: 10,000
- Image size: 28 x 28 pixels
- Number of classes: 10
- Pixel format: grayscale

The pixel values are normalized from the range 0-255 to the range 0-1 before training.

### 5.3 CNN Architecture

The implemented CNN has the following layers:

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 28 x 28 x 1 | Accepts a grayscale digit image |
| Convolution | 32 filters, 3 x 3, ReLU | Extracts basic edges and shapes |
| Max Pooling | 2 x 2 | Reduces spatial dimensions |
| Convolution | 64 filters, 3 x 3, ReLU | Learns more complex patterns |
| Max Pooling | 2 x 2 | Reduces feature-map size |
| Flatten | - | Converts feature maps into a vector |
| Dense | 128 neurons, ReLU | Learns high-level combinations of features |
| Dropout | 0.5 | Reduces overfitting |
| Output | 10 neurons, Softmax | Produces probabilities for digits 0-9 |

The model is compiled using the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric. It is trained for up to 8 epochs with early stopping based on validation loss.

### 5.4 Image Preprocessing

For a user-drawn or uploaded image, the application performs the following steps:

1. Loads the image using Pillow.
2. Converts the image to grayscale.
3. Inverts the image when necessary to match the MNIST style.
4. Detects the non-background region containing the digit.
5. Crops the digit and adds padding.
6. Resizes the digit to 20 x 20 pixels.
7. Places it at the center of a 28 x 28 black canvas.
8. Normalizes pixel values between 0 and 1.
9. Adds batch and channel dimensions before prediction.

### 5.5 Application Pages

The Streamlit application contains the following pages:

- **Home:** Provides an overview of the project.
- **Recognize Digit:** Allows the user to draw a digit and predict it.
- **Upload Image:** Allows the user to upload an image containing a digit.
- **Model Performance:** Displays accuracy, loss, training curves, confusion matrix, and classification report.
- **About:** Describes the problem, dataset, CNN, applications, limitations, and future improvements.

### 5.6 Project Structure

```text
handwrite-check(CNN)/
|-- app.py
|-- train_model.py
|-- predict.py
|-- requirements.txt
|-- README.md
|-- REPORT.md
|-- model/
|   |-- digit_cnn.keras
|   |-- metrics.json
|   |-- training_history.json
|   |-- training_history.png
|   |-- confusion_matrix.png
|   |-- classification_report.txt
|   `-- classification_report.json
|-- utils/
|   |-- preprocessing.py
|   `-- visualization.py
`-- assets/
    `-- sample_images/
```

### 5.7 Description of Important Files

- **app.py:** Implements the Streamlit user interface and navigation.
- **train_model.py:** Loads MNIST, builds the CNN, trains it, evaluates it, and saves the model and reports.
- **predict.py:** Loads the trained model and returns the predicted digit, confidence, and probabilities.
- **utils/preprocessing.py:** Converts input images into the 28 x 28 x 1 format required by the CNN.
- **utils/visualization.py:** Generates training curves and the confusion matrix.
- **requirements.txt:** Lists the required Python packages.
- **model/digit_cnn.keras:** Stores the trained CNN model.

### 5.8 Installation and Execution

Install the project dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Train the model:

```powershell
python train_model.py
```

Run the application:

```powershell
streamlit run app.py
```

The application can then be opened at the local Streamlit URL shown in the terminal.

---

## 6. Output

The model was trained and evaluated successfully. The saved training and validation metrics were:

| Metric | Result |
|---|---:|
| Final training accuracy | 98.87% |
| Final validation accuracy | 99.22% |
| Final training loss | 0.0376 |
| Final validation loss | 0.0307 |
| Test accuracy from evaluation report | 99.15% |
| Test samples evaluated | 10,000 |

The classification report produced an overall precision of 99.15%, recall of 99.15%, and F1-score of 99.15% on the test dataset.

The application successfully provides:

- Digit prediction from the drawing canvas.
- Digit prediction from uploaded images.
- Confidence score for each prediction.
- Probability chart for all ten digit classes.
- Training and validation accuracy curves.
- Training and validation loss curves.
- Confusion matrix.
- Classification report.

The system also handles invalid input safely. For example, a blank image is rejected with a message indicating that no digit was detected.

### Sample Output Format

```text
Predicted Digit: 7
Confidence: 99.XX%
```

The exact confidence value changes according to the input image.

### How the CNN Works

The convolution layers scan the image using learnable filters and detect features such as edges, curves, and strokes. The pooling layers reduce the image dimensions while retaining important information. The flatten layer converts the extracted feature maps into a one-dimensional vector. Dense layers use these features to classify the digit. Finally, the softmax layer produces a probability for each class from 0 to 9, and the class with the highest probability is selected as the prediction.

---

## 7. GitHub Repository Link

[https://github.com/majunu800/Handwritten-Digit-Recognition-Using-Convolutional-Neural-Network](https://github.com/majunu800/Handwritten-Digit-Recognition-Using-Convolutional-Neural-Network)

---

## Conclusion

This mini project successfully demonstrates handwritten digit recognition using a Convolutional Neural Network. The model learns digit patterns from the MNIST dataset and achieves approximately 99% accuracy. The Streamlit interface makes the project interactive by allowing users to draw or upload digits and view predictions immediately.

The project provides a practical introduction to image classification, CNN architecture, image preprocessing, model evaluation, and deployment of a machine learning model through a local web application.

## Future Scope

- Add webcam-based digit recognition.
- Support recognition of multiple digits in one image.
- Improve handling of noisy and poorly centered images.
- Train with a larger custom handwriting dataset.
- Deploy the application to a cloud platform.
- Add user authentication and prediction history.
