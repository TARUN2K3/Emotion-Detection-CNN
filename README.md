# Emotion Detection using CNN

## Overview

This project implements Emotion Detection using Convolutional Neural Networks (CNN). The model is trained to recognize facial expressions such as happiness, sadness, anger, surprise, disgust, and fear. Emotion detection using Convolutional Neural Networks (CNNs) is a popular and effective method in the field of computer vision and machine learning. CNNs are particularly well-suited for this task because they excel at detecting patterns and features in images, which is crucial for recognizing facial expressions and emotions.

## Dataset

The model is trained on the [FER2013 dataset](https://www.kaggle.com/deadskull7/fer2013), which consists of 35,887 grayscale images of size 48x48 pixels, each labeled with one of seven emotions.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV (cv2)
- Matplotlib
- NumPy

You can install the required libraries using pip:

```bash
pip install tensorflow keras opencv-python matplotlib numpy
```

## Usage

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/TARUN2K3/Emotion-Detection-CNNn.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd Emotion-Detection-CNN
    ```

3. **Train the Model:**

    Use the provided Jupyter notebook or Python script to train the CNN model on the FER2013 dataset.

    ```bash
    TrainEmotionDetector.py
    ```

4. **Run Inference:**

    Once the model is trained, you can run inference on images or real-time video using the provided scripts.
    ```bash
    website.py
    ```

## Results

The trained model achieves an accuracy of approximately 70% on the test set.
![image](https://github.com/TARUN2K3/Emotion-Detection-CNN/assets/127468524/0f7102a5-6181-457c-be1b-e2a59067a74f)


## Acknowledgments

- The FER2013 dataset is provided by Pierre-Luc Carrier and contains images from the Facial Expression Recognition 2013 Challenge.
- Inspired by various tutorials and resources on CNN-based emotion detection.

