# Detection of Down Syndrome in Children

## Introduction

This repository contains code for a convolutional neural network (CNN) model aimed at detecting Down syndrome in children based on facial images. The model is trained on a dataset comprising images of both healthy children and children with Down syndrome.

## Dataset

The dataset used in this project consists of facial images of children categorized into two classes: healthy and Down syndrome. The images are preprocessed and augmented before being fed into the CNN model for training.

## Model Architecture

The CNN model architecture is designed to effectively extract features from facial images for classification. The model comprises several convolutional layers followed by batch normalization, activation functions, and max-pooling layers. Dropout layers are also incorporated to prevent overfitting. The model is compiled using the Adam optimizer and binary cross-entropy loss function.

## Training

The model is trained on the preprocessed dataset using TensorFlow. Training is conducted for a specified number of epochs with early stopping enabled to prevent overfitting. During training, metrics such as accuracy, area under the ROC curve (AUC), precision, and recall are monitored.

## Evaluation

After training, the model is evaluated using a separate test dataset to assess its performance on unseen data. Evaluation metrics such as accuracy, AUC, precision, and recall are calculated to measure the model's effectiveness in detecting Down syndrome in children.

## Usage

To use the code:

1. Clone this repository to your local machine.
2. Ensure you have all the necessary dependencies installed (listed in `requirements.txt`).
3. Run the notebook `Down_Syndrome_Detection.ipynb` in an environment that supports Jupyter notebooks (e.g., Google Colab or JupyterLab).

## Results

The trained model achieves competitive performance in detecting Down syndrome in children, as evidenced by high accuracy, AUC, precision, and recall scores.

## Conclusion

The CNN model presented in this project demonstrates promising results in the detection of Down syndrome in children based on facial images. Further improvements and optimizations could be explored to enhance the model's performance and applicability in real-world scenarios.
