# **WORK IN PROGRESS** Leaffliction

**Leaffliction** is a computer vision project focused on classifying images of plant leaf diseases. The project involes data processing: training set Analysis, Augmentation and Features extraction. Model selection, training and evaluation.

## ⚙️ Features

- **Dataset Analysis**: Parses a leaf image dataset and visualizes class distributions via pie and bar charts.
- **Data Augmentation**: Generates image variations (flip, rotate, scale, contrast, luminosity, blur) to balance class distributions.
**Image Transformation**: Features extraction like Gaussian blur, masks, ROI detection, and color histograms.
- **Model Training**: Trains a classifier on the processed dataset and saves model weights and transformation artifacts.
- **Prediction**: Loads trained model to classify new leaf images and outputs the predicted disease.

## 🖼️ Dataset and Model

- Dataset: subset of [plantvillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (Apple and Grape)
- Simple shallow CNN model achieves over 96% accuracy
- EfficientNetV2B0 based model achieves over 99.6% accuracy

## 🧠 Learning Objectives

- Understand dataset distribution and class imbalance
- Apply standard data augmentation techniques
- Extract features using basic image processing operations
- Train and evaluate a model using a training/validation split
- Achieve and demonstrate classification accuracy > 90%
- Automate the full pipeline from raw input to prediction
