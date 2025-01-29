# 🎭🔊 Emotion Recognition from Audio using CNN

## 📌 Project Overview

This project implements an **emotion recognition system** using deep learning models on audio features extracted from speech data. It employs a **feature fusion approach**, leveraging multiple **pre-trained CNN models (ResNet, VGG, and DenseNet)** to extract rich feature representations from spectrograms. These features are concatenated and passed through a classifier to predict emotions.

## 🚀 Key Components

### ✅ Data Preprocessing

- **Audio files** are converted into **spectrograms** to obtain a visual representation.
- The **spectrograms** serve as input to the deep learning model.

### ✅ Model Architecture

- Uses **ResNet, VGG, and DenseNet** as **feature extractors**.
- Extracted features are **concatenated** before classification.
- A **fully connected classifier** predicts the **emotion category**.

### ✅ Training and Evaluation

- **Cross-entropy loss** and **Adam optimizer** are used.
- The model is trained on a **labeled dataset of emotional speech samples**.
- Evaluation is performed using **accuracy and confusion matrix visualization**.

### ✅ Results Visualization

- **Confusion matrix** is plotted to analyze performance across different emotion classes.
- **ROC & AUC curves** are generated for detailed evaluation of classification performance.

## 📊 Dataset

This project utilizes the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, which contains **emotional speech recordings** across various categories such as happy, sad, angry, neutral, etc.

The dataset is publicly available on Kaggle:  
🔗 **[RAVDESS Emotional Speech Audio Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)**

## 🛠 Technologies Used

- **Python** (PyTorch, NumPy, Matplotlib, Scikit-learn)
- **Deep Learning** (CNNs with Feature Fusion)
- **Librosa** (for Audio Processing)
