# Image-Forgery📌 Overview
This project focuses on detecting forged images using Error Level Analysis (ELA) and Convolutional Neural Networks (CNNs). ELA is used to highlight compression artifacts in JPEG images, making it possible to distinguish between authentic and manipulated regions. The CNN model is trained to classify images as real or tampered.

🚀 Features

Error Level Analysis (ELA): Generates ELA-transformed images to reveal discrepancies.
Deep Learning Model: Uses a CNN built with TensorFlow/Keras to classify images as real or forged.
Image Preprocessing: Utilizes PIL and ImageChops for ELA image generation.
Data Augmentation: Applies transformations using ImageDataGenerator.
Performance Evaluation: Includes metrics like accuracy and confusion matrix.

📂 Dataset

The dataset includes real and forged images from CASIA2, a publicly available dataset for image forgery detection.

📦 Dependencies

Install the required libraries using:
pip install numpy matplotlib tensorflow pillow scikit-learn

📖 Usage

Run the Notebook: Open image_forgery.ipynb in Jupyter Notebook or Google Colab.
Mount Google Drive (if using Colab): Ensure the dataset is accessible.
Preprocess Images: Convert images to ELA format.
Train the CNN Model: Train the model on processed images.
Evaluate Performance: Check accuracy and confusion matrix.

📊 Model Architecture

The CNN model consists of:

Convolutional layers with ReLU activation
MaxPooling layers for downsampling
Dropout layers to prevent overfitting

Dense layers for classification

🏆 Results

The model achieves high accuracy in distinguishing forged and real images.
Error Level Analysis helps highlight inconsistencies in manipulated images.
