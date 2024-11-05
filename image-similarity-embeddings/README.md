# Fashion Image Similarity and Recommendation System

This project is a **fashion image similarity and recommendation system** built using deep learning and computer vision techniques. The system allows for finding visually similar fashion items based on a given query image. The project utilizes a pre-trained convolutional neural network to extract embeddings from fashion images, computes similarity scores between images, and recommends similar items to users. It also includes a comprehensive evaluation framework to assess the accuracy and relevance of the recommendations.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)

## Project Overview

The Fashion Image Similarity and Recommendation System is designed to:
1. Load a dataset of fashion images and associated metadata.
2. Extract embeddings using a pre-trained neural network model (e.g., ResNet50 or EfficientNet).
3. Compute similarity between images based on their embeddings.
4. Generate and visualize recommendations for similar items.
5. Evaluate the recommendation system’s accuracy and relevancy based on similarity-driven predictions.

## Features

- **Image Embedding Extraction**: Uses pre-trained CNN models (ResNet50, EfficientNet) to extract feature vectors (embeddings) for fashion images.
- **Similarity Computation**: Calculates cosine similarity scores to measure the visual similarity between images.
- **Recommendation System**: Provides similar fashion items based on a query image.
- **Evaluation Framework**: Includes precision, recall, F1-score, and confusion matrices for assessing recommendation quality.

## Project Structure

![Confusion Matrix](docs/resnet50.png)


```markdown 
# Confusion matrix 

Classification Report for masterCategory:
               precision    recall  f1-score   support

      Apparel       0.98      0.97      0.97       787
  Accessories       0.99      1.00      0.99      1388
     Footwear       0.99      1.00      0.99       642
Personal Care       0.00      0.00      0.00        11
   Free Items       0.95      0.97      0.96       172

     accuracy                           0.98      3000
    macro avg       0.78      0.79      0.78      3000
 weighted avg       0.98      0.98      0.98      3000