# Image Similarity Search

## Overview

This project implements an image similarity search system using a convolutional autoencoder for feature extraction and FAISS (Facebook AI Similarity Search) for efficient nearest neighbor searching. The system is designed to find similar images in a dataset based on a given query image, making it useful for applications in e-commerce, fashion, and any field that requires image analysis.

## Features

- **Image Autoencoder**: A convolutional neural network architecture that compresses images into a lower-dimensional latent space and reconstructs them back to the original space.
- **Similarity Search**: Uses FAISS for efficient searching of similar images based on the encoded features.
- **Custom Dataset Loader**: A PyTorch dataset class that loads images from a specified directory.
- **Visualization**: Utilities for plotting similar images along with their distance scores.


### Description of Key Components

- **`models/autoencoder.py`**: Defines the `ImageAutoencoder` class, which consists of an encoder to compress images and a decoder to reconstruct them.
- **`datasets/image_similarity_dataset.py`**: Implements the `ImageSimilarityDataset` class to load images from a directory, supporting various image formats.
- **`models/similarity_search.py`**: Contains the `ImageSimilaritySearch` class that manages the training of the autoencoder, building of the FAISS index, searching for similar images, and saving/loading models.
- **`utils/plotting.py`**: Provides a function to visualize similar images and their respective distance scores.
- **`main.py`**: The entry point of the project, responsible for orchestrating the training, building the index, and performing similarity searches.

## Installation

To run this project, you will need to install the required dependencies. It is recommended to create a virtual environment.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/image_similarity_search.git
   cd image_similarity_search
   ```

## Usage

python main.py



