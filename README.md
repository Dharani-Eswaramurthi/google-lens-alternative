# Image Similarity Analysis 🔍

This repository contains code and analysis for evaluating various image similarity methods. The methods include ORB (Oriented FAST and Rotated BRIEF), ViT (Vision Transformer), CNN (Convolutional Neural Networks), Perceptual Hashing, and CLIP (Contrastive Language-Image Pre-training). The analysis is performed on three different datasets to compare the performance of these methods.

# Shoutout to 🚀

A huge shoutout to the **Shoppin** organization for being the kickstart motivation for exploring this use case! Before diving into this project, I wasn't familiar with these functions and used various resources to understand and implement them. Starting from scratch, debugging, scrutinizing, and improving has been a rewarding journey. Thanks to this, I had the chance to increase the weightage of my GitHub repo and showcase a trailer of my abilities. 🌟

## Table of Contents 📑
1. [Introduction](#introduction)
2. [Code Implementation](#code-implementation)
3. [Methods](#methods)
4. [Code Documentation](#code-documentation)
5. [Datasets](#datasets)
6. [Results and Analysis](#results-and-analysis)
7. [Performance Comparison](#performance-comparison)
8. [Fine-Tuning for Computational Efficiency](#fine-tuning-for-computational-efficiency)
9. [Citations](#citations)

## Introduction 🧠

Image similarity is a fundamental task in computer vision with applications in image retrieval, object recognition, and more. This repository evaluates five different image similarity methods on three diverse datasets to understand their strengths, weaknesses, and suitability for various applications.

## Code Implementation

1. Open the given `.ipynb` file in Google Colab or Jupyter Notebook.
2. Upload or copy the given dataset zips to the default directory. Do not extract it.
3. Start running the code, it will do the rest for you.

## Methods 🛠️

### 1. ORB (Oriented FAST and Rotated BRIEF) 🔑
- **Why Use ORB?**: ORB is a fast and efficient method for detecting and describing keypoints in images. It is suitable for applications where speed is a priority.
- **What It Does**: ORB detects keypoints using the FAST algorithm and describes them using the BRIEF descriptor. It then matches these descriptors between query and dataset images.
- **Fine-Tuning**: Adjust the number of features (`nfeatures`) and the match distance threshold to balance speed and accuracy.

### 2. ViT (Vision Transformer) 🖼️
- **Why Use ViT?**: ViT is designed to capture long-range dependencies and global context in images, making it highly effective for complex image similarity tasks.
- **What It Does**: ViT extracts features from images using a transformer-based architecture and computes similarity based on these features.
- **Fine-Tuning**: Use a smaller model or reduce the input image size to improve computational efficiency.

### 3. CNN (Convolutional Neural Networks) 🧠
- **Why Use CNN?**: CNNs are robust to variations in position, angle, and background, making them suitable for a wide range of applications.
- **What It Does**: CNNs extract hierarchical features from images using convolutional layers and compute similarity based on these features.
- **Fine-Tuning**: Use a lighter model like MobileNet or EfficientNet for real-time applications.

### 4. Perceptual Hashing 🔍
- **Why Use Perceptual Hashing?**: Perceptual hashing is extremely fast and suitable for quick, approximate similarity checks.
- **What It Does**: Perceptual hashing generates a compact hash representation of the image and computes similarity based on the Hamming distance between hashes.
- **Fine-Tuning**: Adjust the hash size to balance speed and accuracy.

### 5. CLIP (Contrastive Language-Image Pre-training) 🧳
- **Why Use CLIP?**: CLIP is designed to understand both visual and textual information, making it highly effective for multimodal tasks.
- **What It Does**: CLIP extracts features from images using a model trained on a large dataset of image-text pairs and computes similarity based on these features.
- **Fine-Tuning**: Use a smaller model or reduce the input image size to improve computational efficiency.

## Code Documentation 📝

The code is organized into several functions for validating image paths, loading ground truth images, and computing similarity using the five methods. Below is a brief overview of the key functions:

### Utility Functions 🛠️
- **validate_image_paths**: Validates that the provided image paths exist and are of correct format.
- **load_ground_truth**: Loads ground truth images from a specified folder.

### Similarity Methods 💡
- **orb_similarity**: Computes similarity using the ORB algorithm.
- **vit_similarity**: Computes similarity using the Vision Transformer.
- **cnn_similarity**: Computes similarity using a pre-trained ResNet50 model.
- **phash_similarity**: Computes similarity using Perceptual Hashing.
- **clip_similarity**: Computes similarity using the CLIP model.

### Performance Analysis 📊
- **analyze_performance**: Measures the time taken to compute similarities and evaluates precision, recall, and retrieval accuracy.

## Datasets 📚

The analysis is performed on three different datasets:
1. **Dataset 1**: Contains images of animals (e.g., cats, dogs, wolves).
2. **Dataset 2**: Contains images of electronic devices (e.g., iPhones, Samsung phones).
3. **Dataset 3**: Contains images of cartoon characters (e.g., Doraemon, Pokemon).

## Results and Analysis 📈

### Results Table

| Dataset | Method       | Precision | Recall | Retrieval Accuracy | Time Taken (seconds) |
|---------|--------------|-----------|--------|--------------------|----------------------|
| Dataset 1 | ORB         | 0.4000    | 0.2222 | 0.6000             | 2.81                 |
|           | ViT         | 0.8750    | 0.7778 | 0.8800             | 19.58                |
|           | CNN         | 0.7778    | 0.7778 | 0.8400             | 8.18                 |
|           | Perceptual Hashing | 0.4545  | 0.5556 | 0.6000             | 0.26                 |
|           | CLIP        | 0.8000    | 0.8889 | 0.8800             | 99.20                |
| Dataset 2 | ORB         | 0.2500    | 0.4000 | 0.4000             | 0.20                 |
|           | ViT         | 0.4286    | 0.6000 | 0.6000             | 6.00                 |
|           | CNN         | 0.5000    | 0.8000 | 0.6667             | 2.32                 |
|           | Perceptual Hashing | 0.2857  | 0.4000 | 0.4667             | 0.06                 |
|           | CLIP        | 0.5714    | 0.8000 | 0.7333             | 31.73                |
| Dataset 3 | ORB         | 0.8000    | 0.8000 | 0.7778             | 0.86                 |
|           | ViT         | 0.8000    | 0.8000 | 0.7778             | 5.20                 |
|           | CNN         | 0.8000    | 0.8000 | 0.7778             | 1.57                 |
|           | Perceptual Hashing | 0.4000  | 0.4000 | 0.3333             | 0.05                 |
|           | CLIP        | 1.0000    | 1.0000 | 1.0000             | 18.57                |

## Performance Comparison ⚖️

### Precision, Recall, and Accuracy 📏
- **ORB**: Provides good results when images are closely similar but struggles with different positions or angles. It is fast but lacks in understanding global features.
- **ViT**: Provides high precision and recall but is slower due to its complexity. It understands complex features and relationships within the image.
- **CNN**: Offers a good balance between speed and accuracy. It is robust to variations in position, angle, and background.
- **Perceptual Hashing**: Extremely fast but lacks precision and recall. Suitable for quick, approximate similarity checks.
- **CLIP**: Provides the highest precision and recall but is the slowest. It understands semantic features and is robust to variations in image content.

### Computational Efficiency ⚡
- **ORB**: Fast but limited in feature understanding.
- **ViT**: Slower due to complexity but highly effective for complex tasks.
- **CNN**: Balanced between speed and accuracy.
- **Perceptual Hashing**: Extremely fast but limited in accuracy.
- **CLIP**: Slowest but most accurate and robust.

## Fine-Tuning for Computational Efficiency 🛠️

### ORB
- **Adjust `nfeatures`**: Reduce the number of features to improve speed.
- **Match Distance Threshold**: Adjust the threshold to balance speed and accuracy.

### ViT
- **Smaller Model**: Use a smaller ViT model or reduce the input image size.
- **Batch Processing**: Process images in batches to improve efficiency.

### CNN
- **Lighter Model**: Use a lighter model like MobileNet or EfficientNet.
- **Batch Processing**: Process images in batches to improve efficiency.

### Perceptual Hashing
- **Hash Size**: Adjust the hash size to balance speed and accuracy.

### CLIP
- **Smaller Model**: Use a smaller CLIP model or reduce the input image size.
- **Batch Processing**: Process images in batches to improve efficiency.

## Citations 📚

- **ORB**: Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. In 2011 International Conference on Computer Vision (pp. 2564-2571). IEEE.
- **ViT**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
- **CNN**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- **Perceptual Hashing**: Zauner, C. (2010). Implementation of a perceptual image hash function standard. In 2010 IEEE International Conference on Image Processing (pp. 3421-3424). IEEE.
- **CLIP**: Radford, A., Kim, J. W., Hallacy, C., Ramesh, P., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020.

## Conclusion 🎉

This repository provides a comprehensive analysis of five image similarity methods on three diverse datasets. The results highlight the strengths and weaknesses of each method, providing insights into their suitability for various applications. Fine-tuning and optimizations are suggested to improve computational efficiency and scalability for real-time usage scenarios. ✨
