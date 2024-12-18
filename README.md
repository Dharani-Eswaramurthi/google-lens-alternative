# Image Similarity Analysis

This repository contains code for analyzing image similarity using various methods, including ORB (Oriented FAST and Rotated BRIEF), ViT (Vision Transformer), CNN (Convolutional Neural Network), Perceptual Hashing, and CLIP (Contrastive Language-Image Pre-training). The code computes similarity scores between a query image and a dataset of images, classifies images as relevant or irrelevant based on these scores, and evaluates the performance of each method.

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
  - [ORB Similarity](#orb-similarity)
  - [ViT Similarity](#vit-similarity)
  - [CNN Similarity](#cnn-similarity)
  - [Perceptual Hashing](#perceptual-hashing)
  - [CLIP Similarity](#clip-similarity)
- [Performance Analysis](#performance-analysis)
- [Results](#results)
- [References](#references)

## Introduction

Image similarity analysis is a crucial task in computer vision with applications in image retrieval, object recognition, and content-based image search. This repository provides a comprehensive comparison of different image similarity methods, evaluating their performance in terms of precision, recall, retrieval accuracy, and computation time.

## Methods

### ORB Similarity

ORB (Oriented FAST and Rotated BRIEF) is a feature detection and description algorithm that is robust to rotation and scale changes. It detects keypoints and computes descriptors for these keypoints, which are then matched between images to compute similarity scores.

### ViT Similarity

ViT (Vision Transformer) is a transformer-based model for image classification that has shown state-of-the-art performance on various benchmarks. It processes images as sequences of patches and uses self-attention mechanisms to capture global dependencies.

### CNN Similarity

CNN (Convolutional Neural Network) is a deep learning model widely used for image classification tasks. In this repository, we use a pre-trained ResNet50 model to extract features from images and compute similarity scores based on these features.

### Perceptual Hashing

Perceptual Hashing is a technique that generates a hash value for an image based on its perceptual features. The similarity between images is computed based on the Hamming distance between their hash values.

### CLIP Similarity

CLIP (Contrastive Language-Image Pre-training) is a model that learns visual concepts from natural language supervision. It can be used to compute similarity scores between images based on their feature representations.

## Performance Analysis

The performance of each method is evaluated using precision, recall, retrieval accuracy, and computation time. The results are analyzed to compare the effectiveness and efficiency of each method.

## Results

### ORB Similarity

- **Precision:** 1.0000
- **Recall:** 1.0000
- **Retrieval Accuracy:** 1.0000
- **Time Taken:** 1.05 seconds

### ViT Similarity

- **Precision:** 0.6667
- **Recall:** 0.6667
- **Retrieval Accuracy:** 0.7778
- **Time Taken:** 8.16 seconds

### CNN Similarity

- **Precision:** 0.6667
- **Recall:** 0.6667
- **Retrieval Accuracy:** 0.7778
- **Time Taken:** 3.70 seconds

### Perceptual Hashing

- **Precision:** 0.6667
- **Recall:** 0.6667
- **Retrieval Accuracy:** 0.7778
- **Time Taken:** 0.09 seconds

### CLIP Similarity

- **Precision:** 1.0000
- **Recall:** 1.0000
- **Retrieval Accuracy:** 1.0000
- **Time Taken:** 38.10 seconds

### Analysis

- **ORB Similarity:** Achieved perfect precision, recall, and retrieval accuracy with a relatively short computation time. This method is highly effective for keypoint-based image matching.
- **ViT Similarity:** Showed moderate performance in terms of precision, recall, and retrieval accuracy. The computation time was longer compared to other methods, indicating that ViT may not be the most efficient choice for real-time applications.
- **CNN Similarity:** Demonstrated similar performance to ViT in terms of precision, recall, and retrieval accuracy. The computation time was shorter than ViT, making it a more efficient option.
- **Perceptual Hashing:** Achieved moderate performance with the shortest computation time. This method is highly efficient but may not capture complex visual similarities as effectively as other methods.
- **CLIP Similarity:** Achieved perfect precision, recall, and retrieval accuracy but with the longest computation time. CLIP is highly effective for capturing complex visual similarities but may not be suitable for real-time applications due to its high computational cost.

## References

- Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. In 2011 International Conference on Computer Vision (pp. 2564-2571). IEEE.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- Zauner, C. (2010). Implementation of a perceptual image hash function. In Proceedings of the 12th ACM workshop on Multimedia and security (pp. 11-16).
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020.

