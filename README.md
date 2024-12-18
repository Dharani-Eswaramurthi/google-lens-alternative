# Image Similarity Analysis

This repository contains code for analyzing the similarity between query and dataset images using various methods, including ORB, ViT, CNN (VGG16), Perceptual Hashing, and CLIP. The performance of each method is evaluated based on precision, recall, and retrieval accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
  - [ORB Similarity](#orb-similarity)
  - [ViT Similarity](#vit-similarity)
  - [CNN (VGG16) Similarity](#cnn-vgg16-similarity)
  - [Perceptual Hashing](#perceptual-hashing)
  - [CLIP Similarity](#clip-similarity)
- [Performance Analysis](#performance-analysis)
- [Results](#results)
- [References](#references)

## Introduction

Image similarity analysis is a crucial task in computer vision, with applications in image retrieval, object recognition, and more. This repository provides implementations of various image similarity methods and evaluates their performance based on precision, recall, and retrieval accuracy.

## Methods

### ORB Similarity

ORB (Oriented FAST and Rotated BRIEF) is a feature detection and description algorithm that is robust to rotation and scale changes. It detects keypoints in images and describes them using binary descriptors.

**Implementation**:
- Uses OpenCV's ORB detector and BFMatcher for feature matching.
- Normalizes the similarity score based on the number of matched keypoints.

**Fine-Tuning**:
- Adjust the number of features (`nfeatures`) and match distance threshold (`match_distance_threshold`) based on the dataset characteristics.
- Use GPU acceleration for ORB feature extraction if available.

### ViT Similarity

Vision Transformer (ViT) is a deep learning model that applies the Transformer architecture to image data. It extracts features from images and computes similarity based on these features.

**Implementation**:
- Uses the `ViTFeatureExtractor` and `ViTModel` from the Hugging Face Transformers library.
- Precomputes features for query and dataset images for efficiency.

**Fine-Tuning**:
- Adjust the batch size based on available GPU memory.
- Consider using a more recent ViT model if available.

### CNN (VGG16) Similarity

Convolutional Neural Networks (CNNs) are widely used for image classification and feature extraction. VGG16 is a popular CNN architecture that extracts features from images.

**Implementation**:
- Uses the VGG16 model from the Torchvision library.
- Precomputes features for dataset images for efficiency.

**Fine-Tuning**:
- Adjust the batch size based on available GPU memory.
- Consider using a more recent CNN model if available.

### Perceptual Hashing

Perceptual Hashing is a technique that generates a hash value for an image based on its perceptual features. It computes similarity based on the Hamming distance between hash values.

**Implementation**:
- Uses the `imagehash` library for perceptual hashing.
- Precomputes hash values for dataset images for efficiency.

**Fine-Tuning**:
- Consider using other hashing algorithms if needed.
- Adjust the similarity metric based on the dataset characteristics.

### CLIP Similarity

CLIP (Contrastive Language-Image Pre-training) is a model that learns visual concepts from natural language supervision. It extracts features from images and computes similarity based on these features.

**Implementation**:
- Uses the `CLIPProcessor` and `CLIPModel` from the Hugging Face Transformers library.
- Precomputes features for dataset images for efficiency.

**Fine-Tuning**:
- Adjust the batch size based on available GPU memory.
- Consider using a more recent CLIP model if available.

## Performance Analysis

The performance of each method is evaluated based on precision, recall, and retrieval accuracy. The optimal threshold for classifying similarity scores is determined using the median of the scores.

**Metrics**:
- **Precision**: The ratio of correctly retrieved images (true positives) to the total number of retrieved images.
- **Recall**: The ratio of correctly retrieved images (true positives) to the total number of relevant images in the dataset.
- **Retrieval Accuracy**: The ratio of correctly retrieved images (true positives) to the total number of images in the dataset.

## Results

### ORB Similarity

- **Precision**: 0.4000
- **Recall**: 1.0000
- **Retrieval Accuracy**: 0.6667
- **Time Taken**: 1.76 seconds

### ViT Similarity

- **Precision**: 0.5000
- **Recall**: 1.0000
- **Retrieval Accuracy**: 0.7778
- **Time Taken**: 6.88 seconds

### CNN (VGG16) Similarity

- **Precision**: 0.3333
- **Recall**: 1.0000
- **Retrieval Accuracy**: 0.5556
- **Time Taken**: 9.79 seconds

### Perceptual Hashing

- **Precision**: 0.3333
- **Recall**: 1.0000
- **Retrieval Accuracy**: 0.5556
- **Time Taken**: 0.09 seconds

### CLIP Similarity

- **Precision**: 0.5000
- **Recall**: 1.0000
- **Retrieval Accuracy**: 0.7778
- **Time Taken**: 3.56 seconds

## References

- Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. In 2011 International Conference on Computer Vision (pp. 2564-2571). IEEE.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
- Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
- Radhakrishnan, R., Zitnick, C. L., & Parikh, D. (2021). CLIP: Connecting text and images. arXiv preprint arXiv:2103.00020.

