# Image Similarity Analysis

This repository contains code for analyzing the similarity between a query image and a dataset of images using five different methods: ORB, ViT, CNN, Perceptual Hashing, and CLIP. The goal is to classify images as relevant or irrelevant to the query image based on similarity scores.

## Table of Contents

1. [Introduction](#introduction)
2. [Methods](#methods)
3. [Performance Analysis](#performance-analysis)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction

Image similarity analysis is a crucial task in various applications such as image retrieval, content-based image search, and object recognition. This repository provides a comprehensive analysis of five different methods for image similarity: ORB, ViT, CNN, Perceptual Hashing, and CLIP. Each method has its own strengths and weaknesses, and the goal is to compare their performance in terms of precision, recall, and accuracy.

## Methods

### 1. ORB (Oriented FAST and Rotated BRIEF)

ORB is a feature detection and description algorithm that is robust to rotation and scale changes. It detects keypoints in images and describes them using binary descriptors. The similarity between images is computed based on the number of matching keypoints.

### 2. ViT (Vision Transformer)

ViT is a transformer-based model that applies the transformer architecture to image data. It extracts features from images and computes similarity scores using cosine similarity. ViT has shown promising results in various computer vision tasks.

### 3. CNN (Convolutional Neural Network)

CNNs are widely used in image classification and object detection tasks. In this analysis, a pre-trained ResNet50 model is used to extract features from images, and similarity scores are computed using cosine similarity.

### 4. Perceptual Hashing

Perceptual Hashing is a technique that generates a hash value for an image based on its perceptual features. The similarity between images is computed based on the Hamming distance between their hash values.

### 5. CLIP (Contrastive Language-Image Pre-training)

CLIP is a model that learns visual concepts from natural language supervision. It extracts features from images and computes similarity scores using cosine similarity. CLIP has shown state-of-the-art performance in various image retrieval tasks.

## Performance Analysis

The performance of each method is analyzed based on precision, recall, and accuracy. The similarity scores are used to classify images as relevant or irrelevant to the query image. The optimal threshold for classification is determined based on the mean of the similarity scores.

### Fine-Tuning for Computational Efficiency and Scalability

To achieve computational efficiency and scalability for real-time usage scenarios, several fine-tuning techniques can be applied:

1. **Batch Processing**: Process images in batches to utilize GPU resources efficiently.
2. **Feature Caching**: Precompute and cache features for dataset images to avoid redundant computations.
3. **Model Pruning**: Prune the models to reduce their size and improve inference speed.
4. **Quantization**: Quantize the models to reduce memory usage and improve inference speed.

## Results

The results of the analysis are presented in the table below:

| Method          | Precision | Recall | Accuracy | Time Taken (seconds) |
|-----------------|-----------|--------|----------|----------------------|
| ORB Similarity  | 0.6000    | 1.0000 | 0.7778   | 1.04                 |
| ViT Similarity  | 0.4000    | 0.6667 | 0.5556   | 8.47                 |
| CNN Similarity  | 0.6000    | 1.0000 | 0.7778   | 2.87                 |
| Perceptual Hashing | 0.3333  | 0.6667 | 0.4444   | 0.08                 |
| CLIP            | 0.6000    | 1.0000 | 0.7778   | 37.57                |

### Insights

- **ORB Similarity**: ORB provides good precision and recall but is computationally efficient. It is suitable for real-time applications.
- **ViT Similarity**: ViT provides moderate precision and recall but is computationally intensive. It is suitable for applications where high accuracy is required.
- **CNN Similarity**: CNN provides good precision and recall and is computationally efficient. It is suitable for real-time applications.
- **Perceptual Hashing**: Perceptual Hashing provides low precision and recall but is extremely fast. It is suitable for applications where speed is critical.
- **CLIP**: CLIP provides good precision and recall but is computationally intensive. It is suitable for applications where high accuracy is required.

## Conclusion

The analysis shows that each method has its own strengths and weaknesses. The choice of method depends on the specific requirements of the application, such as computational efficiency, scalability, and accuracy. Fine-tuning techniques can be applied to improve the performance of each method for real-time usage scenarios.

## References

1. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. In 2011 International Conference on Computer Vision (pp. 2564-2571). IEEE.
2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
4. Zauner, M., Breuel, T. M., & Bischof, H. (2010). Implementation of an image hash function based on perceptual hashing. In Proceedings of the 2010 ACM symposium on Applied computing (pp. 1897-1900).
5. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020.
