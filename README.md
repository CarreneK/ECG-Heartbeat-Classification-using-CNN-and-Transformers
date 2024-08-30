# ECG Heartbeat Classification using CNN and Transformers

## Overview
This project implements two deep learning models—Convolutional Neural Networks (CNN) and Transformers—to classify ECG heartbeats using the MIT-BIH Arrhythmia dataset. The goal is to explore and compare the performance of these models in accurately classifying different types of heartbeats.

## Dataset
The project utilizes the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/content/mitdb/1.0.0/), a widely used dataset for evaluating algorithms that classify ECG heartbeats. It contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects.

## Model Architectures

### CNN Model
The CNN model is based on the architecture described in the research paper *"ECG Heartbeat Classification: A Deep Transferable Representation"*. It consists of several convolutional layers with ReLU activations, followed by max pooling layers. The final layer is a softmax layer for classification.

### Transformer Model
The Transformer model is built using multi-head attention layers and feedforward networks. It includes several transformer blocks followed by a global average pooling layer and dense layers for classification.

## Results
The performance of the models was evaluated using accuracy, precision, recall, and F1-score on the test set. The CNN model outperformed the Transformer model, consistent with the findings of the referenced research.

**CNN Model:**
- **Accuracy:** 99.94%
- **Precision:** 98%
- **Recall:** 98%
- **F1-Score:** 98%

**Transformer Model:**
- **Accuracy:** 41%
- **Precision:** 75%
- **Recall:** 12%
- **F1-Score:** 10%

## Future Work
- Experiment with different Transformer architectures to improve performance on ECG data.
- Implement data augmentation techniques to improve model generalization.
- Explore the integration of hybrid models combining CNN and Transformer features.

## References
- ["ECG Heartbeat Classification: A Deep Transferable Representation"](https://arxiv.org/pdf/1805.00794)
- [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/content/mitdb/1.0.0/)
