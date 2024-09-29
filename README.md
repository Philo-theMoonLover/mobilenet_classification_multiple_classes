# Classification for Multiple Class

## Overview

This project implements a general framework for multi-class image classification using **MobileNetV3**. It can be easily adapted to different datasets and use cases. The primary goal is to provide a flexible and easy-to-use classification model that can handle various numbers of classes and datasets.

## Features

- **Customizable Data Preprocessing**: Easily apply different transformations to your dataset using `torchvision.transforms`.

- **Model Training and Evaluation**: Supports training, validation, and early stopping.

- **Pre-trained Model**: Utilizes **MobileNetV3** pre-trained on ImageNet, which can be fine-tuned for specific applications.

- **Flexible Class Handling**: No hard-coded class names or numbers, making it adaptable for any multi-class dataset.


## Dataset Structure

dataset/ \
├── train/ \
│   ├── class_01/ \
│   ├── class_02/ \
│   └── class_03/ \
├── valid/ \
│   ├── class_01/ \
│   ├── class_02/ \
│   └── class_03/ \
└── test/ \
    ├── class_01/ \
    ├── class_02/ \
    └── class_03/ \

