# Self-Pruning Neural Network

## Overview
This project implements a convolutional neural network that learns to prune its own weights during training using learnable gates on both convolutional and linear layers.

## Key Idea
Each weight is multiplied by a sigmoid gate. L1 regularization is applied to encourage sparsity.

## Architecture
- 3 Convolutional layers with Batch Normalization and Max Pooling
- 3 Fully Connected layers with Dropout
- PrunableConv2d and PrunableLinear layers with learnable gates

## Loss Function
Total Loss = CrossEntropy + λ × Sparsity Loss

## Training Details
- Optimizer: SGD with momentum and weight decay
- Learning Rate Scheduler: Step decay
- Data Augmentation: Random crop and horizontal flip
- Normalization: CIFAR-10 mean and std

## Results

| Lambda | Accuracy | Sparsity |
|--------|---------|---------|
| 1e-4   |         |         |
| 5e-4   |         |         |
| 1e-3   |         |         |
| 5e-3   |         |         |

## Observation
Higher lambda → more sparsity but lower accuracy.