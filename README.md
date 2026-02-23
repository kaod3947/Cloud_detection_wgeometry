# Cloud and Cloud Shadow Segmentation for High-Resolution Satellite Imagery

## Overview
This project focuses on cloud and cloud shadow segmentation 
in high-resolution satellite imagery using deep learning. 

To overcome spectral limitations (RGBN only), 
geometric information such as solar and sensor angles 
is integrated into the model architecture.

## Motivation
Accurate cloud masking is critical for reliable 
surface analysis and downstream geospatial applications. 
However, high-resolution commercial imagery often lacks 
rich spectral bands, making cloud detection challenging.

## Method
- Backbone: DeepLabV3
- Input: RGBN + angle features
- Fusion Strategy: Early feature fusion
- Loss: Cross-entropy / Focal Loss

## Repository Structure
models/        → network architecture
datasets/      → data loader
train.py       → training script
inference.py   → inference script

## Installation

pip install -r requirements.txt

## Train

python train.py

## Results
| Model | IoU |
|-------|------|
| Baseline | 0.72 |
| + Angle Fusion | 0.78 |

## Future Work

- Multi-satellite generalization
- Domain adaptation
- Temporal consistency modeling
