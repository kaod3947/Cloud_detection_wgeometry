# Cloud and Cloud Shadow Segmentation for Satellite Imagery

This project focuses on cloud and cloud shadow segmentation 
in satellite imagery using deep learning. 

To overcome spectral limitations (RGBN only), 
geometric information such as solar and sensor angles 
is integrated into the model architecture.

## Table of Contents
- [Introduction](#Introduction)
- [Method](#Method)
- [Setup](#Setup)
- [Results](#Results)
- [Conclusions](#Conclusions)

## Introduction
test

## Method
Training cloud/cloud shadow detection model with LandSat8/9 imagery(only RGBN bands)
1. Compare between two train method (Use only imagery/Use with angle information)  [ ] 
- Backbone: DeepLabV3+
- Input: RGBN + angle features
- Fusion Strategy: Early feature fusion
- Loss: Cross-entropy / Focal Loss

2. [ ] Compare with other cloud/cloud shadow detection model
3. [ ] Develop model using different architecture

## Setup
- Python
- Required Python package (requirements.txt)

### Repository Structure
models/        → network architecture
datasets/      → data loader
train.py       → training script
inference.py   → inference script

### Installation
pip install -r requirements.txt

### Train
python train.py

## Results
| Model | IoU |
|-------|------|
| Baseline | 0.72 |
| + Angle Fusion | 0.78 |

## Conclusions
- Multi-satellite generalization
- Domain adaptation
- Temporal consistency modeling

### Finding
*TBA*

### Future Work
- Multi-satellite generalization
- Domain adaptation
- Temporal consistency modeling
