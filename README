# Multi-Class Aerial Object Detection — DOTA v1.0

A complete machine learning pipeline for multi-class object detection in aerial images, using the DOTA v1.0 benchmark dataset. The dataset contains **15 object categories** captured in high-resolution satellite and aerial imagery.

Due to the heavy computational cost of working with the full dataset, a stratified sample of **300 images** (approximately 16% of the available data) was used throughout the study.

Three models were built and compared:

- **SimpleCNN** — a basic crop-level classifier used as a baseline
- **Faster R-CNN** — with a MobileNetV3 backbone for two-stage detection
- **YOLOv8-Large** — a modern single-stage detector

Extensive data preprocessing was carried out, including image tiling, targeted augmentation for rare classes, and format conversion. The results show that **YOLOv8-Large clearly outperforms** the other two models, achieving a test **mAP@0.5 of 0.5212**. Faster R-CNN reached a test mAP@0.5 of 0.0677, and the CNN baseline achieved 0.2067 test accuracy on cropped objects.

---

## Repository structure

```
.
├── notebook/                                  # full pipeline notebook
│   └── aerial_object_detection.ipynb
│
└── google_drive_link_for_the_datasets.txt     # link to download the datasets
```

---

## Background and Problem Statement

Detecting objects in aerial and satellite images is a well-known challenge in computer vision. Unlike regular photographs, aerial images come with very large dimensions, objects that appear at different angles and scales, and significant class imbalance. The DOTA v1.0 dataset (Xia et al., 2018) was created specifically to address this problem. It provides over 188,000 annotated object instances across 15 categories, captured from a range of platforms and altitudes.

High-resolution imagery means images can be up to 20,000 by 20,000 pixels. Objects are small relative to the image, oriented at arbitrary angles, and densely packed, and highly imbalanced.

Standard detection models assume fixed input sizes and upright objects, which makes them poorly suited to raw aerial data. This project addresses these challenges by building a complete detection pipeline: from data exploration through to model evaluation, that is practical to run on limited compute resources.

### Objectives

1. Understand the dataset through exploratory data analysis
2. Set up proper preprocessing steps including tiling and augmentation
3. Train and compare three progressively capable models using appropriate evaluation metrics

---

## Exploratory Data Analysis

### Dataset Overview

DOTA v1.0 provides predefined train/val/test splits, but only the original train and validation sets were used because they include ground-truth labels. These were merged into 1,865 labelled images with around 128k instances, then re-split for evaluation. To reduce compute cost, 300 images were selected using dominant-class stratified sampling to preserve the original class distribution.

### Class Distribution

The dataset shows strong class imbalance, with **ship** and **small-vehicle** appearing most often, while **ground-track-field** and **soccer-ball-field** are rare. This imbalance remains in the 300-image stratified sample, with a **73:1** ratio between the most and least frequent classes.

### Image Dimensions and Object Scale

Image dimensions vary widely, with (387–7530)px in width, (278–6533)px in height, and clustering around 1000px and 4000px, representing bimodal distribution. This necessitates image tiling, since detection models need consistent input sizes. Object scale varies enormously across classes. Large classes such as **ground-track-field** and **soccer-ball-field** occupy much larger areas than **helicopters** or **small vehicles**, and object counts differ strongly between images.

### Key EDA Findings

| Finding | Pipeline Impact |
|---------|----------------|
| 15 classes, consistent format | Fixed 15-class output layer |
| ~73x class imbalance | Augmentation for rare classes + class-weighted loss |
| Image sizes ~300 to ~7000px | Image tiling required |
| Oriented bounding boxes | OBB to AABB conversion for standard detectors |
| No negative coordinates found | Annotations geometrically valid for direct use |

---

## Data Preprocessing and Feature Selection

### Stratified Re-Splitting

The 300-image sample was re-split into 70% training (210 images), 15% validation (45), and 15% test (45) using image-level stratified sampling. This produced 11,599 training, 3,471 validation, and 2,862 test instances, with all 15 classes represented in each split.

### Data Augmentation for Rare Classes

To reduce the impact of class imbalance, images containing rare classes were augmented including flips, rotations, and affine transforms, alongside brightness changes, noise injection, and CLAHE. Three augmented copies were generated per rare-class image, while bounding box coordinates were safely transformed using Albumentations' keypoint-based pipeline (Buslaev et al., 2020).

### Image Tiling

All images were tiled into 512x512 patches with 100px overlap (to avoid cutting off boundary objects). Tiling was applied separately to all three sets to avoid data leakage. Empty train/validation tiles were discarded, producing about 2980 train, 803 validation, and 1867 test tiles.

### Format Conversion and Class Weights

DOTA oriented boxes were converted to axis-aligned boxes for Faster R-CNN, and then to normalised YOLO format for YOLOv8. Class weights were computed using inverse frequency to help models learn rare categories.

---

## Machine Learning Models

### Model 1: SimpleCNN Classifier (Baseline)

A basic crop-level baseline. It classifies resized 224x224 object crops using three convolutional blocks and fully connected layers. It does not perform detection but provides a simple performance floor for comparison (He et al., 2016).

- Trained for **15 epochs** using **AdamW**, cosine annealing, and class-weighted cross-entropy loss
- Early stopping with patience of 6 epochs; image size 224x224; batch size 16; learning rate 8e-4
- Achieved **20.67% test accuracy**, weak and highly biased — crop-based classification was not reliable for aerial object detection

### Model 2: Faster R-CNN (MobileNetV3-FPN)

A two-stage detector for object localisation and classification on full image tiles. Used a MobileNetV3-Large + FPN backbone with ImageNet pre-trained weights. The detection head was trained for the 15 DOTA classes plus background.

- MobileNetV3-Large + FPN backbone, 16 classes including background, NMS 0.3, 100 max detections
- SGD with momentum 0.9, weight decay 1e-4, learning rate 1e-3 with OneCycleLR, trained for 10 epochs with early stopping patience 5
- Achieved a low test **mAP@0.5 of 0.0677**, with the best results for tennis-court (0.457 AP) and plane (0.186 AP)

### Model 3: YOLOv8-Large

A modern single-stage, anchor-free detector (Jocher et al., 2023) that predicts boxes and classes in one pass. Faster than Faster R-CNN and better suited to dense aerial scenes.

- Trained for 15 epochs on 512x512 tiles with batch size 4 using YOLO-format labels
- Early stopping with patience 5; augmentations including Mosaic, MixUp, HSV changes, translation, scaling, and horizontal flipping
- Achieved **0.5212 mAP@0.5** and 0.3205 mAP@0.5:0.95
- Best on tennis-court (0.965 AP), ship (0.897 AP), and plane (0.727 AP)

---

## Results Comparison Across Models

| Model | Metric | Score |
|-------|--------|-------|
| **SimpleCNN** | Test accuracy (crops) | 0.2067 |
| **Faster R-CNN** | mAP@0.5 | 0.0677 |
| **YOLOv8-Large** | mAP@0.5 | **0.5212** |

YOLOv8-Large was the strongest model, achieving **0.5212 mAP@0.5**, over seven times higher than Faster R-CNN's 0.0677. SimpleCNN is not directly comparable because it only reports crop-level accuracy and does not perform localisation.

There is moderate positive correlation (**r = 0.68**) between training samples per class and YOLOv8 test AP. This suggests more samples generally improve performance, but visual clarity also matters — tennis-court performs well, while ground-track-field and swimming-pool remain difficult.

---

## Performance Measures and Evaluation Strategies

Different metrics were used because the models perform different tasks. Faster R-CNN and YOLOv8 were evaluated using mAP@0.5, while YOLOv8 also reported the stricter mAP@0.5:0.95. SimpleCNN used crop-level accuracy because it only classifies cropped objects and does not localise them. Cross-validation was not used because it is costly for object detection and may cause leakage between overlapping aerial tiles. Instead, an image-level train/validation/test split was used.

---

## Conclusion, Recommendations, and Future Work

This project built a complete DOTA v1.0 aerial object detection pipeline covering preprocessing, tiling, augmentation, training, and evaluation. Three models were compared: **SimpleCNN**, **Faster R-CNN**, and **YOLOv8-Large**. YOLOv8-Large performed best with **0.5212 mAP@0.5**, although performance was limited by class imbalance, few epochs, fixed tile size, high GPU demand, axis-aligned box conversion, and weaker detection of small or minority-class objects.

Future work should use the full dataset, improve class balancing, apply multi-scale or SAHI-based inference, test oriented detectors for rotated objects, and explore transformer-based models such as DETR. Ethical considerations were met by using a public research dataset, but real-world aerial detection should remain transparent, regulated, and privacy aware.

---

## Dataset

The dataset link is provided in `google_drive_link_for_the_datasets.txt`.

**Source:** DOTA v1.0 (Xia et al., 2018) — a public benchmark for object detection in aerial images.

---

## Requirements

- Python 3.10+
- PyTorch, TorchVision
- Ultralytics (YOLOv8)
- OpenCV, NumPy, pandas, scikit-learn
- Albumentations
- Matplotlib, Seaborn

---

## Acknowledgements

Developed as part of an MSc project (2025/26).