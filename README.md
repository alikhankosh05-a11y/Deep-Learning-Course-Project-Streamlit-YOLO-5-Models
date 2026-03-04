# CNN-Based Product Recognition & Price Estimation

**Deep Learning Course Project**  
**Student:** Alikhan Koshamet  
**Date:** 4-th March 2026

## Project Overview
This project implements a system that recognizes multiple products from a single image using CNN models and calculates their total price. The system uses object detection (YOLOv8) + classification with 5 different CNN architectures.

## Features
- Recognition of 15 product categories
- Multi-product detection in one photo
- Automatic price calculation
- 5 trained CNN models with comparison
- User-friendly Streamlit web application

## Dataset
- **Domain:** Stationery and electronics (office supplies)
- **Number of classes:** 15
- **Images:** 253 (train: 214, val: 39)
- **Source:** Custom collected dataset
- **Preprocessing:** Data augmentation, resize 224×224, normalization

**Dataset examples:**

![Dataset samples](screenshots/dataset_examples.png)

## Models Implemented
Trained 5 CNN architectures using transfer learning:

| Model            | Test Accuracy | Precision | Recall  | F1-score | Size (MB) |
|------------------|---------------|-----------|---------|----------|-----------|
| AlexNet          | 90.00%        | 0.9117    | 0.9000  | 0.9001   | 217.69    |
| VGG16            | 95.00%        | 0.9688    | 0.9500  | 0.9509   | 512.41    |
| GoogLeNet        | 86.67%        | 0.8649    | 0.8667  | 0.8497   | 21.59     |
| **ResNet50**     | **100%**      | **1.0000**| **1.0000**| **1.0000**| 90.09     |
| EfficientNet-B0  | 86.67%        | 0.8558    | 0.8667  | 0.8458   | 15.64     |

**Best model:** ResNet50

## Web Application Demo
**Live demo:** `streamlit run streamlit_dl.py`

### Main Interface
![Main Interface](/Users/allikhankoshamet/Desktop/dl_project/dl_new_dataset/Screenshot 2026-03-04 at 07.38.39.png)

### Model Comparison Tab
![Model Comparison](/Users/allikhankoshamet/Desktop/dl_project/dl_new_dataset/Screenshot 2026-03-04 at 07.39.10.png)

### Example Recognition
![Recognition Example](/Users/allikhankoshamet/Desktop/dl_project/dl_new_dataset/Screenshot 2026-03-04 at 07.40.35.png)

**Total cost calculation shown in real time.**

## How to Run the Application
