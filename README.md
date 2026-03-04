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
<img width="177" height="115" alt="Screenshot 2026-03-04 at 07 48 37" src="https://github.com/user-attachments/assets/eb70c720-3a82-462e-a511-cfb2102c52f6" />

<img width="668" height="513" alt="Screenshot 2026-03-04 at 07 49 00" src="https://github.com/user-attachments/assets/f14f1d75-38ea-446e-9808-b0131fe0be64" />


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
<img width="1560" height="708" alt="Screenshot 2026-03-04 at 07 38 39" src="https://github.com/user-attachments/assets/3a6cae91-ef78-45af-80aa-2f0c46031eea" />

### Model Comparison Tab
<img width="1518" height="642" alt="Screenshot 2026-03-04 at 07 39 10" src="https://github.com/user-attachments/assets/c81f6444-f6a9-4a6c-b225-dbbd06ad021c" />


### Example Recognition
<img width="1187" height="394" alt="Screenshot 2026-03-04 at 07 40 35" src="https://github.com/user-attachments/assets/930035bd-689d-4dda-b081-65a086eeb5f8" />

