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
  <img width="390" height="238" alt="Screenshot 2026-03-04 at 07 59 32" src="https://github.com/user-attachments/assets/fb582b17-bb7f-41c4-b7cb-24eb6df9a95b" />

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

## Stages of the project

1. Data Parsing  
At this stage, a custom dataset of 253 images across 15 product categories (stationery and electronics) was collected and organized. Images were manually gathered and sorted into separate folders for each class.

2. Raw Data Processing  
The dataset was split into training (214 images) and validation (39 images) sets. Folder structure was prepared according to PyTorch ImageFolder requirements. Class names were standardized for consistent labeling.

3. Handling Missing Values and Feature Engineering  
No missing values were present in the image dataset. Data augmentation techniques were applied to the training set to increase diversity and prevent overfitting:  
- RandomResizedCrop(224)  
- RandomHorizontalFlip()  
Validation set used only deterministic transformations (Resize + CenterCrop).

4. Exploratory analysis  
Class distribution was analyzed (balanced dataset with 15 classes). Sample images from each category were reviewed. Data loaders and batch sizes were tested for optimal training performance.

5. Encoding and data preparation for training  
Images were transformed into tensors and normalized using ImageNet statistics. PyTorch DataLoaders were created with a batch size of 8. All categorical labels were automatically encoded using `datasets.ImageFolder`.

6. Feature and Model Selection  
Five different CNN architectures were trained using transfer learning with pre-trained ImageNet weights:  
- AlexNet  
- VGG16  
- GoogLeNet (Inception)  
- ResNet50  
- EfficientNet-B0  

Hyperparameters were tuned (learning rate 0.001, SGD optimizer with momentum, StepLR scheduler, 8 epochs). Models were evaluated on the test set using accuracy, precision, recall, and F1-score. The best-performing model (ResNet50) was selected.

7. Model deployment  
- All five trained models were saved as `.pth` files.  
- A user-friendly web application was developed using Streamlit.  
- YOLOv8n was integrated for multi-product object detection in a single image.  
- A price database (JSON mapping) was implemented to calculate individual and total costs in real time.  
- The application supports image upload, product recognition, and automatic price summation.

## Conclusion
The best metrics were demonstrated by **ResNet50**, which achieved **100% test accuracy**, 1.0000 precision, 1.0000 recall, and 1.0000 F1-score.  

Interestingly, the default hyperparameters performed excellently, and ResNet50 outperformed other models despite moderate model size (90.09 MB).  

The baseline models showed 85–95% accuracy, whereas the selected ResNet50 model reached 100% accuracy — an improvement that makes the system highly reliable for real-world use.  

For sure, next time I will use a stronger dataset, since according to my data, I can say that there is a small amount of luck cause few photos match exactly, some do not

The final Streamlit web application successfully performs multi-product recognition and price estimation, fully meeting all project requirements.
