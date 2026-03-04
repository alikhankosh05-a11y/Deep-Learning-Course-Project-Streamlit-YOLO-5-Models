import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import pandas as pd

st.set_page_config(page_title="CNN Product Recognition", layout="wide")

# ====================== DATA ======================
class_names = ['battery', 'copybook', 'espnder', 'headphones', 'highlighter', 'marker', 
               'mouse', 'napcin', 'notebook', 'paper holder', 'pen', 'pencile', 
               'stepler', 'sticker', 'usb']

prices = {
    'battery': 250, 'copybook': 420, 'espnder': 180, 'headphones': 3200,
    'highlighter': 160, 'marker': 220, 'mouse': 1450, 'napcin': 90,
    'notebook': 950, 'paper holder': 580, 'pen': 140, 'pencile': 110,
    'stepler': 720, 'sticker': 280, 'usb': 850
}

# Updated with your NEW TEST results
model_info = {
    "alexnet": {
        "name": "AlexNet",
        "acc": "90.00%",
        "desc": "5 convolutional layers + 3 fully connected layers. Baseline model for understanding CNN fundamentals."
    },
    "vgg16": {
        "name": "VGG16",
        "acc": "95.00%",
        "desc": "Deep architecture with small 3x3 filters. Very accurate, but computationally intensive."
    },
    "googlenet": {
        "name": "GoogLeNet (Inception)",
        "acc": "86.67%",
        "desc": "Inception modules + auxiliary classifiers. Efficient parameter usage."
    },
    "resnet50": {
        "name": "ResNet50",
        "acc": "100%",
        "desc": "Skip connections (residual blocks). Solves the vanishing gradient problem - one of the best models."
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "acc": "86.67%",
        "desc": "Best balance between accuracy, size, and speed."
    }
}

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_cnn_model(model_name: str):
    try:
        if model_name == "alexnet":
            model = models.alexnet(weights=None)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))
        elif model_name == "vgg16":
            model = models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))
        elif model_name == "googlenet":
            model = models.googlenet(weights=None, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
        
        model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

@st.cache_resource
def get_yolo_model():
    return YOLO("yolov8n.pt")

# ====================== CLASSIFY CROP ======================
def predict_crop(model, crop_img):
    tensor = transform(crop_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred_idx = torch.max(prob, 0)
    return class_names[pred_idx], confidence.item()

# ====================== INTERFACE ======================
st.title("CNN-Based Product Recognition & Price Estimation")
st.caption("Deep Learning Course Project - 5 Architectures - Object Detection + Price Sum")

tabs = st.tabs(["AlexNet", "VGG16", "GoogLeNet", "ResNet50", "EfficientNet-B0", "Model Comparison"])
model_keys = ["alexnet", "vgg16", "googlenet", "resnet50", "efficientnet_b0"]

for tab, key in zip(tabs[:5], model_keys):
    with tab:
        info = model_info[key]
        st.subheader(info['name'])
        st.write(f"**Test Accuracy:** {info['acc']}")
        st.write(info["desc"])
        
        uploaded = st.file_uploader("Upload a photo with multiple products", 
                                   type=["jpg", "jpeg", "png"], key=key)
        
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Photo", use_column_width=True)
            
            if st.button("Run Recognition", key=f"btn_{key}"):
                with st.spinner("YOLO detecting objects + CNN classifying..."):
                    cnn_model = load_cnn_model(key)
                    if cnn_model is None:
                        continue
                    
                    yolo = get_yolo_model()
                    results = yolo(image, conf=0.25, verbose=False)
                    
                    detected = []
                    total_cost = 0
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = image.crop((x1, y1, x2, y2))
                        
                        product, conf = predict_crop(cnn_model, crop)
                        price = prices.get(product, 0)
                        
                        detected.append({
                            "Product": product.capitalize(),
                            "Confidence": f"{conf:.1%}",
                            "Price": f"{price} ₸"
                        })
                        total_cost += price
                    
                    if detected:
                        st.success("Recognition completed!")
                        df = pd.DataFrame(detected)
                        st.dataframe(df, use_container_width=True)
                        st.markdown(f"### Total Cost: {total_cost:,} ₸")
                    else:
                        st.warning("YOLO didn't detect any objects. Try another photo or lower the confidence threshold.")

# ====================== MODEL COMPARISON TAB ======================
with tabs[5]:
    st.subheader("Model Comparison (Test Results)")
    st.write("All metrics calculated on the test set + model file sizes")
    
    metrics_df = pd.DataFrame({
        'Model': ['AlexNet', 'VGG16', 'GoogLeNet', 'ResNet50', 'EfficientNet-B0'],
        'Test Accuracy': [0.9000, 0.9500, 0.8667, 1.0000, 0.8667],
        'Precision': [0.9117, 0.9688, 0.8649, 1.0000, 0.8558],
        'Recall': [0.9000, 0.9500, 0.8667, 1.0000, 0.8667],
        'F1-score': [0.9001, 0.9509, 0.8497, 1.0000, 0.8458]
    })
    
    size_df = pd.DataFrame({
        'Model': ['AlexNet', 'VGG16', 'GoogLeNet', 'ResNet50', 'EfficientNet-B0'],
        'Model Size (MB)': [217.69, 512.41, 21.59, 90.09, 15.64]
    })
    
    st.dataframe(metrics_df.style.format({
        'Test Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-score': '{:.4f}'
    }).highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    st.dataframe(size_df, use_container_width=True)
    
    st.success("ResNet50 shows the best overall performance!")

st.caption("Developed for Deep Learning Course Project - Streamlit + YOLO + Your 5 Models")