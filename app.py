import streamlit as st
from PIL import Image
import torch
import pandas as pd

from src.predict import load_trained_model, predict_image
from src.transforms import get_val_transforms

# -------------------
# Config
# -------------------
MODEL_PATH = "models/best_model.pth"
LABELS_PATH = "data/raw/labels.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load model (cache)
# -------------------
@st.cache_resource
def load_model():
    df = pd.read_csv(LABELS_PATH)
    breeds = sorted(df["breed"].unique())
    idx_to_breed = {idx: breed for idx, breed in enumerate(breeds)}
    model = load_trained_model(MODEL_PATH, num_classes=len(breeds), device=DEVICE)
    return model, idx_to_breed

model, idx_to_breed = load_model()

# -------------------
# UI
# -------------------
st.title("🐶 What dog is it?! 🐶")
st.caption("Upload an image and let a deep learning model guess the breed.")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", width='stretch')

    # Predict
    preds = predict_image(image, model, get_val_transforms(), idx_to_breed, DEVICE, top_k=3)

    st.markdown("---")
    st.subheader("Top Predictions")

    for breed, prob in preds:
        st.write(f"**{breed}** — {prob*100:.1f}%")
        st.progress(float(prob))