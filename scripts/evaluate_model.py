import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import DogDataset
from src.transforms import get_val_transforms
from src.predict import load_trained_model
from src.evaluate import evaluate_topk

# -------------------
# Config
# -------------------
DATA_DIR = "data/raw"
TRAIN_DIR = "data/raw/train"
LABELS_PATH = "data/raw/labels.csv"
MODEL_PATH = "models/best_model.pth"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# -------------------
# Load data
# -------------------
df = pd.read_csv(LABELS_PATH)

# Create label mapping
breeds = sorted(df["breed"].unique())
breed_to_idx = {breed: idx for idx, breed in enumerate(breeds)}

# Train / validation split (stratify to maintain class distribution in train and val sets)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['breed'])

# -------------------
# Transforms + Dataset + DataLoader
# -------------------
val_transform = get_val_transforms()
val_dataset = DogDataset(val_df, TRAIN_DIR, breed_to_idx, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------
# Model
# -------------------
model = load_trained_model(MODEL_PATH, num_classes=len(breed_to_idx), device=DEVICE)

# -------------------
# Evaluate
# -------------------
acc = evaluate_topk(model, val_loader, DEVICE)

print("\n=== Validation Results ===")
for k, v in acc.items():
    print(f"Top-{k} accuracy: {v:.4f}")