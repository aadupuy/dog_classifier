import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

from src.data import DogDataset
from src.transforms import get_val_transforms
from src.predict import load_trained_model
from src.evaluate import evaluate_topk, collect_predictions

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
# Collect predictions for confusion analysis
# -------------------
all_true, all_pred = collect_predictions(model, val_loader, DEVICE)

# -------------------
# Confusion analysis
# -------------------
wrong_pairs = [
    (true_idx, pred_idx)
    for true_idx, pred_idx in zip(all_true, all_pred)
    if true_idx != pred_idx
]

pair_counts = Counter(wrong_pairs)
most_common = pair_counts.most_common(10)

# Convert indices back to breed names
idx_to_breed = {idx: breed for breed, idx in breed_to_idx.items()}
rows = []
for (true_idx, pred_idx), count in most_common:
    rows.append({
        "true_breed": idx_to_breed[true_idx],
        "predicted_breed": idx_to_breed[pred_idx],
        "count": count
    })

confusion_df = pd.DataFrame(rows)
confusion_df = confusion_df.sort_values(by="count", ascending=False)
confusion_df["true_breed"] = confusion_df["true_breed"].str.replace("_", " ")
confusion_df["predicted_breed"] = confusion_df["predicted_breed"].str.replace("_", " ")
confusion_df.to_csv("outputs/metrics/confusion_pairs.csv", index=False)

print("\n=== Most Confused Breed Pairs ===\n")
print(confusion_df.to_string(index=False))

# Visualize the most confused pairs (mini-confusion matrix)
labels = [
    f"{row['true_breed']} → {row['predicted_breed']}"
    for _, row in confusion_df.iterrows()
]

plt.figure(figsize=(6, 4))
plt.barh(labels[::-1], confusion_df["count"][::-1])
plt.xlabel("Count")
ax = plt.gca()
ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))
plt.title("Top Confused Breed Pairs")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_pairs.png", dpi=300, bbox_inches="tight")
plt.show()