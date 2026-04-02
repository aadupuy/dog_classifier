import torch
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.predict import load_trained_model, predict_image
from src.transforms import get_val_transforms

LABELS_PATH = "data/raw/labels.csv"
TRAIN_DIR = "data/raw/train"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = pd.read_csv(LABELS_PATH)

breeds = sorted(df["breed"].unique())
breed_to_idx = {breed: idx for idx, breed in enumerate(breeds)}
idx_to_breed = {idx: breed for idx, breed in enumerate(breeds)}

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["breed"]
)

# Pick N random samples from the validation set
N = 5
val_df_sample = val_df.sample(n=N, random_state=42)

model = load_trained_model("models/best_model.pth", num_classes=len(breed_to_idx), device=DEVICE)
val_transform = get_val_transforms()

fig, ax = plt.subplots(1, N, figsize=(15, 5))

for i, (_, row) in enumerate(val_df_sample.iterrows()):
    image_path = os.path.join(TRAIN_DIR, row["id"] + ".jpg")
    true_breed = row["breed"]

    preds = predict_image(
        image=image_path,
        model=model,
        transform=val_transform,
        idx_to_breed=idx_to_breed,
        device=DEVICE,
        top_k=3
    )

    img = Image.open(image_path).convert("RGB")
    ax[i].imshow(img)

    true_display = true_breed.replace("_", " ")
    top3_text = "\n".join([
        f"{j+1}. {breed} ({prob:.2f})"
        for j, (breed, prob) in enumerate(preds)
    ])

    ax[i].set_title(
        f"True: {true_display}\n{top3_text}",
        fontsize=10
    )
    ax[i].axis("off")

os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/sample_predictions.png", dpi=300, bbox_inches="tight")
plt.show()