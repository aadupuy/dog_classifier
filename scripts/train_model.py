import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data import DogDataset
from src.model import get_model
from src.train import train_model
from src.transforms import get_train_transforms, get_val_transforms

# -------------------
# Config
# -------------------
DATA_DIR = "data/raw"
TRAIN_DIR = "data/raw/train"
LABELS_PATH = "data/raw/labels.csv"
BATCH_SIZE = 32
NUM_EPOCHS = 10
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
# Transforms
# -------------------
train_transform = get_train_transforms()
val_transform = get_val_transforms()

# -------------------
# Datasets
# -------------------
train_dataset = DogDataset(train_df, TRAIN_DIR, breed_to_idx, transform=train_transform)
val_dataset = DogDataset(val_df, TRAIN_DIR, breed_to_idx, transform=val_transform)

# -------------------
# DataLoaders
# -------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------
# Model
# -------------------
model = get_model(num_classes=len(breed_to_idx))
model.to(DEVICE)

# -------------------
# Training setup
# -------------------
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # because we monitor accuracy
    factor=0.3,        # reduce LR by 70%
    patience=2,        # wait 2 epochs
)

# Check that only the last layers are trainable
# print(model.layer3[0].conv1.weight.requires_grad)  # should be False
# print(model.layer4[0].conv1.weight.requires_grad)  # should be True
# print(model.fc.weight.requires_grad)               # should be True

# -------------------
# Train
# -------------------
train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, num_epochs=NUM_EPOCHS, scheduler=scheduler)