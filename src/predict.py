import torch
from PIL import Image
from src.model import get_model

# Load a trained model for inference
def load_trained_model(model_path, num_classes, device):
    model = get_model(num_classes=num_classes)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return model

def predict_image(image, model, transform, idx_to_breed, device, top_k=3):
    model.eval()

    # If image is a path, load it (keeps flexibility)
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Preprocess
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        top_probs, top_idxs = torch.topk(probs, top_k, dim=1)

    # Remove batch dimension
    top_probs = top_probs[0].tolist()
    top_idxs = top_idxs[0].tolist()

    # Convert to readable format
    preds = [
        (idx_to_breed[idx].replace("_", " "), prob)
        for idx, prob in zip(top_idxs, top_probs)
    ]

    return preds