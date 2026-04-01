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

# Prediction from a single image
def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # add batch dimension and move to device

    with torch.no_grad():
        outputs = model(image)

        # Top 1 prediction
        # predicted_idx = outputs.argmax(dim=1) # cleaner way to get the index of the max log-probability

        # Top 3 predictions with probabilities
        probs = torch.softmax(outputs, dim=1)  # convert to probabilities
        top3_probs, top3_idxs = torch.topk(probs, 3)

    # return predicted_idx.item()
    return top3_probs.tolist(), top3_idxs.tolist()