from torchvision import models
import torch.nn as nn

def get_model(num_classes):
    # Load pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model