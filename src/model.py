from torchvision import models
import torch.nn as nn

def get_model(num_classes):
    # Load pre-trained ResNet-18 model
    # model = models.resnet18(pretrained=True)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
    # updated for latest torchvision versions, old pretrained=True is deprecated

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze all layers except the last few (layer4 and fc) for fine-tuning
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model