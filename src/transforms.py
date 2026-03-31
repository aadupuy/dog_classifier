from torchvision import transforms

# Train traoforms with data augmentation, resize, and normalization
def get_train_transforms():
    train_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224), # better than just resizing for data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform

# Test transforms with just resize and normalization
def get_val_transforms():
    val_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224), # don't want data augmentation for validation/test, just resize and normalize
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return val_transform