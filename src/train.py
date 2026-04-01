import torch
import os
os.makedirs("models", exist_ok=True)

# One epoch of training
def epoch_train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0) # accumulate loss weighted by batch size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# One epoch of validation
def epoch_val(model, val_loader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_running_loss / val_total
    val_epoch_acc = val_correct / val_total

    return val_epoch_loss, val_epoch_acc

# Multiple epochs of training and validation
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, scheduler=None):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        epoch_loss, epoch_acc = epoch_train(model, train_loader, criterion, optimizer, device)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        val_epoch_loss, val_epoch_acc = epoch_val(model, val_loader, criterion, device)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            # torch.save(model.state_dict(), "models/best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, "models/best_model.pth")
            print("Saved best model at epoch {} with val acc: {:.4f}".format(epoch, best_val_acc))

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {val_epoch_loss:.4f}, Val   Acc: {val_epoch_acc:.4f}")
        print("-" * 40)

        scheduler.step(val_epoch_acc)

    return train_losses, train_accuracies, val_losses, val_accuracies