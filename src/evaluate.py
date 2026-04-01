import torch

def evaluate_topk(model, dataloader, device, k_list=[1, 3, 5]):
    model.eval()

    correct = {k: 0 for k in k_list}
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # (batch_size, num_classes)

            # Get top-k predictions (max k needed)
            max_k = max(k_list)
            _, topk_preds = torch.topk(outputs, max_k, dim=1)  
            # shape: (batch_size, max_k)

            for k in k_list:
                # check if true label is in top-k
                correct_k = (topk_preds[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
                correct[k] += correct_k

            total += labels.size(0)

    acc = {k: correct[k] / total for k in k_list}
    return acc