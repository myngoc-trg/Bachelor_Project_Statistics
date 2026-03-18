import torch
from torchmetrics.classification import MulticlassRecall, MulticlassAccuracy, MulticlassConfusionMatrix


def eval_epoch(model, loader, device
               ,num_classes=10, loss_fn=None
               ,inspect_features=False):
    
    recall_metric = MulticlassRecall(
        num_classes=num_classes, average = None).to(device)
    accuracy_metric = MulticlassAccuracy(
        num_classes=num_classes, average = "micro").to(device)  
    
    confmat_metric = MulticlassConfusionMatrix(num_classes=10).to(device)

    model.eval()
    
    recall_metric.reset()
    accuracy_metric.reset()
    if confmat_metric is not None:
        confmat_metric.reset()
    
    all_mins, all_maxs, all_means, all_stds = [], [], [], []

    total_loss = 0
    batches = 0
    
    with torch.no_grad():
        for imgs, sizes, labels in loader:
            imgs = imgs.to(device)
            sizes = sizes.to(device)
            labels = labels.to(device)

            outputs = model(imgs, sizes)
            preds = outputs.argmax(dim=1)
            
            recall_metric.update(preds, labels)
            accuracy_metric.update(preds, labels)
            if confmat_metric is not None:
                confmat_metric.update(preds, labels)
            
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                batches += 1
            
            if inspect_features and hasattr(model, "embed"):
                features = model.embed(imgs)

                all_mins.append(features.min().item())
                all_maxs.append(features.max().item())
                all_means.append(features.mean().item())
                all_stds.append(features.std().item())

    recall_per_class = recall_metric.compute().cpu() # tensor of shape (num_classes,)
    accuracy = accuracy_metric.compute().cpu().item() # scalar
    conf_matrix = confmat_metric.compute().cpu() if confmat_metric is not None else None
    val_loss = total_loss / batches if loss_fn is not None else None
    
    if inspect_features and len(all_mins) > 0:
        print("\n=== Backbone Feature Inspection ===")
        print("Min of mins:", min(all_mins))
        print("Max of maxs:", max(all_maxs))
        print("Mean of batch means:", sum(all_means) / len(all_means))
        print("Mean of batch stds:", sum(all_stds) / len(all_stds))
        print("==================================\n")

    return recall_per_class, accuracy, val_loss, conf_matrix