import torch

def eval_epoch(model, loader, device, recall_metric, accuracy_metric, confmat_metric=None, loss_fn=None):
    model.eval()
    
    recall_metric.reset()
    accuracy_metric.reset()
    if confmat_metric is not None:
        confmat_metric.reset()
    
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
            
    recall_per_class = recall_metric.compute().cpu() # tensor of shape (num_classes,)
    accuracy = accuracy_metric.compute().cpu().item() # scalar
    conf_matrix = confmat_metric.compute().cpu() if confmat_metric is not None else None
    val_loss = total_loss / batches if loss_fn is not None else None
    
    return recall_per_class, accuracy, val_loss, conf_matrix