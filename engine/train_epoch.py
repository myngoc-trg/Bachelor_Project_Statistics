import torch
from tqdm import tqdm

def train_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for imgs, sizes, labels in tqdm(loader):
        imgs = imgs.to(device)
        sizes = sizes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, sizes)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy