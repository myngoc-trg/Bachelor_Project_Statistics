from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from Data_Utility.binary_subset import BinaryPairDataset
from models.specialist_binary import BinarySpecialist
from engine.train_epoch import train_epoch
from engine.eval_epoch import eval_epoch
from torchmetrics.classification import MulticlassRecall, MulticlassAccuracy

def train_binary_specialist(
    stage1_model,
    train_dataset,
    val_dataset,
    device,
    class_a,
    class_b,
    lr=1e-3,
    weight_decay=1e-4,
    min_delta=1e-4,
    epochs=8,
    batch_size=32,
    num_workers=0,
    print_every=5,
    early_stopping_patience=5,
):
    train_bin = BinaryPairDataset(train_dataset, class_a, class_b)
    val_bin = BinaryPairDataset(val_dataset, class_a, class_b) if val_dataset is not None else None

    train_loader = DataLoader(train_bin, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_bin, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_bin is not None else None

    specialist = BinarySpecialist(stage1_model=stage1_model, freeze_stage1=True).to(device)
    
    # Binary metrics 
    bin_recall = MulticlassRecall(num_classes=2, average=None).to(device)
    bin_acc    = MulticlassAccuracy(num_classes=2).to(device)

    optimizer = torch.optim.AdamW(specialist.head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()


    best_val_acc = -float("inf")
    best_epoch = 0
    best_state = None
    patience_left = early_stopping_patience

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(specialist, train_loader, device, optimizer, loss_fn)

        if val_loader is not None:
         
            val_recall_per_class, val_acc, val_loss, _ = eval_epoch(
                model=specialist,
                loader=val_loader,
                device=device,
                recall_metric=bin_recall,
                accuracy_metric=bin_acc,
                loss_fn=loss_fn,
                confmat_metric=None
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            if epoch == 1 or epoch % print_every == 0:
                print(f"Val Recall per class: {val_recall_per_class}")

    
            improved = (val_acc - best_val_acc) > min_delta
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in specialist.state_dict().items()}
                patience_left = early_stopping_patience
                print(f"New best specialist (Val Acc={best_val_acc:.4f}) at epoch {best_epoch}")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch}. Best Val Acc={best_val_acc:.4f} (epoch {best_epoch}).")
                    break

        else:
            # no val set -> just train through epochs (can't select best)
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

    # --- load best before returning ---
    if val_loader is not None and best_state is not None:
        specialist.load_state_dict(best_state)
        specialist.to(device)
        print(f"Reloaded best specialist from epoch {best_epoch} (Val Acc={best_val_acc:.4f})")

    return specialist, best_val_acc, best_epoch