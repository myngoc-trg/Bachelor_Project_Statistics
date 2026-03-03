from torch.utils.data import DataLoader
import torch
import os
from .train_epoch import train_epoch
from .eval_epoch import eval_epoch

def train_model(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    optimizer,
    loss_fn,
    device,
    recall_metric,
    accuracy_metric,
    model_name: str,
    batch_size: int = 32,
    epochs: int = 35,
    num_workers: int = 0,
    print_every: int = 5,
    early_stopping_patience: int = 8,
    min_delta: float = 1e-4,
):
    """
    Full training pipeline using train_epoch() and eval_epoch().
    """
    
    save_dir = os.path.join("models", "saved_models")
    os.makedirs(save_dir, exist_ok=True)   # create folder if not exists

    save_path = os.path.join(save_dir, model_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_left = early_stopping_patience

    for epoch in range(1, epochs + 1):
        
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            loss_fn
        )
        if val_loader is not None:
            val_recall_per_class, val_acc, val_loss,_ = eval_epoch(
                model,
                val_loader,
                device,
                recall_metric,
                accuracy_metric,
                None,
                loss_fn
            )
        else:
            # No validation set: treat val metrics as None
            val_recall_per_class, val_acc, val_loss = None, None, None

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loader is not None:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            if epoch == 1 or epoch % print_every == 0:
                print(f"Val Recall per class: {val_recall_per_class}")
        else:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

 
        if val_loader is not None:
            improved = (val_acc - best_val_acc) > min_delta
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_left = early_stopping_patience
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_acc": best_val_acc,
                    },
                    save_path
                )
                print(f"Saved new best model (Val Acc = {best_val_acc:.4f})")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"Best Val Acc = {best_val_acc:.4f} at epoch {best_epoch}."
                    )
                    break
    if val_loader is not None and os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Reloaded best model from epoch {ckpt['epoch']} (Val Acc={ckpt['best_val_acc']:.4f})")
    return history, best_val_acc, test_loader