import matplotlib.pyplot as plt

def plot_training_history(history, epochs_num):
    """
    Plots training & validation loss and accuracy.

    Parameters
    ----------
    history : dict
        Dictionary containing:
        - 'train_loss'
        - 'val_loss'
        - 'train_acc'
        - 'val_acc'
    epochs_num : int
        Number of epochs trained
    """

    epochs = range(1, epochs_num + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Loss ----
    ax[0].plot(epochs, history["train_loss"], label="Train Loss")
    ax[0].plot(epochs, history["val_loss"], label="Val Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training and Validation Loss")
    ax[0].legend()
    ax[0].grid(True)

    # ---- Accuracy ----
    ax[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    ax[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Training and Validation Accuracy")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()