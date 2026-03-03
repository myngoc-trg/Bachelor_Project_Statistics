import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_confusion_matrix(conf,
                          idx_to_class,
                          normalize=True,
                          figsize=(10, 8),
                          cmap="coolwarm"):

    conf = conf.cpu()

    if normalize:
        conf = conf.float()
        conf = conf / conf.sum(dim=1, keepdim=True).clamp(min=1)
        fmt = ".2f"
        vmax = 1
        title = "Normalized Confusion Matrix"
    else:
        conf = conf.round().to(torch.int64)   # 🔥 force integer type
        fmt = "d"
        vmax = None
        title = "Confusion Matrix (Counts)"

    plt.figure(figsize=figsize)

    sns.heatmap(
        conf.numpy(),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        xticklabels=[idx_to_class[i] for i in range(len(idx_to_class))],
        yticklabels=[idx_to_class[i] for i in range(len(idx_to_class))]
    )

    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()