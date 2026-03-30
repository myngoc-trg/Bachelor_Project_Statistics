import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_confusion_matrix(
    conf,
    idx_to_class,
    normalize=True,
    sort_by_recall_desc=True,
    figsize=(10, 8),
    cmap="coolwarm"
):
    conf = conf.cpu()

    row_sums = conf.sum(dim=1).clamp(min=1)
    recalls = conf.diag().float() / row_sums.float()

    order = list(range(conf.shape[0]))

    if sort_by_recall_desc:
        order = torch.argsort(recalls, descending=True).tolist()
        conf = conf[order][:, order]

    # reorder labels to match reordered matrix
    sorted_labels = [idx_to_class[i] for i in order]

    if normalize:
        conf = conf.float()
        conf = conf / conf.sum(dim=1, keepdim=True).clamp(min=1)
        fmt = ".2f"
        vmax = 1
        title = "Normalized Confusion Matrix"
    else:
        conf = conf.round().to(torch.int64)
        fmt = "d"
        vmax = None
        title = "Confusion Matrix (Counts)"

    if sort_by_recall_desc:
        title += " (sorted by recall descending)"

    plt.figure(figsize=figsize)

    sns.heatmap(
        conf.numpy(),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        xticklabels=sorted_labels,
        yticklabels=sorted_labels
    )

    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()