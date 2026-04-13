import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

BASELINE_RECALL_ORDER_NAMES = [
    "Brassica napus",
    "Tussilago farfara",
    "Cichorium intybus",
    "Capsella bursa-pastoris",
    "Tragopogon pratensis",
    "Hieracium umbellatum",
    "Sonchus arvensis",
    "Hypochaeris radicata",
    "Bellis perennis",
    "Crepis capillaris",
]

def plot_confusion_matrix(
    conf,
    idx_to_class,
    normalize=True,
    sort_by_recall_desc=False,
    figsize=(10, 8),
    bg_color="#E6E6E6",
    annot_size=16,      # numbers inside cells
    tick_size=14,       # class names on axes
    label_size=16,      # axis labels
    title_size=18       # title
    ,fixed_order =None
):
    conf = conf.cpu()

    row_sums = conf.sum(dim=1).clamp(min=1)
    recalls = conf.diag().float() / row_sums.float()

    order = list(range(conf.shape[0]))
    if sort_by_recall_desc:
        order = torch.argsort(recalls, descending=True).tolist()
        conf = conf[order][:, order]

    sorted_labels = [idx_to_class[i] for i in order]

    cb = sns.color_palette("colorblind")
    diag_color = cb[0]
    offdiag_color = cb[1]

    diag_cmap = LinearSegmentedColormap.from_list(
        "diag_blue_cb", ["#FFFFFF", diag_color]
    )
    offdiag_cmap = LinearSegmentedColormap.from_list(
        "offdiag_orange_cb", ["#FFFFFF", offdiag_color]
    )

    if normalize:
        conf_plot = conf.float() / conf.sum(dim=1, keepdim=True).clamp(min=1)
        conf_plot = conf_plot.numpy()

        annot = np.empty(conf_plot.shape, dtype=object)
        for i in range(conf_plot.shape[0]):
            for j in range(conf_plot.shape[1]):
                annot[i, j] = f"{conf_plot[i, j]:.2f}" if conf_plot[i, j] != 0 else ""

        title = "Normalized Confusion Matrix"
    else:
        conf_int = conf.round().to(torch.int64).numpy()
        conf_plot = conf_int.astype(float)

        annot = np.empty(conf_int.shape, dtype=object)
        for i in range(conf_int.shape[0]):
            for j in range(conf_int.shape[1]):
                annot[i, j] = str(conf_int[i, j]) if conf_int[i, j] != 0 else ""

        title = "Confusion Matrix (Counts)"

    if sort_by_recall_desc:
        title += " (sorted by recall descending)"

    n = conf_plot.shape[0]
    diag_mask = np.eye(n, dtype=bool)

    diag_data = np.where(diag_mask & (conf_plot != 0), conf_plot, np.nan)
    offdiag_data = np.where((~diag_mask) & (conf_plot != 0), conf_plot, np.nan)

    diag_annot = np.where(~np.isnan(diag_data), annot, "")
    offdiag_annot = np.where(~np.isnan(offdiag_data), annot, "")

    vmax = np.nanmax(conf_plot) if np.nanmax(conf_plot) > 0 else 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(bg_color)

    sns.heatmap(
        offdiag_data,
        annot=offdiag_annot,
        fmt="",
        annot_kws={"size": annot_size},
        cmap=offdiag_cmap,
        vmin=0,
        vmax=vmax,
        cbar=False,
        xticklabels=sorted_labels,
        yticklabels=sorted_labels,
        linewidths=0.5,
        linecolor="gray",
        mask=np.isnan(offdiag_data),
        ax=ax
    )

    sns.heatmap(
        diag_data,
        annot=diag_annot,
        fmt="",
        annot_kws={"size": annot_size},
        cmap=diag_cmap,
        vmin=0,
        vmax=vmax,
        cbar=False,
        xticklabels=sorted_labels,
        yticklabels=sorted_labels,
        linewidths=0.5,
        linecolor="gray",
        mask=np.isnan(diag_data),
        ax=ax
    )

    plt.xlabel("Predicted Class", fontsize=label_size)
    plt.ylabel("True Class", fontsize=label_size)
    plt.title(title, fontsize=title_size)

    plt.xticks(rotation=35, ha="right", fontsize=tick_size)
    plt.yticks(rotation=0, fontsize=tick_size)

    plt.tight_layout()
    plt.show()