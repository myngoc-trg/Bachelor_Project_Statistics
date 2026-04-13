import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
import os

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


def _prepare_confusion_for_plot(
    conf,
    idx_to_class,
    normalize=True,
    sort_by_recall_desc=False,
    fixed_order=BASELINE_RECALL_ORDER_NAMES
):
    conf = conf.detach().cpu()

    row_sums = conf.sum(dim=1).clamp(min=1)
    recalls = conf.diag().float() / row_sums.float()

    order = list(range(conf.shape[0]))
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}

    if fixed_order is not None:
        if len(fixed_order) == 0:
            raise ValueError("fixed_order is empty.")

        if isinstance(fixed_order[0], str):
            order = [class_to_idx[name] for name in fixed_order]
        else:
            order = list(fixed_order)

    elif sort_by_recall_desc:
        order = torch.argsort(recalls, descending=True).tolist()

    conf = conf[order][:, order]
    sorted_labels = [idx_to_class[i] for i in order]

    if normalize:
        conf_plot = conf.float() / conf.sum(dim=1, keepdim=True).clamp(min=1)
        conf_plot = conf_plot.numpy()

        annot = np.empty(conf_plot.shape, dtype=object)
        for i in range(conf_plot.shape[0]):
            for j in range(conf_plot.shape[1]):
                annot[i, j] = f"{conf_plot[i, j]:.2f}" if conf_plot[i, j] != 0 else ""
    else:
        conf_int = conf.round().to(torch.int64).numpy()
        conf_plot = conf_int.astype(float)

        annot = np.empty(conf_int.shape, dtype=object)
        for i in range(conf_int.shape[0]):
            for j in range(conf_int.shape[1]):
                annot[i, j] = str(conf_int[i, j]) if conf_int[i, j] != 0 else ""

    return conf_plot, annot, sorted_labels


def _draw_confusion_matrix_on_ax(
    ax,
    conf_plot,
    annot,
    labels,
    title,
    show_ylabel=True,
    show_yticklabels=True,
    bg_color="#E6E6E6",
    annot_size=14,
    tick_size=12,
    label_size=14,
    title_size=16,
    vmax=None
):
    cb = sns.color_palette("colorblind")
    diag_color = cb[0]
    offdiag_color = cb[1]

    diag_cmap = LinearSegmentedColormap.from_list(
        "diag_blue_cb", ["#FFFFFF", diag_color]
    )
    offdiag_cmap = LinearSegmentedColormap.from_list(
        "offdiag_orange_cb", ["#FFFFFF", offdiag_color]
    )

    n = conf_plot.shape[0]
    diag_mask = np.eye(n, dtype=bool)

    diag_data = np.where(diag_mask & (conf_plot != 0), conf_plot, np.nan)
    offdiag_data = np.where((~diag_mask) & (conf_plot != 0), conf_plot, np.nan)

    diag_annot = np.where(~np.isnan(diag_data), annot, "")
    offdiag_annot = np.where(~np.isnan(offdiag_data), annot, "")

    if vmax is None:
        vmax = np.nanmax(conf_plot) if np.nanmax(conf_plot) > 0 else 1

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
        xticklabels=labels,
        yticklabels=labels if show_yticklabels else False,
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
        xticklabels=labels,
        yticklabels=labels if show_yticklabels else False,
        linewidths=0.5,
        linecolor="gray",
        mask=np.isnan(diag_data),
        ax=ax
    )

    ax.set_xlabel("Predicted Class", fontsize=label_size)

    if show_ylabel:
        ax.set_ylabel("True Class", fontsize=label_size)
    else:
        ax.set_ylabel("")

    ax.set_title(title, fontsize=title_size, pad=10)

    ax.tick_params(axis="x", labelrotation=45, labelsize=tick_size)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    if show_yticklabels:
        ax.tick_params(axis="y", labelrotation=0, labelsize=tick_size)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

def plot_confusion_matrix(
    conf,
    idx_to_class,
    normalize=True,
    sort_by_recall_desc=False,
    fixed_order=BASELINE_RECALL_ORDER_NAMES,
    figsize=(12, 8),
    bg_color="#E6E6E6",
    annot_size=14,
    tick_size=12,
    label_size=14,
    title_size=16,
    save_path=None
):
    conf_plot, annot, labels = _prepare_confusion_for_plot(
        conf=conf,
        idx_to_class=idx_to_class,
        normalize=normalize,
        sort_by_recall_desc=sort_by_recall_desc,
        fixed_order=fixed_order
    )

    if normalize:
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix (Counts)"

    if fixed_order is not None:
        title += " (baseline recall order)"
    elif sort_by_recall_desc:
        title += " (sorted by recall descending)"

    vmax = np.nanmax(conf_plot) if np.nanmax(conf_plot) > 0 else 1

    fig, ax = plt.subplots(figsize=figsize)

    _draw_confusion_matrix_on_ax(
        ax=ax,
        conf_plot=conf_plot,
        annot=annot,
        labels=labels,
        title=title,
        show_ylabel=True,
        show_yticklabels=True,
        bg_color=bg_color,
        annot_size=annot_size,
        tick_size=tick_size,
        label_size=label_size,
        title_size=title_size,
        vmax=vmax
    )

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)

    plt.show()



def plot_confusion_matrices_horizontal(
    conf_list,
    model_titles,
    idx_to_class,
    normalize=True,
    fixed_order=BASELINE_RECALL_ORDER_NAMES,
    bg_color="#E6E6E6",
    annot_size=13,
    tick_size=11,
    label_size=13,
    title_size=15,
    panel_width=6.0,
    panel_height=5.8,
    save_path=None
):
    """
    Plot multiple confusion matrices side by side using the same class order.

    Parameters
    ----------
    conf_list : list of torch.Tensor
        List of confusion matrices, one per model.
    model_titles : list of str
        Titles shown above each subplot.
    idx_to_class : dict
        Mapping from class index to class name.
    fixed_order : list[str] or list[int]
        Use the baseline order for all models.
    """

    if len(conf_list) != len(model_titles):
        raise ValueError("conf_list and model_titles must have the same length.")

    prepared = []
    global_vmax = 1

    for conf in conf_list:
        conf_plot, annot, labels = _prepare_confusion_for_plot(
            conf=conf,
            idx_to_class=idx_to_class,
            normalize=normalize,
            sort_by_recall_desc=False,
            fixed_order=fixed_order
        )
        prepared.append((conf_plot, annot, labels))
        current_max = np.nanmax(conf_plot) if np.nanmax(conf_plot) > 0 else 1
        global_vmax = max(global_vmax, current_max)

    n_models = len(conf_list)
    figsize = (panel_width * n_models, panel_height)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        conf_plot, annot, labels = prepared[i]

        _draw_confusion_matrix_on_ax(
            ax=ax,
            conf_plot=conf_plot,
            annot=annot,
            labels=labels,
            title=model_titles[i],
            show_ylabel=(i == 0),
            show_yticklabels=(i == 0),
            bg_color=bg_color,
            annot_size=annot_size,
            tick_size=tick_size,
            label_size=label_size,
            title_size=title_size,
            vmax=global_vmax
        )

    if normalize:
        fig.suptitle("Normalized Confusion Matrices (baseline recall order)", fontsize=title_size + 1, y=0.98)
    else:
        fig.suptitle("Confusion Matrices (Counts) (baseline recall order)", fontsize=title_size + 1, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)

    plt.show()