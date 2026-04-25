import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

def extract_features_labels_flowerids(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []
    all_filenames = []
    all_flower_ids = []

    with torch.no_grad():
        for x_img, x_size, y, filenames, flower_ids in loader:
            x_img = x_img.to(device)
            x_size = x_size.to(device).float()
            y = y.to(device)

            feats = model.forward_features(x_img, x_size)

            all_features.append(feats.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_filenames.extend(list(filenames))
            all_flower_ids.extend(list(flower_ids))

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y, np.array(all_filenames), np.array(all_flower_ids)


def plot_pca_tsne_flowerid_two_panels(
    X_train,
    y_train,
    flower_id_train,
    X_test,
    y_test,
    flower_id_test,
    class_names,
    confusing_classes,
    method="pca",
    figsize=(6.8, 3.4),   
    point_size=14,
    alpha=0.80,
    title_fontsize=11,
    label_fontsize=10,
    tick_fontsize=9,
    legend_fontsize=8,
    random_state=42,
    show_flower_legend=False,
    max_flower_ids_in_legend=12,
    species_legend_h=0.90,
    suptitile_h=0.90,
    rect_h=0.85,
    main_title=None,
    same_axes=True,
    sharex=True,
    save_path=None,
    dpi=300
):
    """
    Two-panel PCA/t-SNE plot:
      - left panel: Train
      - right panel: Test
      - species encoded by marker shape
      - flower_id encoded by color

    Parameters
    ----------
    flower_id_train, flower_id_test : array-like of shape (n_samples,)
        Flower IDs aligned with X_train/y_train and X_test/y_test.
    show_flower_legend : bool
        Whether to show a flower-ID legend.
    max_flower_ids_in_legend : int
        Only show flower legend if number of unique flower IDs is small enough.
    """

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    flower_id_train = np.asarray(flower_id_train)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    flower_id_test = np.asarray(flower_id_test)

    confusing_idx = [class_names.index(c) for c in confusing_classes]

    train_mask = np.isin(y_train, confusing_idx)
    test_mask = np.isin(y_test, confusing_idx)

    X_train_sub = X_train[train_mask]
    y_train_sub = y_train[train_mask]
    flower_train_sub = flower_id_train[train_mask]

    X_test_sub = X_test[test_mask]
    y_test_sub = y_test[test_mask]
    flower_test_sub = flower_id_test[test_mask]

    if len(X_train_sub) == 0 or len(X_test_sub) == 0:
        raise ValueError("Filtered train or test set is empty.")

    # Same embedding space
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        X_train_2d = reducer.fit_transform(X_train_sub)
        X_test_2d = reducer.transform(X_test_sub)
        xlabel, ylabel = "PC1", "PC2"

    elif method.lower() == "tsne":
        X_all = np.vstack([X_train_sub, X_test_sub])
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=random_state
        )
        X_all_2d = reducer.fit_transform(X_all)
        X_train_2d = X_all_2d[:len(X_train_sub)]
        X_test_2d = X_all_2d[len(X_train_sub):]
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"

    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=sharex, sharey=False)

    # Same species = same marker shape
    available_markers = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "*"]
    if len(confusing_idx) > len(available_markers):
        raise ValueError("Too many confusing classes for available marker styles.")

    marker_map = {
        cls_idx: available_markers[i]
        for i, cls_idx in enumerate(confusing_idx)
    }

    # Same flower_id = same color across train and test
    all_flower_ids = np.unique(np.concatenate([flower_train_sub, flower_test_sub]))
    cmap = plt.get_cmap("tab20", len(all_flower_ids))
    flower_color_map = {fid: cmap(i) for i, fid in enumerate(all_flower_ids)}

    # --- Train panel ---
    for cls_idx in confusing_idx:
        cls_mask = (y_train_sub == cls_idx)
        for fid in np.unique(flower_train_sub[cls_mask]):
            mask = cls_mask & (flower_train_sub == fid)
            axes[0].scatter(
                X_train_2d[mask, 0],
                X_train_2d[mask, 1],
                s=point_size,
                alpha=alpha,
                marker=marker_map[cls_idx],
                color=flower_color_map[fid],
                edgecolors="black",
                linewidths=0.35
            )

    # --- Test panel ---
    for cls_idx in confusing_idx:
        cls_mask = (y_test_sub == cls_idx)
        for fid in np.unique(flower_test_sub[cls_mask]):
            mask = cls_mask & (flower_test_sub == fid)
            axes[1].scatter(
                X_test_2d[mask, 0],
                X_test_2d[mask, 1],
                s=point_size,
                alpha=alpha,
                marker=marker_map[cls_idx],
                color=flower_color_map[fid],
                edgecolors="black",
                linewidths=0.35
            )

    # Same scale across panels
    if same_axes:
        x_min = min(X_train_2d[:, 0].min(), X_test_2d[:, 0].min())
        x_max = max(X_train_2d[:, 0].max(), X_test_2d[:, 0].max())
        y_min = min(X_train_2d[:, 1].min(), X_test_2d[:, 1].min())
        y_max = max(X_train_2d[:, 1].max(), X_test_2d[:, 1].max())

        axes[0].set_xlim(x_min, x_max)
        axes[1].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)

    axes[0].set_title("Train", fontsize=title_fontsize - 2)
    axes[1].set_title("Test", fontsize=title_fontsize - 2)

    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        #remove box, keep axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    # Species legend: shape only, shown inside left panel at upper-left corner
    # Species legend: shape only
    
    # Hide y-axis on the right subplot
    axes[1].set_ylabel("")                 # remove label
    axes[1].tick_params(axis='y', left=False, labelleft=False)  # remove ticks + numbers
    
    species_handles = [
        Line2D(
            [0], [0],
            marker=marker_map[cls_idx],
            linestyle="None",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label=class_names[cls_idx]
        )
        for cls_idx in confusing_idx
    ]

    fig.legend(
        handles=species_handles,
        #title="Species",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc="upper left",
        bbox_to_anchor=(0, species_legend_h)
    )

    # Optional flower legend
    if show_flower_legend and len(all_flower_ids) <= max_flower_ids_in_legend:
        flower_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor=flower_color_map[fid],
                markeredgecolor="black",
                markersize=7,
                label=str(fid)
            )
            for fid in all_flower_ids
        ]

        fig.legend(
            handles=flower_handles,
            title="Flower ID",
            fontsize=max(8, legend_fontsize - 1),
            title_fontsize=max(8, legend_fontsize - 1),
            loc="upper right",
            #bbox_to_anchor=(0.86, 0.10)
            #frameon=True
        )

    if main_title is None:
        main_title = method.upper()

    fig.suptitle(main_title, fontsize=title_fontsize, y=suptitile_h)
    plt.tight_layout(rect=[0, 0, 1, rect_h]) 
    
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return {
        "flower_color_map": flower_color_map,
        "marker_map": marker_map
    }