import torch

def choose_specialist_pair(recall_per_class, conf, ignore_classes=None):
    """
    recall_per_class: Tensor [C]
    conf: Tensor [C, C] with counts (true rows, predicted cols)
    ignore_classes: optional set/list of class indices to exclude

    Returns:
      (c_low, c_conf): lowest-recall class and its most confused-with class
    """
    C = conf.shape[0]
    ignore = set(ignore_classes) if ignore_classes is not None else set()

    # 1) pick lowest recall among allowed classes
    recall = recall_per_class.clone()
    for c in ignore:
        recall[c] = 1.0  # exclude by setting high recall

    c_low = int(torch.argmin(recall).item())

    # 2) pick the class it is most often predicted as (row c_low, excluding diagonal)
    row = conf[c_low].clone()
    row[c_low] = 0
    c_conf = int(torch.argmax(row).item())

    return c_low, c_conf