import torch

def two_stage_predict(
    stage1_model,
    specialist,
    pair,                 # (class_a, class_b) in ORIGINAL labels
    imgs,
    sizes,
    device,
    gate="hard",          # "hard" or "margin"
    margin_thresh=0.15
):
    """
    Returns refined predictions.
    gate:
      - "hard": run specialist if stage1 predicts class_a or class_b
      - "margin": run specialist if (p_a + p_b) is high and margin small
    """
    class_a, class_b = pair
    stage1_model.eval()
    specialist.eval()

    with torch.no_grad():
        logits = stage1_model(imgs, sizes)
        preds = logits.argmax(1)

        if gate == "hard":
            mask = (preds == class_a) | (preds == class_b)

        else:
            probs = torch.softmax(logits, dim=1)
            pa = probs[:, class_a]
            pb = probs[:, class_b]
            mask = ((pa + pb) > 0.6) & (torch.abs(pa - pb) < margin_thresh)

        if mask.any():
            # specialist outputs binary {0,1} corresponding to {class_a,class_b}
            bin_logits = specialist(imgs[mask], sizes[mask])
            bin_pred = bin_logits.argmax(1)
            refined = torch.where(
                bin_pred == 0,
                torch.tensor(class_a, device=device),
                torch.tensor(class_b, device=device),
            )
            preds[mask] = refined

    return preds