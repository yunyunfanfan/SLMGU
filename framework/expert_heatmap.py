import csv
import os

import torch


def _gate_matrix(layer):
    """Return relation x expert gate probabilities for a MoE deletion layer."""
    if not (hasattr(layer, "del_relation_emb") and hasattr(layer, "del_gate")):
        return None
    device = next(layer.parameters()).device
    rel = torch.arange(layer.num_relations, device=device)
    with torch.no_grad():
        logits = layer.del_gate(layer.del_relation_emb(rel))
        probs = torch.softmax(logits / layer.gate_temperature, dim=-1)
        if getattr(layer, "gate_mode", "soft") == "hard":
            idx = probs.argmax(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
    return probs.detach().cpu()


def _save_csv(matrix, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relation"] + [f"expert_{i}" for i in range(matrix.shape[1])])
        for r, row in enumerate(matrix.tolist()):
            writer.writerow([r] + row)


def _save_png(matrix, path, title):
    # Import lazily so normal training does not require matplotlib until enabled.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_rel, n_exp = matrix.shape
    fig_h = max(3.0, min(18.0, 0.22 * n_rel + 1.8))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * n_exp + 2.5), fig_h))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Relation type")
    ax.set_xticks(range(n_exp))
    ax.set_xticklabels([f"E{i}" for i in range(n_exp)])
    if n_rel <= 60:
        ax.set_yticks(range(n_rel))
        ax.set_yticklabels([str(i) for i in range(n_rel)])
    else:
        step = max(1, n_rel // 30)
        ax.set_yticks(range(0, n_rel, step))
        ax.set_yticklabels([str(i) for i in range(0, n_rel, step)])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("gate weight")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_expert_heatmaps(model, checkpoint_dir, tag="final"):
    """Save MoE expert heatmaps for deletion1/deletion2 if the model has them.

    Outputs:
      expert_heatmaps/{tag}_deletion{1,2}_gate_heatmap.png
      expert_heatmaps/{tag}_deletion{1,2}_gate_weights.csv
    """
    out_dir = os.path.join(checkpoint_dir, "expert_heatmaps")
    saved = []
    for name in ["deletion1", "deletion2"]:
        layer = getattr(model, name, None)
        matrix = _gate_matrix(layer) if layer is not None else None
        if matrix is None:
            continue
        csv_path = os.path.join(out_dir, f"{tag}_{name}_gate_weights.csv")
        png_path = os.path.join(out_dir, f"{tag}_{name}_gate_heatmap.png")
        _save_csv(matrix, csv_path)
        try:
            _save_png(matrix, png_path, f"{tag}: {name} relation-to-expert gate")
            saved.append(png_path)
        except Exception as exc:
            # Keep the run alive even if plotting fails; CSV is still usable.
            print(f"[expert heatmap] failed to save {png_path}: {exc}")
        saved.append(csv_path)
    return saved
