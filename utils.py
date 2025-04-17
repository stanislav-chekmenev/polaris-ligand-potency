import io
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torchvision

import config as cfg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Seeding everything with seed: {seed}")


def get_next_run_folder(base_dir="runs", evaluation=False):
    """
    Creates the next folder name with consecutive numbers in the given base directory.
    Example: 001, 002, etc.
    """
    if evaluation:
        base_dir = os.path.join(base_dir, "evaluation")
        if cfg.DEBUG:
            base_dir = os.path.join(base_dir, "debug")
        else:
            base_dir = os.path.join(base_dir, "final")
    else:
        suffix = "debug" if cfg.DEBUG else "train"
        base_dir = os.path.join(base_dir, suffix)
    os.makedirs(base_dir, exist_ok=True)
    existing = [name for name in os.listdir(base_dir) if name.isdigit()]
    next_number = max([int(name) for name in existing], default=0) + 1
    return f"{next_number:03d}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_name():
    full_model_name = "mol_predictor" if not cfg.MODEL_NAME else cfg.MODEL_NAME
    full_model_name += ".pth"
    model_name = "baseline_mlp.pth" if cfg.BASE else full_model_name
    return model_name


def plot_attention_heatmap(att_weights, title="Attention Weights"):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(
        att_weights,
        cmap="viridis",
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.ylabel("0: Mol emb, 1: GAT emb, 2: MACE emb, 3: Barycenter emb")

    # Add numerical values to each cell
    for (i, j), val in np.ndenumerate(att_weights):
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
        )

    # Convert plot to image tensor
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = torchvision.transforms.ToTensor()(plt.imread(buf))
    plt.close(fig)
    return image
