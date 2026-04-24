import matplotlib.pyplot as plt
import numpy as np


def plot_attention(weights, xlabels=None, ylabels=None, title="Attention weights", ax=None):
    """Heatmap of an (n_queries, n_keys) attention matrix."""
    w = np.asarray(weights)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.5 * w.shape[1] + 2, 0.5 * w.shape[0] + 1.5))
    im = ax.imshow(w, cmap="viridis", vmin=0, vmax=max(1.0, w.max()))
    ax.set_xlabel("key j")
    ax.set_ylabel("query i")
    ax.set_title(title)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax


def plot_softmax_bars(probs, title="softmax distribution", ax=None):
    p = np.asarray(probs)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(p)), p)
    ax.set_ylim(0, 1)
    ax.set_xlabel("index")
    ax.set_ylabel("probability")
    ax.set_title(title)
    return ax
