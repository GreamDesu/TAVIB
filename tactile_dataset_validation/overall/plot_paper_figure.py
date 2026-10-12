"""Render the saved test matrices for the manuscript; no training is performed.

Run: python tactile_dataset_validation/overall/plot_paper_figure.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from summary import TASKS, VALIDATION, matrix


OUTPUT = VALIDATION / "overall" / "results" / "confusion_matrices_row.pdf"


def main() -> None:
    # Dimensions in millimetres match the manuscript's full text width.
    width, height, side, bottom = 176, 74, 31, 32
    lefts = [21, 56, 97, 144]
    titles = ["Experimental", "Simulated", "Strawberries", "Contact motion"]
    colors = ["#ffffff", "#d9e7f2", "#9abed9", "#4d8bb6", "#175680", "#092f50"]
    cmap = LinearSegmentedColormap.from_list("paper_blue", colors)
    norm = Normalize(0, 100)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.stretch": "condensed",
        "font.size": 10.5,
        "text.color": "#20262b",
        "axes.labelcolor": "#20262b",
        "xtick.color": "#20262b",
        "ytick.color": "#20262b",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    for index, ((task, folder, method, _), left, title) in enumerate(zip(TASKS, lefts, titles)):
        labels, values = matrix(VALIDATION / folder / "results" / f"test_{method}_confusion_matrix.csv")
        assert values.shape == (len(labels), len(labels))
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-12)
        display = [f"Day {label}" for label in labels] if task == "strawberries" else [label.replace("_", " ").capitalize() if task == "slippage" else label for label in labels]
        ax = fig.add_axes([left / width, bottom / height, side / width, side / height])
        edges = np.arange(len(labels) + 1) - 0.5
        mesh = ax.pcolormesh(edges, edges, values * 100, cmap=cmap, norm=norm,
                             edgecolors="#dbe2e8", linewidth=0.35, rasterized=False)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(len(labels)), display)
        ax.set_yticks(range(len(labels)), display)
        ax.tick_params(length=0, pad=2, labelsize=10.5)
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right", rotation_mode="anchor")
        # The two material tasks share the same row labels, shown beside panel (a).
        if index == 1:
            ax.tick_params(labelleft=False)
        for spine in ax.spines.values():
            spine.set_color("#8a99a6")
            spine.set_linewidth(0.45)
        for i, j in np.ndindex(values.shape):
            value = values[i, j]
            if value == 0:
                continue
            label = "100" if value == 1 else f"{100 * value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10.5,
                    color="white" if value >= 0.6 else "#20262b")
        ax.set_title(f"({chr(97 + index)}) {title}", loc="left", fontsize=11,
                     fontweight="bold", pad=8)
    fig.text(0.5, 13 / height, "Predicted class", ha="center", va="bottom", fontsize=10.5)
    fig.text(0.018, (bottom + side / 2) / height, "True class", ha="center",
             va="center", rotation=90, fontsize=10.5)
    cax = fig.add_axes([58 / width, 7 / height, 60 / width, 2 / height])
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    colorbar.ax.set_xticklabels(["0%", "50%", "100%"])
    colorbar.ax.tick_params(labelsize=10.5, length=2, width=0.4, pad=2)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.outline.set_linewidth(0.4)
    colorbar.outline.set_edgecolor("#8a99a6")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={"Title": "Test confusion matrices",
                                "Subject": "Row-normalized percentages; slippage averaged equally across recordings"})
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
