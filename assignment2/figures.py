"""Builds the report figures from the eval images: one row per test chair, titled columns (like the course example)."""
import imageio.v3 as iio
import matplotlib.pyplot as plt

CHAIRS = (0, 400, 100)
COLS = ("Image", "Ground truth", "Prediction")


def panels(path):
    """Eval images are square panels side by side (input | GT | prediction) -> list of panels."""
    im = iio.imread(path)[..., :3]
    w = im.shape[0]
    return [im[:, i * w:(i + 1) * w] for i in range(im.shape[1] // w)]


def figure(out, rows, titles, colorbar=None):
    fig, axes = plt.subplots(len(rows), len(titles), figsize=(2.4 * len(titles), 2.4 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            axes[r][c].imshow(im)
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(titles[c])
    if colorbar:  # (label, max value) for the coolwarm error colouring
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, colorbar[1]))
        cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.03, pad=0.02, aspect=40)
        cb.set_label(colorbar[0])
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


for t in ("vox", "point", "mesh"):
    figure(f"output/q2_{t}.png", [panels(f"vis/{s}_{t}.png") for s in CHAIRS], COLS)

figure("output/q24_wsmooth.png",
       [panels(f"vis/{s}_mesh_10k.png")[:2] + [panels(f"vis/{s}_mesh{tag}.png")[2] for tag in ("_10k", "_ws1", "_ws5")]
        for s in CHAIRS],
       ("Image", "Ground truth", "w_smooth 0.1", "w_smooth 1", "w_smooth 5"))

figure("output/q25_error.png", [panels(f"output/q25_error_{s}.png") for s in CHAIRS], COLS,
       colorbar=("Distance to nearest point (red = 0.1 or more)", 0.1))

# chair #100 comes out empty for the implicit model, so use #500 here
figure("output/q31_vox_vs_implicit.png",
       [panels(f"vis/{s}_vox.png") + [panels(f"vis/{s}_implicit.png")[2]] for s in (0, 400, 500)],
       ("Image", "Ground truth", "Voxel", "Implicit"))
