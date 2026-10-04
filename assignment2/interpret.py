"""Q2.5: where does the point-cloud model fail?
Per-point error colouring for a few test chairs, and error / point placement vs. height over the whole test set."""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import knn_points, sample_points_from_meshes

import dataset_location
import vis
from eval_model import get_args_parser, preprocess
from model import SingleViewto3D
from r2n2_custom import R2N2

SHOW = (0, 400, 100)  # same test chairs as the report
BINS = 10             # height bands, floor (0) to top (1)

args = get_args_parser().parse_args(["--type", "point"])
# same seeding and construction order as eval_model, so the test views match the report
random.seed(21); torch.manual_seed(21)
ds = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)
loader = iter(torch.utils.data.DataLoader(ds, batch_size=1, num_workers=6, collate_fn=collate_batched_R2N2, pin_memory=True, drop_last=True))
model = SingleViewto3D(args).to(args.device).eval()
model.load_state_dict(torch.load("checkpoint_point.pth")["model_state_dict"])

def colors(d):  # distance -> blue (0) .. red (>= 0.1)
    return torch.tensor(plt.cm.coolwarm((d / 0.1).clamp(0, 1).cpu().numpy())[:, :3])

covered, total, share_pred, share_gt = (np.zeros(BINS) for _ in range(4))
with torch.no_grad():
    for step, feed in enumerate(loader):
        images, mesh = preprocess(feed, args)
        pred = model(images, args)[0]
        gt = sample_points_from_meshes(mesh.to(args.device), 5000)[0]
        d_pred = knn_points(pred[None], gt[None]).dists[0, :, 0].sqrt()  # how far each prediction is from the surface
        d_gt = knn_points(gt[None], pred[None]).dists[0, :, 0].sqrt()    # how far each surface point is from a prediction

        lo, hi = gt[:, 1].min(), gt[:, 1].max()
        band = lambda y: ((y - lo) / (hi - lo) * BINS).long().clamp(0, BINS - 1).cpu().numpy()
        b_gt, b_pred = band(gt[:, 1]), band(pred[:, 1])
        total += np.bincount(b_gt, minlength=BINS)
        covered += np.bincount(b_gt, weights=(d_gt < 0.05).cpu().numpy(), minlength=BINS)
        share_gt += np.bincount(b_gt, minlength=BINS) / len(b_gt)
        share_pred += np.bincount(b_pred, minlength=BINS) / len(b_pred)

        if step in SHOW:
            img = feed['images'][0].cpu().numpy().repeat(2, 0).repeat(2, 1)
            renders = [vis.render(vis.points(p, colors(d)), size=img.shape[0])[0] for p, d in ((gt, d_gt), (pred, d_pred))]
            plt.imsave(f"output/q25_error_{step}.png", np.concatenate([(img * 255).astype(np.uint8)] + renders, 1))

n = step + 1
recall, share_gt, share_pred = 100 * covered / total, 100 * share_gt / n, 100 * share_pred / n
centers = (np.arange(BINS) + 0.5) / BINS
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#333333", "#e6e6e3"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)

a.barh(centers, recall, height=0.07, color=BLUE)
for i in (0, int(recall.argmax()), BINS - 1):  # label the ends and the best band only
    a.text(recall[i] + 1.5, centers[i], f"{recall[i]:.0f}%", va="center", fontsize=9)
a.set(xlim=(0, 100), xlabel="Recall at 0.05 (%)", ylabel="Normalized height",
      yticks=[0, 0.5, 1], yticklabels=["0 (floor)", "0.5", "1 (top)"])
a.set_title("Recall by height", loc="left", fontweight="bold")

for y, label, color in ((share_gt, "Ground truth", BLUE), (share_pred, "Prediction", ORANGE)):
    b.plot(y, centers, "o-", color=color, linewidth=2, markersize=6, label=label)
b.set(xlim=(0, 30), xlabel="Share of points (%)")
b.set_title("Point distribution by height", loc="left", fontweight="bold")
b.legend(frameon=False, loc="upper right")
plt.savefig("output/q25_height.png", bbox_inches="tight", dpi=150)
for c, r, g, p in zip(centers, recall, share_gt, share_pred):
    print(f"height {c:.2f}: covered {r:5.1f}%  GT share {g:5.1f}%  pred share {p:5.1f}%")
