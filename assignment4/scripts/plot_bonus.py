"""Plots for the Q2.3 and Q3.3 bonus analyses."""
import json

import matplotlib.pyplot as plt

# Q2.3: validation PSNR during training with 2 vs 4 source views.
fig, ax = plt.subplots(figsize=(6, 4))
for s in (2, 4):
    with open(f"outputs/bonus/lvsm_s{s}/log.jsonl") as f:
        rows = [json.loads(line) for line in f]
    ax.plot([r["epoch"] for r in rows], [r["val"]["psnr"] for r in rows], label=f"Trained with S={s}")
ax.set(xlabel="Epoch", ylabel="Val PSNR (dB, at the training S)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("images/q2_bonus_curves.png", dpi=120)

# Q3.3: accuracy, runtime and memory against the number of input views.
with open("outputs/bonus/vggt_views.json") as f:
    results = json.load(f)
views = [int(v) for v in results]
r = list(results.values())
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(views, [m["depth_abs_rel"] for m in r], "o-", label="Depth AbsRel")
axes[0].plot(views, [1 - m["depth_delta1"] for m in r], "o-", label="1 - Depth δ1")
axes[0].set(title="Depth error (lower is better)")
for key, label in [("rotation_auc_30", "Rotation"), ("translation_auc_30", "Translation"), ("pose_auc_30", "Joint")]:
    axes[1].plot(views, [m[key] for m in r], "o-", label=label)
axes[1].set(title="Pose AUC@30 (higher is better)")
axes[2].plot(views, [m["peak_memory_mb"] for m in r], "o-", label="Peak memory (MB)")
axes[2].set(title="Peak GPU memory, one sample")
for ax in axes:
    ax.set(xscale="log", xlabel="Number of views")
    ax.set_xticks(views, views, minor=False)
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig("images/q3_bonus_views.png", dpi=120)
