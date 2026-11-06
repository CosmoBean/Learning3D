"""Plot Mini-LVSM training curves from log.jsonl."""
import json
import sys

import matplotlib.pyplot as plt

log, out = sys.argv[1:]
with open(log) as f:
    rows = [json.loads(line) for line in f]
epochs = [r["epoch"] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, key, title in zip(axes, ["loss", "psnr", "ssim"], ["MSE loss", "PSNR (dB)", "SSIM"]):
    if key in rows[0]["train"]:
        ax.plot(epochs, [r["train"][key] for r in rows], label="Train")
    ax.plot(epochs, [r["val"][key] for r in rows], label="Val")
    ax.set(title=title, xlabel="Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=120)
