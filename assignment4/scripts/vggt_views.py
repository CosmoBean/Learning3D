"""Q3.3: evaluate a trained Mini-VGGT with different numbers of input views."""
import argparse
import json
import time

import torch

from a4data.loading import make_dataloader
from models.vggt.eval import evaluate
from models.vggt.runtime import build_components

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="outputs/vggt_train/best.pt")
parser.add_argument("--views", type=int, nargs="+", default=[2, 4, 8, 16])
parser.add_argument("--output", default="outputs/bonus/vggt_views.json")
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
config = checkpoint["config"]
model, loss_fn, geometry = build_components(config, from_checkpoint=True)
model.load_state_dict(checkpoint["model"])
model.cuda().eval()

results = {}
for v in args.views:
    data = config["data"] | {"num_views": v, "eval_num_frames": 32, "seed": 825}
    metrics = evaluate(model, make_dataloader(data, "val", geometry), "cuda", config["loss"])
    # Runtime and peak memory of one forward pass on a single V-view sample.
    rgb = torch.rand(1, v, 3, 224, 224, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model({"rgb": rgb})
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            model({"rgb": rgb})
        torch.cuda.synchronize()
    metrics["forward_ms"] = (time.time() - start) / 10 * 1000
    metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    results[v] = metrics
    print(v, {k: round(metrics[k], 4) for k in
              ["depth_abs_rel", "depth_delta1", "rotation_auc_30", "translation_auc_30",
               "pose_auc_30", "forward_ms", "peak_memory_mb"]}, flush=True)

with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
