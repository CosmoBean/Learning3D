"""Check online geometry and the common dataloader."""

import argparse
import json
from pathlib import Path

from a4data import geometry as default_geometry
from a4data.loading import add_data_arguments, data_config, make_dataloader
from a4data.validation import validate_sample


def check_loader(loader, max_samples=4):
    stats = []
    for batch in loader:
        for i, name in enumerate(batch["sequence_name"]):
            sample = {k: v[i] for k, v in batch.items() if k != "sequence_name"}
            stats.append({"sequence_name": name, **validate_sample(sample)})
            if max_samples and len(stats) >= max_samples:
                break
        if max_samples and len(stats) >= max_samples:
            break
    return {
        "dataset_samples": len(loader.dataset),
        "checked_samples": len(stats),
        "samples": stats,
        "batch_shapes": {
            k: list(v.shape) for k, v in batch.items() if k != "sequence_name"
        },
    }


def main(geometry=default_geometry):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument(
        "--max-samples", type=int, default=4, help="0 checks the entire dataset"
    )
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    loader = make_dataloader(data_config(args), args.split, geometry)
    report = {"split": args.split, **check_loader(loader, args.max_samples)}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
