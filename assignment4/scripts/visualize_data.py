"""Visualize online RGB, depth, cameras and points in a standalone HTML viewer."""

import argparse
from pathlib import Path

from a4data import geometry as default_geometry
from a4data.loading import add_data_arguments, data_config, make_dataloader
from a4data.visualization import export_html


def main(geometry=default_geometry):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument(
        "--stage",
        choices=["unprojection", "canonicalization", "full"],
        default="full",
        help="Stop after the selected Q1 step",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Show the provided prepared validation reference",
    )
    parser.add_argument("--index", type=int, default=0, help="First scene index")
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Display limit per view; does not change loaded data",
    )
    parser.add_argument("--output", required=True, help="Directory for index.html")
    args = parser.parse_args()
    if args.reference and args.stage != "full":
        parser.error("--reference supports only --stage full")
    if args.split != "train":
        parser.error("Use train for Q1 geometry; use --reference for validation")
    config = data_config(args)
    if args.reference and not config.get("eval_root"):
        parser.error("--reference requires a prepared eval_root")
    dataset = make_dataloader(
        config, "val" if args.reference else "train", geometry
    ).dataset
    if not 0 <= args.index < len(dataset) or args.num_scenes < 1:
        parser.error("Choose a valid scene index and a positive --num-scenes")
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    samples = [
        dataset[i]
        if args.reference
        else dataset.get_sample(i, args.stage, fixed_sampling=True)
        for i in range(args.index, min(args.index + args.num_scenes, len(dataset)))
    ]
    for sample in samples:
        sample["visualization_stage"] = args.stage
        sample["visualization_reference"] = args.reference
        sample["visualization_label"] = (
            "Provided validation reference"
            if args.reference
            else f"Training geometry: {args.stage}"
        )
    destination = export_html(
        samples, Path(args.output) / "index.html", args.max_points
    )
    print(f"Saved {len(samples)} scenes to {destination.resolve()}")
    print("Open index.html in a browser; no server or port forwarding is required.")


if __name__ == "__main__":
    main()
