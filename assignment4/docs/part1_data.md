# 1. Multi-View Data Preprocessing (20 points)

[Assignment overview](../README.md) · [Data contract](DATA_CONTRACT.md)

The course archive extracts into:

```text
data/scannetpp/
├── processed/                         # 224 training scenes
│   └── <scene>/iphone/
│       ├── images/*.jpg
│       ├── depths/*.png
│       └── cameras.json
└── prepared_eval/
    ├── manifest.json
    ├── val/<scene>/frame_XXXXXX.pt      # 28 scenes, 128 frames each
    └── test/<scene>/frame_XXXXXX.pt     # 28 scenes, 128 frames each
```

In this section, you will implement preprocessing functions that the training dataloader calls for each sample. Preprocessed validation and test data are provided. A key step in training multi-view models is **canonicalization**: expressing each sample in its first camera’s coordinate frame and normalizing its scale. This gives the model consistent coordinate and scale conventions across training samples.

Complete the `# TODO (Q1.x)` blocks in [a4data/geometry.py](../a4data/geometry.py).

Read [the coordinate conventions](DATA_CONTRACT.md) before starting.

## 1.1. Resizing Camera Intrinsics

Our dataloader first center-crops inputs to a square and adjusts the camera intrinsics. Implement `resize_intrinsics` to update these intrinsics for the subsequent resize to 224×224.

## 1.2. Depth Unprojection

Implement `unproject_depth` to convert depth pixels into world-space 3D points using the supplied camera poses. Depth is optical-axis z-depth, not distance along a unit ray.

After Q1.1–1.2, visualize points and cameras in the original world frame (meters):

```bash
python scripts/visualize_data.py --stage unprojection --num-scenes 5 --max-points 10000 --output outputs/unprojection
```

## 1.3. Camera and Depth Canonicalization

Implement `canonicalize_cameras` to express geometry in the first camera's frame and apply the shared scene scale from `compute_scene_scale`. See the function docstring for the steps and output conventions.

After Q1.3, visualize canonicalized geometry. Camera 0 should be identity, with the valid-point centroid approximately 1 unit from the origin:

```bash
python scripts/visualize_data.py --stage canonicalization --num-scenes 5 --max-points 10000 --output outputs/canonicalization
```

Open each output directory's `index.html` to select scenes/views and inspect camera coordinates. Both commands use the same frames when the seed and data settings match.

**Optional provided validation reference:**

```bash
python scripts/visualize_data.py --reference --num-scenes 5 --max-points 10000 --output outputs/geometry_reference
```

Open `outputs/geometry_reference/index.html` to view the validation reference. It bypasses your Q1 implementation; submit your own data visualizations, not the reference.

## 1.4. Camera Rays

Implement `camera_rays` to compute each pixel's ray origin and unit direction in the canonical world frame, using the resized intrinsics and camera-to-world poses.

## 1.5. Plücker Coordinates

Implement `plucker_coordinates` to represent each ray as six channels: direction followed by moment. Follow the convention in the data contract.

## 1.6. Understanding the Provided Dataloader

The dataloader is provided; no additional implementation is required here. Its settings are in [configs/scannetpp/iphone/data.json](../configs/scannetpp/iphone/data.json):

- **Train:** The loader samples `num_views` frames in random order from a `sampling_range` window and calls your Q1 functions. The first selected view is camera 0.
- **Validation/test:** The loader selects fixed views from the first `eval_num_frames` prepared frames, keeping frame 0 as the reference. It applies `compute_scene_scale` after sampling and bypasses your Q1 functions.

Defaults: 8 views, 224px, seed 825 and a 128-frame evaluation prefix. Each model section provides its evaluation commands. `sampling_range` affects training only.

## 1.7. Final Check

After completing all Q1 TODOs, check four training samples:

```bash
python scripts/check_data.py --config configs/scannetpp/iphone/data.json --split train --max-samples 4
```

Check the reported shapes, depth validity, point/ray reprojection errors (< 0.01 pixels) and centroid distance (approximately 1).

> **Submission:** Include screenshots of your data visualizations after unprojection and canonicalization. In the canonicalization screenshots, show the first camera center and valid-point centroid distance. Include the `check_data.py` output and briefly explain canonicalization, world-space ray directions and the Plücker moment.
