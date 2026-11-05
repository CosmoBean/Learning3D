# Data conventions

Shared coordinate and tensor conventions for all three parts. Function-specific instructions are in [geometry.py](../a4data/geometry.py).

## Geometry

- Camera coordinates: +X right, +Y down, +Z forward. Use column vectors: `X_world = R @ X_camera + t`. Poses are camera-to-world transforms.
- `K` uses pixel coordinates. Pixel centers are integer `(u,v)`, starting at `(0,0)`; image tensors use `[v,u]` indexing.
- Depth is optical-axis **z-depth**, in meters before canonicalization. Valid depth is finite and positive. Invalid depth and output points are zero; depth validity does not mask RGB.
- Rays have camera-center origins and unit directions. Plücker channels are `[d_x,d_y,d_z,m_x,m_y,m_z]`, where `m = o × d`.

## Canonicalization

Geometry uses the first sampled camera's frame and a shared scene scale. Camera 0 is identity, and the valid-point centroid has distance 1 from the origin. See `canonicalize_cameras` and the provided [compute_scene_scale](../a4data/scale.py) for implementation details.

## Dataloader output

Shapes below describe one sample with `V` views and image size `H×W`; the dataloader prepends a batch dimension `B`. Floating-point tensors are float32, validity is bool, and indices are int64.

| Field | Shape | Meaning |
|---|---|---|
| `rgb` | `(V,3,H,W)` | Cropped/resized RGB images in [0,1] |
| `K` | `(V,3,3)` | Pixel intrinsics |
| `T_world_from_camera` | `(V,4,4)` | Canonicalized camera-to-world poses |
| `depth`, `depth_valid` | `(V,1,H,W)` | Canonicalized z-depth and validity |
| `points_world` | `(V,3,H,W)` | Canonicalized depth-unprojected points |
| `ray_origins`, `ray_directions` | `(V,3,H,W)` | Rays in the canonical world frame |
| `plucker` | `(V,6,H,W)` | Direction followed by moment |
| `frame_numbers` | `(V,)` | Original frame IDs |
| `original_world_from_reference` | `(4,4)` | Original reference camera pose |
| `scene_scale` | scalar | Scale divisor in meters |
| `sequence_name` | string | Scene/clip identifier |

Geometry TODOs return channel-last maps `(V,H,W,3/6)`; the provided loader converts them to the channel-first layouts above. Model input and supervision helpers are documented in [a4data/dataset.py](../a4data/dataset.py).

## Prepared validation/test

The provided [evaluation loader](../a4data/scannetpp/prepared.py) reads first-frame geometry stored in meters, computes the shared scene scale after selecting views, and returns the same canonicalized fields above. It bypasses student geometry. See [Part 1, Section 1.6](part1_data.md#16-understanding-the-provided-dataloader) for sampling and configuration.
