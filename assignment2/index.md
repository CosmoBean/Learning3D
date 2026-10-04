# 16-825 Assignment 2

**AndrewId**: sbandred

**Name:** Sri Datta Bandreddi

Trained on one H100. The image encoder is trained from the images directly; `--load_feat` was not used.

## 1.1 Fitting a voxel grid
Fit a random grid to a chair using binary cross-entropy.

![voxel fit](output/q1_vox.gif)

## 1.2 Fitting a point cloud
Fit 5000 random points to a chair using chamfer loss (written from scratch).

![point cloud fit](output/q1_point.gif)

## 1.3 Fitting a mesh
Deform a sphere into a chair using chamfer loss plus smoothing. A sphere can't make holes, so it stretches thin sheets between the legs.

![mesh fit](output/q1_mesh.gif)

## 2. Setup
A ResNet18 turns the image into 512 numbers, and a decoder turns those into a 3D shape. Adam, learning rate 4e-4, batch size 32. Each model was trained for 10k steps, then 20k, and the better one is shown. All models are tested on the same 678 chairs with the same views.

## 2.1 Image to voxel grid
```
Linear 512→16384, ReLU                    → reshaped to 256×4×4×4
ConvTranspose3d 256→128, k4 s2, BN, ReLU  → 128×8×8×8
ConvTranspose3d 128→64,  k4 s2, BN, ReLU  → 64×16×16×16
ConvTranspose3d 64→32,   k4 s2, BN, ReLU  → 32×32×32×32
Conv3d 32→1, k3                           → 1×32×32×32 logits
```

**F1@0.05 = 70.3**

![voxel results](output/q2_vox.png)

![F1 vox](eval_vox.png)

Thin chairs (bottom row) are hard: most voxels are empty, so the model leaves thin parts out.

## 2.2 Image to point cloud
```
Linear 512→1024, ReLU
Linear 1024→1024, ReLU
Linear 1024→3000, tanh                    → reshaped to 1000×3 points
```

**F1@0.05 = 79.9**

![point results](output/q2_point.png)

![F1 point](eval_point.png)

## 2.3 Image to mesh
```
Linear 512→1024, ReLU
Linear 1024→1024, ReLU
Linear 1024→7686                          → reshaped to 2562×3 vertex offsets
```
The offsets move the points of a sphere mesh.

**F1@0.05 = 73.2**

![mesh results](output/q2_mesh.png)

![F1 mesh](eval_mesh.png)

**Comparison:**

| Representation | F1 at 10k steps | F1 at 20k steps |
|---|---|---|
| Point cloud | **79.9** | 79.0 |
| Mesh | 71.1 | **73.2** |
| Voxel grid | 58.6 | **70.3** |

- **Points score best.** Each point can move anywhere, and the loss is close to what F1 measures.
- **Meshes come next.** They use the same loss, but a sphere can't make holes, so thin parts become spikes.
- **Voxels score lowest.** A 32³ grid is coarse, and thin parts vanish.
- **Longer training** helped voxels and meshes, but the point model started to memorize the training chairs, so its 10k version is kept.

## 2.4 Effects of hyperparameters
Change the mesh smoothing weight `w_smooth` (10k steps each).

![w_smooth comparison](output/q24_wsmooth.png)

| w_smooth | F1@0.05 |
|---|---|
| 0.1 | **71.1** |
| 1 | 67.8 |
| 5 | 60.2 |

More smoothing gives cleaner meshes but a lower score. At 5, the legs merge into one tent shape. F1 only checks that points land near the surface, so the spiky meshes score higher.

## 2.5 Interpreting the model
Where does the point model go wrong? Each point is coloured by its distance to the nearest point of the other shape.

![error colouring](output/q25_error.png)

Recall and point distribution by height, over all 678 test chairs:

![error by height](output/q25_height.png)

Recall is about 85% in the middle of the chair but drops to 62% at the feet and 68% at the top of the backrest. The model places 74% of its points in the seat region (height 0.3 to 0.7), which holds only 57% of the surface. Seats look alike across chairs, so this is a safe bet under chamfer loss; legs and backrests vary more, so the model spreads fewer points there.

## 3.1 Implicit network
An MLP takes the image features and one 3D point, and says whether the point is inside the chair. Asking it at every point of a 32³ grid gives a voxel grid, trained and tested like 2.1.
```
image features (512) + point (x, y, z)    → 515
Linear 515→512, ReLU
Linear 512→512, ReLU
Linear 512→512, ReLU
Linear 512→1                              → inside / outside score
```

![voxel vs implicit](output/q31_vox_vs_implicit.png)

| Decoder | F1 at 10k | F1 at 20k | Empty outputs (of 678) |
|---|---|---|---|
| Voxel (3D conv) | 58.6 | **70.3** | 0 |
| Implicit (MLP) | 41.6 | 47.8 | 80 |

![F1 implicit](eval_implicit.png)

The implicit model is much worse. It makes smooth blobs and loses thin parts like legs, and 80 chairs come out empty. A plain MLP on raw (x, y, z) has trouble with sharp detail; adding a positional encoding (as in NeRF) is the usual fix. Its one advantage is size: 14× smaller than the voxel decoder.
