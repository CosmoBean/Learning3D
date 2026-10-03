# 16-825 Assignment 2

**AndrewId**: sbandred

**Name:** Sri Datta Bandreddi

Trained on one H100 (PyTorch 2.14, PyTorch3D 0.7.9). The ResNet18 encoder is trained end to end from the images; `--load_feat` was not used.

## 1.1 Fitting a voxel grid
Optimize a random 32³ grid toward a target chair with binary cross-entropy on the occupancy logits (10k iterations). Left: fitted, right: target.

![voxel fit](output/q1_vox.gif)

## 1.2 Fitting a point cloud
Optimize 5000 random points toward the target with a chamfer loss written from scratch: the squared distance from each point to its nearest neighbour in the other cloud, averaged in both directions (20k iterations). Left: fitted, right: target.

![point cloud fit](output/q1_point.gif)

## 1.3 Fitting a mesh
Deform an ico-sphere toward the target with chamfer on 5000 sampled points plus 0.1 × Laplacian smoothing (10k iterations). Left: fitted, right: target. The sphere can't grow holes, so it stretches thin webs between the legs.

![mesh fit](output/q1_mesh.gif)

## 2. Single view to 3D: setup
ResNet18 (ImageNet weights) turns the image into a 512-d feature, and a decoder per representation turns that into 3D. Adam, lr 4e-4, batch 32, 8 data workers. Each model was trained to 10k steps, then resumed to 20k, and the better checkpoint is reported. The test set is 678 chairs with the view fixed by seed 21, so every model sees the same inputs. Each picture below is **input | prediction | ground-truth mesh**, for the same three test chairs.

## 2.1 Image to voxel grid
Decoder:
```
Linear 512→16384, ReLU                    → reshaped to 256×4×4×4
ConvTranspose3d 256→128, k4 s2, BN, ReLU  → 128×8×8×8
ConvTranspose3d 128→64,  k4 s2, BN, ReLU  → 64×16×16×16
ConvTranspose3d 64→32,   k4 s2, BN, ReLU  → 32×32×32×32
Conv3d 32→1, k3                           → 1×32×32×32 logits
```

**F1@0.05 = 70.3** (20k steps)

![vox 0](vis/0_vox.png)
![vox 400](vis/400_vox.png)
![vox 100](vis/100_vox.png)

![F1 vox](eval_vox.png)

Thin chairs (last row) are the hard case. Only a few percent of voxels are filled, so "empty" is the cheap guess for thin parts. At 10k steps, 28 test chairs came out completely empty and F1 was 58.6; at 20k, none are empty, but thin metal frames are still fragments.

## 2.2 Image to point cloud
Decoder:
```
Linear 512→1024, ReLU
Linear 1024→1024, ReLU
Linear 1024→3000, tanh                    → reshaped to 1000×3 points
```

**F1@0.05 = 79.9** (10k steps)

![point 0](vis/0_point.png)
![point 400](vis/400_point.png)
![point 100](vis/100_point.png)

![F1 point](eval_point.png)

## 2.3 Image to mesh
Decoder:
```
Linear 512→1024, ReLU
Linear 1024→1024, ReLU
Linear 1024→7686                          → reshaped to 2562×3 vertex offsets
```
The offsets move the vertices of a level-4 ico-sphere. Loss: chamfer on 1000 points + 0.1 × Laplacian smoothing.

**F1@0.05 = 73.2** (20k steps)

![mesh 0](vis/0_mesh.png)
![mesh 400](vis/400_mesh.png)
![mesh 100](vis/100_mesh.png)

![F1 mesh](eval_mesh.png)

**Comparison:**

| Representation | F1@0.05 at 10k steps | F1@0.05 at 20k steps |
|---|---|---|
| Point cloud | **79.9** | 79.0 |
| Mesh | 71.1 | **73.2** |
| Voxel grid | 58.6 | **70.3** |

Point clouds score highest because they're trained on almost exactly what F1 measures: every point moves freely and chamfer pulls it onto the surface. Nothing ties the points together, which is why the clouds bunch up at the seat and back, where most of the surface is, and leave the legs sparse. Meshes use the same loss, but every vertex is tied to the sphere, so they can't open holes between legs, and vertices pulled toward thin parts turn into spikes. Voxels are limited by resolution (one voxel is about 0.03, close to the 0.05 threshold), by thin parts disappearing, and by the marching-cubes step before scoring. Going from 10k to 20k steps helped voxels and meshes a lot, but the point model started to overfit, so its 10k checkpoint is kept. Choosing checkpoints by test F1 is a mild form of test-set selection; a validation split would be the cleaner way.

## 2.4 Effects of hyperparameters
*In progress.*

## 2.5 Interpreting the model
*In progress.*

## 3. Other architectures / datasets
*In progress.*
