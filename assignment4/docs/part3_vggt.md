# 3. Multi-View Geometry with Mini-VGGT (40 points)

[Assignment overview](../README.md) · [Data contract](DATA_CONTRACT.md)

In this section, you will implement and train **Mini-VGGT** to reconstruct camera parameters and depth from multiple RGB views. Q1's canonicalized cameras and depth provide supervision. For visualization, point clouds are reconstructed from the predicted cameras and depth.

```text
RGB views → Frozen DINO → Alternating frame/global attention
                         ├→ Camera head → Camera loss
                         └→ Depth + confidence heads → Depth loss
```

## 3.0. Implementation

Complete the `# TODO (Q3.0)` blocks below. The encoder, transformer blocks, camera conversion and training/evaluation pipeline are provided. Follow the function docstrings for tensor shapes, activations and loss definitions. Model inputs and targets are prepared by `vggt_inputs` and `vggt_targets` in [a4data/dataset.py](../a4data/dataset.py).

### Alternating Attention

Implement `MiniVGGT.aggregate` in [model.py](../models/vggt/model.py) to alternate attention within each view and across views. Combine the final frame/global features for the prediction heads.

### Camera and Depth Prediction

Implement `CameraHead.predict` and `DepthHead.forward` in [heads.py](../models/vggt/heads.py) to predict camera parameters, depth and confidence from the aggregated features.

Compared with [official VGGT](https://github.com/facebookresearch/vggt), our heads make one camera prediction and use a depth decoder with RGB refinement, rather than iterative camera refinement and DPT.

### Camera and Depth Losses

Implement `camera_loss`, `depth_gradient_loss` and `depth_loss` in [losses.py](../models/vggt/losses.py) to supervise predictions using Q1's canonicalized camera and depth targets. The depth objective includes confidence-weighted regression and spatial gradient losses; `compute_loss` combines the camera and depth losses.

## 3.1. Overfitting (10 points)

Once your implementation is complete, use single-scene overfitting to debug the model and losses before full training.

**Default model:** A frozen DINOv2 ViT-B/14 encoder feeds 8 frame/global transformer pairs (16 blocks; width 384; 6 attention heads). The camera head adds 4 cross-view transformer blocks, and the depth head decodes patch features with RGB refinement.

Fit one fixed training clip:

```bash
python -m models.vggt.train --config configs/vggt.json --overfit --output outputs/vggt_overfit
```

Inspect `outputs/vggt_overfit/loss_curve.png` and `outputs/vggt_overfit/visualization/index.html`. Check that loss decreases and predicted geometry approaches GT.

> **Submission (Q3.1):** Include the overfit loss curve image (total, camera and depth loss) and screenshots comparing GT and predicted geometry on your assignment webpage.

## 3.2. Training and Evaluation (30 points)

Once single-scene overfitting works, train on the full training split. We recommend starting with the defaults in [configs/vggt.json](../configs/vggt.json). You may adjust other hyperparameters while keeping `num_views=8` and `sampling_range=32`.

```bash
python -m models.vggt.train --config configs/vggt.json --output outputs/vggt_train
```

For four GPUs, replace `python` with `torchrun --standalone --nproc_per_node=4`. The script saves `loss_curve.png`, `last.pt`, `best.pt` and `visualization/index.html` under the output directory. The HTML compares GT and predicted geometry on validation scenes; `best.pt` is selected by validation loss.

**TA's reference:** [Four-scene GT/prediction viewer](assets/minivggt_ta_reference.html) (download and open in a browser). Trained on a single NVIDIA A100 for ~4.4 hours. We encourage you to explore and improve on this baseline. Do not submit this reference.

To continue an interrupted run with its saved settings, optimizer and schedule:

```bash
python -m models.vggt.train --resume outputs/vggt_train/last.pt
```

Evaluate `best.pt` on validation and test using **8 views from the first 32 prepared frames, 224px and seed 825**. These flags override the checkpoint's data settings. Run the full splits without `--max-batches`:

```bash
python -m models.vggt.eval --checkpoint outputs/vggt_train/best.pt --split val --eval-root data/scannetpp/prepared_eval --num-views 8 --eval-num-frames 32 --image-size 224 --seed 825 --report outputs/vggt_train/metrics.json --visualization outputs/vggt_train/visualization/index.html
python -m models.vggt.eval --checkpoint outputs/vggt_train/best.pt --split test --eval-root data/scannetpp/prepared_eval --num-views 8 --eval-num-frames 32 --image-size 224 --seed 825 --report outputs/vggt_train/test_metrics.json
```

Report the following metrics for both splits:

| Metric | Definition | Better |
| --- | --- | --- |
| Depth AbsRel | Mean `abs(pred − GT) / GT` over valid pixels | Lower |
| Depth δ1 | Fraction with `max(pred/GT, GT/pred) < 1.25` | Higher |
| Rotation AUC | Pairwise relative rotation accuracy | Higher |
| Translation AUC | Pairwise translation-direction accuracy, allowing sign ambiguity | Higher |
| Joint Pose AUC | Uses the larger rotation/translation error per pair | Higher |

Report each AUC at 5°, 15° and 30°; values range from 0 to 1. Depth metrics use all GT-valid pixels; translation/joint AUC excludes pairs with undefined GT translation direction.

> **Submission (Q3.2):** Include the normal-training loss curve image (total, camera and depth loss), validation/test metrics, and screenshots comparing GT and predicted geometry for **four different scenes**. State your training settings and briefly discuss errors on held-out scenes.

## 3.3. Bonus: Analysis and Exploration (10 points)

Explore Mini-VGGT beyond the baseline and analyze what you find. Choose one of the following directions or propose your own question. One exploration is sufficient.

- **Point-map prediction:** Our baseline reconstructs points from predicted cameras and depth. Add a direct point-map prediction head and loss, then compare the two approaches.
- **Multi-task synergy:** Does learning cameras, depth and/or point maps jointly improve each task? Compare single-task and joint training.
- **Unfreezing DINO:** How does training the DINO encoder affect performance compared with keeping it frozen?
- **Number of views:** How does the number of input views affect accuracy, runtime and memory? Explore changes during training and/or evaluation.

Use controlled comparisons and support your conclusions with metrics and visualizations. Thoughtful analysis is encouraged even when an experiment does not improve performance.

> **Submission (Q3.3):** Describe your question, experimental settings, results and conclusions on your webpage. Include relevant plots or screenshots and discuss possible explanations and limitations.
