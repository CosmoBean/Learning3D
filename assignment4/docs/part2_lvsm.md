# 2. Novel-View Synthesis with Mini-LVSM (40 points)

[Assignment overview](../README.md) · [Data contract](DATA_CONTRACT.md)

In this section, you will predict target RGB images from source RGB images and source/target camera rays. Mini-LVSM is a decoder-only transformer: source tokens come from source RGB and Plücker rays, target-query tokens come from the target Plücker rays alone, and the model never sees target RGB.

The diagram below shows how images are turned into tokens, processed by the transformer and decoded back into the target view, trained with a photometric loss.

```text
source RGB + rays, target rays → Patchify + Linear → Concat per target + pos emb
                              → Transformer blocks, full attention
                              → Target-query tokens → RGB head → Unpatchify
                              → Predicted view → MSE loss
```

## 2.0. Implementation

Complete the `# TODO (Q2.0)` blocks below. The transformer blocks, positional encoding, evaluation metrics, evaluation script and visualization are provided. Use `lvsm_inputs` and `lvsm_targets` from [a4data/dataset.py](../a4data/dataset.py) for the input and target dictionaries; their shapes are documented in the model/loss docstrings.

Set `num_source_views` in [configs/lvsm.json](../configs/lvsm.json) and pass the same value to `lvsm_inputs(batch, num_source_views=...)` and `lvsm_targets(batch, num_source_views=...)`. Sources are selected from the randomly ordered views; all remaining views are targets. This partition does not change the shared camera/point normalization.

### Patchify and Unpatchify

Implement `patchify` and `unpatchify` in [model.py](../models/lvsm/model.py). `patchify` splits `(..., C, H, W)` images into non-overlapping `p×p` patches in row-major order and flattens each one in (row, column, channel) order, giving `(..., N, p·p·C)`. `unpatchify` is its exact inverse. Both accept any number of leading dimensions.

### Token Assembly and Forward Pass

Implement `MiniLVSM.build_tokens` and `MiniLVSM.forward` in [model.py](../models/lvsm/model.py).

- `build_tokens`: add the positional embedding (`sincos_2d_pos_embed`, shared by source and target tokens), then build one sequence per target view: all source tokens followed by that target's query tokens. Fold the target dimension into the batch: the output is `(B·T, (S+1)·N, D)`.
- `forward`: patchify and project the source RGB (shifted to `[-1, 1]`) concatenated with the source rays using `source_proj`, and the target rays using `target_proj`. Build the tokens, apply `input_norm` and the blocks, keep the target-query tokens, and decode them with `out_norm`, `rgb_head`, a sigmoid and `unpatchify`. Return `{"rgb": (B, T, 3, H, W)}`.

### Loss and Training Step

- Implement `view_synthesis_loss` in [losses.py](../models/lvsm/losses.py): the photometric loss (mean squared error) between predicted and ground-truth target RGB, averaged over all views and pixels. The provided `compute_loss` calls it.
- In `train_one_epoch` in [train.py](../models/lvsm/train.py), implement the forward pass and loss. Use bfloat16 autocast when `amp` is set, but compute the loss in float32. The optimizer step, scheduler, logging and checkpointing are provided.

## 2.1. Overfitting (10 points)

Once your implementation is complete, check that it can memorize one fixed batch before full training:

```bash
python -m models.lvsm.train --overfit 2000 --output-dir outputs/lvsm_overfit
```

Check that the loss decreases and PSNR reaches about 30 dB or more, and inspect `outputs/lvsm_overfit/val_epoch000.png`. Fix any problems before full training.

> **Submission (Q2.1):** Include the final overfit PSNR and the `val_epoch000.png` visualization on your assignment webpage.

## 2.2. Training and Evaluation (30 points)

Once overfitting works, train on the full training split using [configs/lvsm.json](../configs/lvsm.json). It references the shared data config; `data_overrides` can change view count, sampling range or batch size for this model.

```bash
python -m models.lvsm.train --config configs/lvsm.json
```

This prints train and validation metrics every epoch and saves `best.pt`, `last.pt`, `log.jsonl` (for your training curves) and `val_epoch*.png` (visualizations) under `outputs/lvsm/`. Delete `outputs/lvsm` before re-running, since `log.jsonl` is appended to.

Evaluate `best.pt` with ablations over the number of source views and the target-ray conditioning. This reports metrics for every combination and saves `metrics.json` and images to `outputs/lvsm/eval_val/`:

```bash
python -m models.lvsm.eval --checkpoint outputs/lvsm/best.pt --num-source-views 1 2 4 --conditioning correct shuffled none
```

`correct` uses the true target rays, `shuffled` uses the rays of a different target view, and `none` uses all-zero rays.

Report the following metrics for every combination:

| Metric | Definition | Better |
| --- | --- | --- |
| PSNR | How close the predicted image is to the real one, in dB, based on the pixel error (MSE) | Higher |
| SSIM | How similar the structure (edges, contrast, local patterns) of the two images is, from 0 to 1 | Higher |

> **Submission (Q2.2):** Include the training curves, validation PSNR/SSIM for each ablation setting, and the training and evaluation visualizations (columns: source views, ground truth, prediction). State your training settings and briefly discuss the effect of the number of source views and of the ray-conditioning ablation.

## 2.3. Bonus: Analysis and Exploration (10 points)

Explore Mini-LVSM beyond the baseline and analyze what you find. Choose one of the following directions or propose your own question. One exploration is sufficient.

- **Loss function:** Add a loss term such as L1, SSIM or a perceptual loss alongside the MSE loss. How do PSNR, SSIM and visual sharpness change?
- **Model capacity:** How do the patch size, transformer width or depth affect quality, runtime and memory?
- **Training view count:** Train with a different number of source views, or a random number per step. How does this change evaluation with 1, 2 and 4 source views?
- **Camera encoding:** Replace the Plücker ray tokens with another camera encoding, such as raw camera parameters, and compare.

Use controlled comparisons and support your conclusions with metrics and visualizations. Thoughtful analysis is encouraged even when an experiment does not improve performance.

> **Submission (Q2.3):** Describe your question, experimental settings, results and conclusions on your webpage. Include relevant plots or screenshots and discuss possible explanations and limitations.
