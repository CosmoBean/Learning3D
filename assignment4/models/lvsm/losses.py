"""Q2.0: Mini-LVSM supervision, a photometric loss on the rendered target views.

Fill the None placeholder and remove the NotImplementedError.
"""

# ruff: noqa: F841

import torch.nn.functional as F  # noqa: F401 -- used in TODOs


def view_synthesis_loss(pred_rgb, target_rgb):
    """Q2.0: Mean squared error between rendered and ground-truth target RGB.

    Args:
        pred_rgb: (B, T, 3, H, W), predicted RGB in [0, 1].
        target_rgb: (B, T, 3, H, W), ground-truth RGB in [0, 1].

    Returns:
        Scalar tensor averaged over all views and pixels. All pixels count:
        depth validity does not mask RGB.
    """
    # TODO (Q2.0): Mean squared error over every element of both tensors.
    loss = F.mse_loss(pred_rgb, target_rgb)  # Scalar ().
    return loss


def compute_loss(predictions, targets):
    """Provided: return a scalar target-image reconstruction loss.

    Args:
        predictions: Dict with rgb (B,T,3,H,W), predicted RGB in [0,1].
        targets: lvsm_targets(batch, num_source_views), with the same RGB shape,
            range and target view order. No depth-valid mask is applied to RGB.

    Returns:
        loss: Scalar tensor suitable for loss.backward().
    """
    return view_synthesis_loss(predictions["rgb"], targets["rgb"])
