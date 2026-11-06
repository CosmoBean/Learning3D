"""Q3.0: Camera + depth supervision for all selected views.

Targets use Q1's normalized first-camera coordinates. No GT alignment or
confidence-based pixel selection is applied before computing the loss.
Fill the None placeholders and remove the corresponding NotImplementedError.
"""


import torch

from models.vggt.camera import pose_encoding
from models.vggt.loss_utils import trimmed_mean

DEFAULT_WEIGHTS = {"camera": 5.0, "depth": 1.0}


def camera_loss(predictions, targets):
    """Q3.0: Return a scalar camera loss, averaged over B and all V views.

    Args:
        predictions: pose_enc (B,V,9), the raw activated camera prediction.
        targets: K (B,V,3,3), T_world_from_camera (B,V,4,4), depth (B,V,1,H,W).

    The provided pose_encoding returns (B,V,9): world-to-camera translation
    (3), XYZW quaternion (4), and vertical/horizontal FoV (2).
    Use translation L1 (clamped at 100), quaternion L1, and 0.5 * FoV L1.
    Compare pose_enc directly to the GT encoding; do not decode and re-encode
    predictions. The camera head produces one prediction stage.
    """
    hw = targets["depth"].shape[-2:]
    predicted = predictions["pose_enc"]
    expected = pose_encoding(targets["T_world_from_camera"], targets["K"], hw)
    pred_t, gt_t = predicted[..., :3], expected[..., :3]
    pred_q, gt_q = predicted[..., 3:7], expected[..., 3:7]
    pred_fov, gt_fov = predicted[..., 7:], expected[..., 7:]
    # TODO (Q3.0): Mean translation L1, clamping each error at 100 before averaging.
    translation = (pred_t - gt_t).abs().clamp(max=100).mean()
    # TODO (Q3.0): Mean quaternion L1 and mean FoV L1.
    rotation = (pred_q - gt_q).abs().mean()
    fov = (pred_fov - gt_fov).abs().mean()
    return translation + rotation + 0.5 * fov


def depth_gradient_loss(prediction, target, valid):
    """Q3.0: Return the four-scale masked depth-residual gradient loss.

    Args:
        prediction, target: (B,V,1,H,W), normalized z-depth.
        valid: (B,V,1,H,W), bool GT mask.

    At strides 1,2,4,8, subsample residuals/mask, compute horizontal/vertical
    absolute residual differences (clamped at 100), and keep edges only if
    both pixels are valid. Divide each scale's sum by its valid pixel count
    (clamped to >=1), then average scales. Preserve autograd when no edge is valid.
    """
    residual = torch.where(valid, prediction - target, 0.0)
    total = prediction.sum() * 0
    for stride in (1, 2, 4, 8):
        diff = residual[..., ::stride, ::stride]
        mask = valid[..., ::stride, ::stride]
        valid_x = mask[..., 1:] & mask[..., :-1]
        valid_y = mask[..., 1:, :] & mask[..., :-1, :]
        # TODO (Q3.0): Absolute adjacent residual differences, clamped at 100.
        dx = (diff[..., 1:] - diff[..., :-1]).abs().clamp(max=100)
        dy = (diff[..., 1:, :] - diff[..., :-1, :]).abs().clamp(max=100)
        # TODO (Q3.0): Sum valid x/y differences, divide by mask.sum().clamp_min(1).
        edges = torch.where(valid_x, dx, 0).sum() + torch.where(valid_y, dy, 0).sum()
        scale_loss = edges / mask.sum().clamp_min(1)
        total = total + scale_loss
    return total / 4


def depth_loss(predictions, targets):
    """Q3.0: Return scalar regression + confidence + gradient depth loss.

    Args:
        predictions: depth and depth_conf, both (B,V,1,H,W); confidence >=1.
        targets: depth (B,V,1,H,W), depth_valid (B,V,1,H,W), bool.

    Use trimmed_mean(error) and trimmed_mean(confidence*error - 0.2*log(confidence))
    with the provided 98% filtering helper, plus your depth_gradient_loss.
    Only the finite, positive, GT-valid pixels below participate in regression.
    """
    valid = (
        targets["depth_valid"]
        & torch.isfinite(targets["depth"])
        & (targets["depth"] > 0)
    )
    if not valid.any():
        raise ValueError("Loss requires positive valid depth")
    pred_depth = predictions["depth"][valid]
    gt_depth = targets["depth"][valid]
    confidence = predictions["depth_conf"][valid]
    # TODO (Q3.0): Per-pixel absolute depth error.
    error = (pred_depth - gt_depth).abs()  # (num_valid_pixels,).
    # TODO (Q3.0): Use trimmed_mean for regression and confidence-weighted loss.
    regression = trimmed_mean(error)
    confidence_loss = trimmed_mean(confidence * error - 0.2 * torch.log(confidence))
    gradient = depth_gradient_loss(predictions["depth"], targets["depth"], valid)
    return regression + confidence_loss + gradient


def compute_loss(predictions, targets, weights=None, return_components=False):
    """Provided: combine camera/depth losses and return logging components."""
    weights = DEFAULT_WEIGHTS if weights is None else DEFAULT_WEIGHTS | weights
    terms = {
        "camera": camera_loss(predictions, targets),
        "depth": depth_loss(predictions, targets),
    }
    loss = weights["camera"] * terms["camera"] + weights["depth"] * terms["depth"]
    return (loss, terms) if return_components else loss


def point_loss(predictions, targets):
    """Reserved for Q3.3 (bonus); not part of Q3.0.

    Add this loss to compute_loss with your chosen weight and logging term
    when enabling point prediction; it is not called by the default setup.
    """
    raise NotImplementedError("Q3.3: Point loss is reserved for the bonus question")
