"""Image-quality metrics for view synthesis (provided to students)."""

import torch
import torch.nn.functional as F


def psnr(pred, target):
    """Per-image PSNR in dB for RGB in [0, 1].

    Args:
        pred, target: (N, 3, H, W).

    Returns:
        (N,) tensor.
    """
    mse = ((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-10))


def _gaussian_window(size, sigma, channels, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(pred, target, window_size=11, sigma=1.5):
    """Per-image SSIM (Gaussian window, valid convolution) for RGB in [0, 1].

    Args:
        pred, target: (N, C, H, W); H, W >= window_size.

    Returns:
        (N,) tensor, mean of the SSIM map over channels and pixels.
    """
    pred, target = pred.clamp(0, 1), target.clamp(0, 1)
    c = pred.shape[1]
    window = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)

    def blur(x):
        return F.conv2d(x, window, groups=c)

    mu_x, mu_y = blur(pred), blur(target)
    var_x = blur(pred * pred) - mu_x**2
    var_y = blur(target * target) - mu_y**2
    cov = blur(pred * target) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return ssim_map.flatten(1).mean(1)
