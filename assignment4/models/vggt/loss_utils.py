"""Provided VGGT-style regression outlier filtering."""


def trimmed_mean(values, quantile=0.98):
    """Match VGGT's 98% outlier filtering for regression/confidence terms."""
    if values.numel() > 1000:
        values = values.clamp(max=100)
        threshold = (
            values.detach()
            .flatten()
            .kthvalue(round(quantile * (values.numel() - 1)) + 1)
            .values
        )
        retained = values[values < threshold]
        if retained.numel() > 1000:
            values = retained
    return values.mean()
