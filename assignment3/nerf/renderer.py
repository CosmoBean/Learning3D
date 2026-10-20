import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        decay = torch.exp(-rays_density * deltas)
        T = torch.cumprod(
            torch.cat([torch.ones_like(decay[:, :1]), decay[:, :-1] + eps], dim=1), dim=1
        )

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = T * (1.0 - decay)
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))
            if self._white_background:
                feature = feature + (1.0 - weights.sum(dim=1))

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q4.3): Convert signed distance to density with alpha, beta parameters
    # alpha * Psi_beta(-d), where Psi_beta is the CDF of a zero-mean Laplace distribution
    psi = 0.5 * torch.exp(-torch.abs(signed_distance) / beta)
    return alpha * torch.where(signed_distance > 0, psi, 1.0 - psi)

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q4.3): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'feature': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# (Q4.2) Inverse transform sampling of depths from a piecewise-constant PDF along each ray
def sample_pdf(bins, weights, n_samples, deterministic=False, eps=1e-5):
    # bins: (B, M + 1) bin edges, weights: (B, M) bin weights
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (B, M + 1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device).expand(bins.shape[0], n_samples)
    else:
        u = torch.rand(bins.shape[0], n_samples, device=bins.device)
    u = u.contiguous()

    # Find the bin each u falls in and linearly interpolate within it
    idx = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(idx - 1, min=0)
    above = torch.clamp(idx, max=cdf.shape[-1] - 1)

    cdf_below, cdf_above = torch.gather(cdf, -1, below), torch.gather(cdf, -1, above)
    bins_below, bins_above = torch.gather(bins, -1, below), torch.gather(bins, -1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom

    return bins_below + t * (bins_above - bins_below)


# (Q4.2) Volume renderer with a coarse pass (stratified samples) and a fine pass
# (stratified + importance samples drawn from the coarse weights)
class VolumeRendererCoarseFine(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self.n_pts_fine = cfg.n_pts_fine

    def _render(self, implicit_fn, ray_bundle):
        n_pts = ray_bundle.sample_shape[1]

        implicit_output = implicit_fn(ray_bundle)
        density = implicit_output['density']
        feature = implicit_output['feature']

        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        weights = self._compute_weights(
            deltas.view(-1, n_pts, 1),
            density.view(-1, n_pts, 1)
        )

        feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))
        if self._white_background:
            feature = feature + (1.0 - weights.sum(dim=1))

        depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

        return feature, depth, weights

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Coarse pass
            cur_ray_bundle = sampler(cur_ray_bundle)
            feature_coarse, _, weights = self._render(implicit_fn.coarse, cur_ray_bundle)

            # Importance sample new depths from the coarse weights
            z_vals = cur_ray_bundle.sample_lengths[..., 0]  # (B, n_pts)
            z_mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_fine = sample_pdf(
                z_mids, weights[..., 1:-1, 0].detach(), self.n_pts_fine,
                deterministic=not self.training,
            )
            z_vals, _ = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)
            z_vals = z_vals[..., None]  # (B, n_pts + n_pts_fine, 1)

            fine_ray_bundle = cur_ray_bundle._replace(
                sample_points=cur_ray_bundle.origins[:, None] + cur_ray_bundle.directions[:, None] * z_vals,
                sample_lengths=z_vals,
            )

            # Fine pass
            feature, depth, _ = self._render(implicit_fn.fine, fine_ray_bundle)

            # Return
            cur_out = {
                'feature': feature,
                'feature_coarse': feature_coarse,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'volume_sdf': VolumeSDFRenderer,
    'volume_coarse_fine': VolumeRendererCoarseFine,
}
