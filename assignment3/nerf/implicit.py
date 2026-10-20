import math

import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers - 1):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            
        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        self.mlp_xyz = MLPWithInputSkips(
            cfg.n_layers_xyz,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_xyz,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_xyz,
            input_skips=cfg.append_xyz,
        )
        self.density_head = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1)

        # ReLU density can collapse to zero everywhere on dark scenes (e.g. materials); softplus cannot
        self.density_activation = torch.nn.Softplus() if cfg.get('density_activation', 'relu') == 'softplus' else torch.nn.ReLU()

        # (Q4.1) Optionally condition color on the viewing direction
        self.view_dependence = cfg.get('view_dependence', False)
        color_input_dim = cfg.n_hidden_neurons_xyz + (embedding_dim_dir if self.view_dependence else 0)
        self.color_head = torch.nn.Sequential(
            torch.nn.Linear(color_input_dim, cfg.n_hidden_neurons_dir),
            torch.nn.ReLU(True),
            torch.nn.Linear(cfg.n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )

    def forward(self, ray_bundle):
        points = ray_bundle.sample_points.view(-1, 3)
        embedding = self.harmonic_embedding_xyz(points)
        features = torch.relu(self.mlp_xyz(embedding, embedding))

        color_input = features
        if self.view_dependence:
            directions = ray_bundle.directions[:, None].expand(ray_bundle.sample_points.shape)
            color_input = torch.cat(
                [features, self.harmonic_embedding_dir(directions.reshape(-1, 3))], dim=-1
            )

        return {
            'density': self.density_activation(self.density_head(features)),
            'feature': self.color_head(color_input),
        }


# (Q4.2) Separate coarse and fine NeRFs for hierarchical sampling
class NeuralRadianceFieldCoarseFine(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.coarse = NeuralRadianceField(cfg)
        self.fine = NeuralRadianceField(cfg)

    def forward(self, ray_bundle):
        return self.fine(ray_bundle)


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q4.3): Implement Neural Surface MLP to output per-point SDF
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        self.mlp_distance = MLPWithInputSkips(
            cfg.n_layers_distance,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_distance,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_distance,
            input_skips=cfg.append_distance,
        )
        self.distance_head = torch.nn.Linear(cfg.n_hidden_neurons_distance, 1)
        self._geometric_init(embedding_dim_xyz, cfg.get('init_radius', 1.0))

        # TODO (Q4.3): Implement Neural Surface MLP to output per-point color
        self.mlp_color = MLPWithInputSkips(
            cfg.n_layers_color,
            cfg.n_hidden_neurons_distance,
            3,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_color,
            input_skips=cfg.append_color,
        )

    def _geometric_init(self, embedding_dim, radius):
        # Initialize the distance network to approximate the SDF of a sphere (Atzmon & Lipman, SAL / IDR).
        # Only the raw xyz part of the embedding (its last 3 entries) is used at initialization.
        for li, layer in enumerate(self.mlp_distance.mlp):
            linear = layer[0] if isinstance(layer, torch.nn.Sequential) else layer
            torch.nn.init.normal_(linear.weight, 0.0, (2.0 / linear.out_features) ** 0.5)
            torch.nn.init.zeros_(linear.bias)

            if li == 0:
                torch.nn.init.zeros_(linear.weight[:, :embedding_dim - 3])
            elif li in self.mlp_distance._input_skips:
                torch.nn.init.zeros_(linear.weight[:, -embedding_dim:-3])

        n_hidden = self.distance_head.in_features
        torch.nn.init.normal_(self.distance_head.weight, (math.pi / n_hidden) ** 0.5, 1e-4)
        torch.nn.init.constant_(self.distance_head.bias, -radius)

    def _get_features(self, points):
        embedding = self.harmonic_embedding_xyz(points)
        return torch.relu(self.mlp_distance(embedding, embedding)), embedding

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q4.3
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        features, _ = self._get_features(points)
        return self.distance_head(features)
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q4.3
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        return self.get_distance_color(points)[1]
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q4.3
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points = points.view(-1, 3)
        features, embedding = self._get_features(points)
        distance = self.distance_head(features)
        color = torch.sigmoid(self.mlp_color(features, embedding))
        return distance, color
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'nerf_coarse_fine': NeuralRadianceFieldCoarseFine,
    'neural_surface': NeuralSurface,
}
