import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import ball_query, knn_gather, knn_points, sample_farthest_points


def mlp(dims, d=1):
    """Shared MLP of 1x1 convs + BN + ReLU over (B, C, N) (d=1) or (B, C, N, K) (d=2) inputs."""
    conv, bn = (nn.Conv1d, nn.BatchNorm1d) if d == 1 else (nn.Conv2d, nn.BatchNorm2d)
    return nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in (conv(a, b, 1), bn(b), nn.ReLU())])


def cls_head(num_classes):
    return nn.Sequential(
        nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.feat = mlp([3, 64, 64, 64, 128, 1024])
        self.head = cls_head(num_classes)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        return self.head(self.feat(points.transpose(1, 2)).amax(2))


class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.local = mlp([3, 64, 64])
        self.glob = mlp([64, 64, 128, 1024])
        self.head = nn.Sequential(*mlp([1088, 512, 256, 128]), nn.Conv1d(128, num_seg_classes, 1))

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        x = self.local(points.transpose(1, 2))
        g = self.glob(x).amax(2, keepdim=True).expand(-1, -1, x.shape[2])
        return self.head(torch.cat([x, g], 1)).transpose(1, 2)


# ------ Q4: PointNet++ (single-scale grouping) ------
class SetAbstraction(nn.Module):
    """Sample centroids with FPS, group neighbors in a ball, and run a PointNet on each local region."""
    def __init__(self, npoint, radius, k, dims):
        super().__init__()
        self.npoint, self.radius, self.k = npoint, radius, k
        self.mlp = mlp(dims, d=2)

    def forward(self, xyz, feat):  # xyz (B, N, 3), feat (B, N, C) or None
        if self.npoint is None:  # group all points into one region
            new_xyz, grouped = xyz[:, :1] * 0, xyz.unsqueeze(1)
            if feat is not None:
                grouped = torch.cat([grouped, feat.unsqueeze(1)], -1)
        else:
            new_xyz, _ = sample_farthest_points(xyz, K=self.npoint)
            _, idx, _ = ball_query(new_xyz, xyz, K=self.k, radius=self.radius)
            idx = torch.where(idx < 0, idx[..., :1], idx)  # pad empty slots with the first neighbor
            grouped = knn_gather(xyz, idx) - new_xyz.unsqueeze(2)  # local coordinates
            if feat is not None:
                grouped = torch.cat([grouped, knn_gather(feat, idx)], -1)
        return new_xyz, self.mlp(grouped.permute(0, 3, 1, 2)).amax(3).transpose(1, 2)


class FeaturePropagation(nn.Module):
    """Interpolate coarse features back to the dense points (inverse distance, 3-NN) and fuse with skip features."""
    def __init__(self, dims):
        super().__init__()
        self.mlp = mlp(dims)

    def forward(self, xyz, sub_xyz, feat, sub_feat):
        if sub_xyz.shape[1] == 1:
            x = sub_feat.expand(-1, xyz.shape[1], -1)
        else:
            d, idx, _ = knn_points(xyz, sub_xyz, K=3)
            w = 1 / (d + 1e-8)
            x = (knn_gather(sub_feat, idx) * (w / w.sum(-1, keepdim=True)).unsqueeze(-1)).sum(2)
        if feat is not None:
            x = torch.cat([feat, x], -1)
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


def set_abstractions():
    return nn.ModuleList([
        SetAbstraction(512, 0.2, 32, [3, 64, 64, 128]),
        SetAbstraction(128, 0.4, 64, [128 + 3, 128, 128, 256]),
        SetAbstraction(None, None, None, [256 + 3, 256, 512, 1024]),
    ])


class cls_pp_model(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.sa = set_abstractions()
        self.head = cls_head(num_classes)

    def forward(self, points):
        xyz, feat = points, None
        for sa in self.sa:
            xyz, feat = sa(xyz, feat)
        return self.head(feat.squeeze(1))


class seg_pp_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super().__init__()
        self.sa = set_abstractions()
        self.fp = nn.ModuleList([FeaturePropagation([256 + 1024, 256, 256]),
                                 FeaturePropagation([128 + 256, 256, 128]),
                                 FeaturePropagation([128, 128, 128, 128])])
        self.head = nn.Sequential(*mlp([128, 128]), nn.Dropout(0.5), nn.Conv1d(128, num_seg_classes, 1))

    def forward(self, points):
        xyz, feat = [points], [None]
        for sa in self.sa:
            x, f = sa(xyz[-1], feat[-1])
            xyz.append(x)
            feat.append(f)
        f = feat[-1]
        for i, fp in zip([2, 1, 0], self.fp):
            f = fp(xyz[i], xyz[i + 1], feat[i], f)
        return self.head(f.transpose(1, 2)).transpose(1, 2)
