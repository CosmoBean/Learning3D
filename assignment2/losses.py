import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d (occupancy logits)
	# voxel_tgt: b x h x w x d
	loss = F.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3
	# squared distance to the nearest neighbour, in both directions
	d_src = knn_points(point_cloud_src, point_cloud_tgt, K=1).dists[..., 0]
	d_tgt = knn_points(point_cloud_tgt, point_cloud_src, K=1).dists[..., 0]
	loss_chamfer = d_src.mean() + d_tgt.mean()
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	return loss_laplacian
