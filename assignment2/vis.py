import imageio
import mcubes
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch3d.renderer import (AlphaCompositor, FoVPerspectiveCameras, HardPhongShader, MeshRasterizer, MeshRenderer,
                                PointLights, PointsRasterizationSettings, PointsRasterizer, PointsRenderer,
                                RasterizationSettings, TexturesVertex, look_at_view_transform)
from pytorch3d.structures import Meshes, Pointclouds

import utils_vox


def mesh(verts, faces, color=(0.7, 0.7, 1.0)):
    verts = torch.as_tensor(verts).float()
    tex = TexturesVertex([torch.tensor(color, device=verts.device).expand_as(verts)])
    return Meshes([verts], [torch.as_tensor(faces).long().to(verts.device)], textures=tex)


def points(pts, color=(0.2, 0.4, 0.9)):
    """color: one rgb tuple, or an N x 3 tensor of per-point colors."""
    pts = pts.detach().reshape(-1, 3).float()
    color = torch.as_tensor(color, device=pts.device).float()
    return Pointclouds([pts], features=[color.expand_as(pts)])


def vox_mesh(vox, thresh=0.5):
    """Voxel grid (... x 32 x 32 x 32 occupancy) -> mesh in the GT mesh frame (same transform as eval_model)."""
    verts, faces = mcubes.marching_cubes(vox.detach().squeeze().cpu().numpy(), thresh)
    verts = utils_vox.Mem2Ref(torch.tensor(verts).float()[None], *vox.shape[-3:])[0]
    return mesh(verts * torch.tensor([-1.0, 1.0, -1.0]), faces.astype(np.int64))  # rotate pi about y


def render(obj, n=1, size=256, dist=1.3, elev=20.0, azim=30.0, device="cuda"):
    """Render one Meshes/Pointclouds from n azimuths around it -> n x size x size x 3 uint8."""
    R, T = look_at_view_transform(dist, elev, azim + torch.arange(n) * 360.0 / n)
    cams = FoVPerspectiveCameras(R=R, T=T, device=device)
    obj = obj.detach().to(device).extend(n)
    if isinstance(obj, Pointclouds):
        renderer = PointsRenderer(PointsRasterizer(cams, PointsRasterizationSettings(image_size=size, radius=0.015)),
                                  AlphaCompositor(background_color=(1, 1, 1)))
    else:
        lights = PointLights(location=cams.get_camera_center(), device=device)
        renderer = MeshRenderer(MeshRasterizer(cams, RasterizationSettings(image_size=size, bin_size=0)),
                                HardPhongShader(device, cams, lights))
    return (renderer(obj)[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def save_gif(path, *renders, titles=None):
    """Save 360-degree renders (each n x H x W x 3) side by side as a looping gif, optionally with a title per panel."""
    frames = np.concatenate(renders, axis=2)
    if titles:
        w = renders[0].shape[2]
        header = Image.new("RGB", (frames.shape[2], 32), "white")
        draw, font = ImageDraw.Draw(header), ImageFont.load_default(size=18)
        for i, t in enumerate(titles):
            draw.text((w * i + w / 2, 16), t, fill=(51, 51, 51), font=font, anchor="mm")
        frames = np.concatenate([np.repeat(np.asarray(header)[None], len(frames), 0), frames], axis=1)
    imageio.mimsave(path, list(frames), duration=60, loop=0)
