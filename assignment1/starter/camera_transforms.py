"""
Usage:
    python -m starter.camera_transforms --image_size 512

Finds the relative camera transforms (R_relative, T_relative) that reproduce the
four target images. New extrinsics: R = R_relative @ R_0, T = R_relative @ T_0 + T_relative.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


# Rotations about each axis by 90 degrees.
Rz = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]   # in-plane (roll): rotate the image clockwise
Ry = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]   # look at the cow from its side

# (name, R_relative, T_relative)
TRANSFORMS = {
    # transform1: pure roll about the view axis -> cow appears rotated 90 in-plane.
    "transform1": (Rz, [0, 0, 0]),
    # transform2: rotate 90 about up (y) axis to view the side, with a translation
    # that keeps the cow centered 3 units in front (T ends up back at [0,0,3]).
    "transform2": (Ry, [-3, 0, 3]),
    # transform3: move the camera straight back -> the cow looks smaller / farther.
    "transform3": ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 3]),
    # transform4: slide the camera in-plane -> the cow shifts left and down.
    "transform4": ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.4, -0.4, 0]),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    device = get_device()
    for name, (R_rel, T_rel) in TRANSFORMS.items():
        image = render_textured_cow(
            cow_path=args.cow_path,
            image_size=args.image_size,
            R_relative=R_rel,
            T_relative=T_rel,
            device=device,
        )
        out = "output/%s.jpg" % name
        plt.imsave(out, np.clip(image, 0, 1))
        print("saved", out)
