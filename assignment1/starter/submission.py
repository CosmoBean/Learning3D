"""
Solutions for 16-825 Assignment 1 (base questions).

Usage:
    python -m starter.submission --question 1.1   # 360 cow gif
    python -m starter.submission --question 2.1   # tetrahedron
    python -m starter.submission --question 2.2   # cube
    python -m starter.submission --question 3     # retextured cow
    python -m starter.submission --question 5.1   # rgb-d point clouds
    python -m starter.submission --question 5.2   # torus + hyperboloid point clouds
    python -m starter.submission --question 5.3   # torus + hollow-cube implicit meshes
    python -m starter.submission --question 6     # rainbow grid cow
    python -m starter.submission --question 7     # sampling points on the cow mesh
    python -m starter.submission --question all   # run everything
"""
import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.render_generic import load_rgbd_data
from starter.utils import (
    get_device,
    get_mesh_renderer,
    get_points_renderer,
    load_cow_mesh,
    unproject_depth_image,
)


def save_gif(images, path, fps=15):
    """Save a list of float (H,W,3) images in [0,1] as a looping gif."""
    frames = [(np.clip(im, 0, 1) * 255).astype(np.uint8) for im in images]
    imageio.mimsave(path, frames, duration=1000 // fps, loop=0)
    print("saved", path)


def turntable_cameras(num_frames=36, dist=3, device=None):
    """A set of cameras spanning 360 degrees of azimuth."""
    azims = torch.linspace(0, 360, num_frames)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, azim=azims)
    return pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)


def render_mesh_turntable(mesh, image_size=256, dist=3, num_frames=36, device=None):
    """Render a mesh from many azimuths and return a list of images."""
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    cameras = turntable_cameras(num_frames, dist, device=device)
    images = []
    for i in range(num_frames):
        cam = cameras[i]
        rend = renderer(mesh, cameras=cam, lights=lights)
        images.append(rend[0, ..., :3].cpu().numpy())
    return images


def render_points_turntable(point_cloud, image_size=256, dist=6, num_frames=36, device=None):
    """Render a point cloud from many azimuths and return a list of images."""
    renderer = get_points_renderer(image_size=image_size, device=device)
    cameras = turntable_cameras(num_frames, dist, device=device)
    images = []
    for i in range(num_frames):
        rend = renderer(point_cloud, cameras=cameras[i])
        images.append(rend[0, ..., :3].cpu().numpy())
    return images


def single_color_mesh(vertices, faces, color, device):
    """Build a Meshes with one uniform vertex color."""
    verts = vertices.unsqueeze(0)
    textures = torch.ones_like(verts) * torch.tensor(color)
    return pytorch3d.structures.Meshes(
        verts=verts,
        faces=faces.unsqueeze(0),
        textures=pytorch3d.renderer.TexturesVertex(textures),
    ).to(device)


# ---------------------------------------------------------------------------
# 1.1 360-degree render of the cow
# ---------------------------------------------------------------------------
def q1_1(device):
    vertices, faces = load_cow_mesh("data/cow.obj")
    mesh = single_color_mesh(vertices, faces, [0.7, 0.7, 1.0], device)
    save_gif(render_mesh_turntable(mesh, device=device), "output/cow_360.gif")


# ---------------------------------------------------------------------------
# 2.1 Tetrahedron: 4 vertices, 4 triangle faces
# ---------------------------------------------------------------------------
def q2_1(device):
    vertices = torch.tensor(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    mesh = single_color_mesh(vertices, faces, [0.2, 0.8, 0.9], device)
    save_gif(render_mesh_turntable(mesh, dist=5, device=device), "output/tetrahedron_360.gif")
    print("tetrahedron: %d vertices, %d faces" % (len(vertices), len(faces)))


# ---------------------------------------------------------------------------
# 2.2 Cube: 8 vertices, 12 triangle faces (2 per square side)
# ---------------------------------------------------------------------------
def q2_2(device):
    vertices = torch.tensor(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2], [0, 2, 3],  # back
            [4, 6, 5], [4, 7, 6],  # front
            [0, 4, 5], [0, 5, 1],  # bottom
            [3, 2, 6], [3, 6, 7],  # top
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ]
    )
    mesh = single_color_mesh(vertices, faces, [0.9, 0.6, 0.2], device)
    save_gif(render_mesh_turntable(mesh, dist=5, device=device), "output/cube_360.gif")
    print("cube: %d vertices, %d faces" % (len(vertices), len(faces)))


# ---------------------------------------------------------------------------
# 3 Re-texture the cow with a z-based color gradient
# ---------------------------------------------------------------------------
def q3(device):
    color1 = torch.tensor([0.0, 0.0, 1.0])  # front (z_min): blue
    color2 = torch.tensor([1.0, 0.0, 0.0])  # back  (z_max): red
    vertices, faces = load_cow_mesh("data/cow.obj")
    z = vertices[:, 2]
    alpha = ((z - z.min()) / (z.max() - z.min())).unsqueeze(1)
    colors = alpha * color2 + (1 - alpha) * color1  # (N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        textures=pytorch3d.renderer.TexturesVertex(colors.unsqueeze(0)),
    ).to(device)
    save_gif(render_mesh_turntable(mesh, device=device), "output/cow_retextured_360.gif")
    print("color1 (front) = [0,0,1] blue, color2 (back) = [1,0,0] red")


# ---------------------------------------------------------------------------
# 5.1 Point clouds from RGB-D images
# ---------------------------------------------------------------------------
def q5_1(device):
    data = load_rgbd_data()
    clouds = []
    for i in (1, 2):
        rgb = torch.tensor(data["rgb%d" % i]).float()
        mask = torch.tensor(data["mask%d" % i]).float()
        depth = torch.tensor(data["depth%d" % i]).float()
        cam = data["cameras%d" % i].to(device)
        pts, rgba = unproject_depth_image(rgb, mask, depth, cam)
        clouds.append((pts[::4], rgba[::4, :3]))  # subsample for fast CPU rendering

    # The CO 3D cameras are y-down; rotate 180 about z so the plant is upright.
    flip = torch.tensor([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]], device=device)

    def make(points, rgb):
        points = (flip @ points.to(device).T).T
        return pytorch3d.structures.Pointclouds(
            points=[points], features=[rgb.to(device)]
        )

    pc1 = make(*clouds[0])
    pc2 = make(*clouds[1])
    pc_union = make(
        torch.cat([clouds[0][0], clouds[1][0]]),
        torch.cat([clouds[0][1], clouds[1][1]]),
    )

    frames = []
    for imgs in zip(
        render_points_turntable(pc1, dist=7, device=device),
        render_points_turntable(pc2, dist=7, device=device),
        render_points_turntable(pc_union, dist=7, device=device),
    ):
        frames.append(np.concatenate(imgs, axis=1))
    save_gif(frames, "output/rgbd_pointclouds.gif")


# ---------------------------------------------------------------------------
# 5.2 Parametric point clouds: torus + hyperboloid
# ---------------------------------------------------------------------------
def parametric_cloud(x, y, z, device):
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    return pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(device)


def q5_2(device):
    n = 150
    u = torch.linspace(0, 2 * np.pi, n)
    v = torch.linspace(0, 2 * np.pi, n)
    U, V = torch.meshgrid(u, v)

    # Torus (R major radius, r minor radius), hole clearly visible.
    R, r = 1.0, 0.4
    x = (R + r * torch.cos(V)) * torch.cos(U)
    y = (R + r * torch.cos(V)) * torch.sin(U)
    z = r * torch.sin(V)
    torus = parametric_cloud(x, y, z, device)
    save_gif(render_points_turntable(torus, dist=4, device=device), "output/torus_points_360.gif")

    # Hyperboloid of one sheet (an hourglass), the "new object".
    t = torch.linspace(-1, 1, n)
    theta = torch.linspace(0, 2 * np.pi, n)
    T, Th = torch.meshgrid(t, theta)
    rad = torch.sqrt(1 + T ** 2)
    x = rad * torch.cos(Th)
    y = rad * torch.sin(Th)
    z = T
    hyper = parametric_cloud(x, y, z, device)
    save_gif(render_points_turntable(hyper, dist=5, device=device), "output/hyperboloid_points_360.gif")


# ---------------------------------------------------------------------------
# 5.3 Implicit surfaces (marching cubes): torus + hollow cube
# ---------------------------------------------------------------------------
def implicit_mesh(voxels, min_v, max_v, voxel_size, device):
    import mcubes

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels.numpy()), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices / voxel_size) * (max_v - min_v) + min_v
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    return pytorch3d.structures.Meshes(
        [vertices], [faces], textures=pytorch3d.renderer.TexturesVertex([textures])
    ).to(device)


def q5_3(device):
    voxel_size = 64

    # Torus implicit surface.
    min_v, max_v = -1.5, 1.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_v, max_v, voxel_size)] * 3)
    R, r = 1.0, 0.4
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2 - r ** 2
    torus = implicit_mesh(voxels, min_v, max_v, voxel_size, device)
    save_gif(render_mesh_turntable(torus, dist=4, device=device), "output/torus_implicit_360.gif")

    # Hollow cube frame (the "new object"): keep only the 12 edges, so it is see-through.
    min_v, max_v = -1.5, 1.5
    voxel_size = 96  # finer grid so the thin bars resolve cleanly
    X, Y, Z = torch.meshgrid([torch.linspace(min_v, max_v, voxel_size)] * 3)
    a, w = 1.0, 0.25  # cube half-extent and bar thickness
    inside = (X.abs() <= a) & (Y.abs() <= a) & (Z.abs() <= a)
    near = (X.abs() >= a - w).int() + (Y.abs() >= a - w).int() + (Z.abs() >= a - w).int()
    material = inside & (near >= 2)  # on an edge = near at least two faces
    voxels = torch.where(material, -1.0, 1.0)  # negative inside the bars
    cube = implicit_mesh(voxels, min_v, max_v, voxel_size, device)
    save_gif(render_mesh_turntable(cube, dist=4, device=device), "output/hollow_cube_implicit_360.gif")


# ---------------------------------------------------------------------------
# 6 Do something fun: a rainbow grid cow (each grid cell a random color)
# ---------------------------------------------------------------------------
def make_random_grid(size=512, cells=16):
    """An image split into a grid of cells, each painted a random color."""
    block = size // cells
    grid = np.random.rand(cells, cells, 3).astype(np.float32)   # one random color per cell
    img = np.repeat(np.repeat(grid, block, axis=0), block, axis=1)
    return torch.tensor(img)


def q6(device):
    np.random.seed(0)  # keep the pattern reproducible
    verts, faces, aux = pytorch3d.io.load_obj("data/cow.obj")
    tex = pytorch3d.renderer.TexturesUV(
        maps=[make_random_grid().to(device)],
        faces_uvs=[faces.textures_idx.to(device)],
        verts_uvs=[aux.verts_uvs.to(device)],
    )
    mesh = pytorch3d.structures.Meshes([verts], [faces.verts_idx], textures=tex).to(device)
    save_gif(render_mesh_turntable(mesh, device=device), "output/cow_rainbow_grid_360.gif")


# ---------------------------------------------------------------------------
# 7 Sampling points on meshes (stratified / area-weighted)
# ---------------------------------------------------------------------------
def sample_points_on_mesh(vertices, faces, n):
    """Uniformly sample n points on a triangle mesh surface."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    # 1. pick faces with probability proportional to their area
    idx = torch.multinomial(areas, n, replacement=True)
    # 2. uniform barycentric coordinates (fold the unit square into a triangle)
    u, v = torch.rand(n), torch.rand(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    w = 1 - u - v
    # 3. combine
    pts = w[:, None] * v0[idx] + u[:, None] * v1[idx] + v[:, None] * v2[idx]
    return pts


def q7(device):
    vertices, faces = load_cow_mesh("data/cow.obj")
    color = [0.7, 0.7, 1.0]
    mesh = single_color_mesh(vertices, faces, color, device)
    mesh_frames = render_mesh_turntable(mesh, device=device)

    panels = [mesh_frames]
    for n in (10, 100, 1000, 10000):
        pts = sample_points_on_mesh(vertices, faces, n)
        rgb = torch.ones_like(pts) * torch.tensor(color)
        pc = pytorch3d.structures.Pointclouds(points=[pts], features=[rgb]).to(device)
        panels.append(render_points_turntable(pc, dist=3, device=device))

    frames = [np.concatenate(f, axis=1) for f in zip(*panels)]
    save_gif(frames, "output/mesh_sampling.gif")


QUESTIONS = {
    "1.1": q1_1,
    "2.1": q2_1,
    "2.2": q2_2,
    "3": q3,
    "5.1": q5_1,
    "5.2": q5_2,
    "5.3": q5_3,
    "6": q6,
    "7": q7,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True, choices=list(QUESTIONS) + ["all"])
    args = parser.parse_args()
    device = get_device()
    todo = QUESTIONS if args.question == "all" else {args.question: QUESTIONS[args.question]}
    for name, fn in todo.items():
        print("=== Question", name, "===")
        fn(device)
