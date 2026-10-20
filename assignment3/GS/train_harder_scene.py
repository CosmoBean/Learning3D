import os
import math
import torch
import torch.nn.functional as F
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad_(True)
    gaussians.pre_act_scales.requires_grad_(True)
    gaussians.colours.requires_grad_(True)
    gaussians.pre_act_opacities.requires_grad_(True)
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad_(True)

def setup_optimizer(gaussians, baseline=False):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    if baseline:
        # Same setup as question 6 (train.py)
        parameters = [
            {'params': [gaussians.pre_act_opacities], 'lr': 0.01, "name": "opacities"},
            {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
            {'params': [gaussians.colours], 'lr': 0.01, "name": "colours"},
            {'params': [gaussians.means], 'lr': 0.0005, "name": "means"},
        ]
    else:
        parameters = [
            {'params': [gaussians.pre_act_opacities], 'lr': LR_OPACITIES, "name": "opacities"},
            {'params': [gaussians.pre_act_scales], 'lr': LR_SCALES, "name": "scales"},
            {'params': [gaussians.colours], 'lr': LR_COLOURS, "name": "colours"},
            {'params': [gaussians.means], 'lr': LR_MEANS_INIT, "name": "means"},
            {'params': [gaussians.pre_act_quats], 'lr': LR_QUATS, "name": "quats"},
        ]
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

    return optimizer

# Learning rates for the improved method
LR_OPACITIES = 0.05
LR_SCALES = 0.005
LR_COLOURS = 0.01
LR_QUATS = 0.001
LR_MEANS_INIT = 0.0016
LR_MEANS_FINAL = 0.000016

def means_lr(itr, num_itrs):
    # Exponential (log-linear) decay of the means learning rate
    t = min(itr / max(num_itrs, 1), 1.0)
    return math.exp((1 - t) * math.log(LR_MEANS_INIT) + t * math.log(LR_MEANS_FINAL))

def compute_ssim(pred, gt, window_size=11):
    # Differentiable SSIM with a gaussian window; pred, gt: (H, W, 3) in [0, 1]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(3, 1, window_size, window_size).contiguous()

    x = pred.permute(2, 0, 1)[None]
    y = gt.permute(2, 0, 1)[None]
    blur = lambda z: F.conv2d(z, window, padding=window_size // 2, groups=3)

    mu_x, mu_y = blur(x), blur(y)
    sigma_x = blur(x * x) - mu_x ** 2
    sigma_y = blur(y * y) - mu_y ** 2
    sigma_xy = blur(x * y) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean()

def init_random_in_ball(gaussians, num_points, radius):
    # Spread the random initial means uniformly over a ball covering the visible scene
    dirs = F.normalize(torch.randn((num_points, 3)), dim=-1)
    r = radius * torch.rand((num_points, 1)) ** (1.0 / 3.0)
    gaussians.means = (dirs * r).to(gaussians.means.device)
    gaussians.pre_act_scales = torch.full_like(gaussians.pre_act_scales, math.log(0.02))
    gaussians.pre_act_opacities = torch.full_like(gaussians.pre_act_opacities, -2.0)  # sigmoid ~ 0.12

def densify_and_prune(gaussians, grad_accum, grad_count, max_gaussians, grad_thresh, min_opacity):
    # Simplified adaptive density control: clone / split gaussians with large positional
    # gradients and prune nearly transparent ones.
    with torch.no_grad():
        means = gaussians.means.detach()
        scales = gaussians.pre_act_scales.detach()
        quats = gaussians.pre_act_quats.detach()
        colours = gaussians.colours.detach()
        opacities = gaussians.pre_act_opacities.detach()

        avg_grad = grad_accum / grad_count.clamp(min=1)
        keep = torch.sigmoid(opacities) > min_opacity

        budget = max(max_gaussians - int(keep.sum()), 0)
        grow = (avg_grad > grad_thresh) & keep
        if int(grow.sum()) > budget:
            top = torch.topk(avg_grad * grow, budget).indices
            grow = torch.zeros_like(grow)
            grow[top] = True

        # Large gaussians are split into two smaller ones, small gaussians are cloned
        big = grow & (torch.exp(scales).max(dim=1).values > 0.02)
        small = grow & ~big

        new_means = [means[keep], means[small]]
        new_scales = [scales[keep], scales[small]]
        if big.any():
            std = torch.exp(scales[big])
            if std.shape[1] == 1:
                std = std.repeat(1, 3)
            new_means.append(means[big] + torch.randn_like(std) * std)
            new_scales.append(scales[big] - math.log(1.6))
            scales = scales.clone()
            scales[big] -= math.log(1.6)
            new_scales[0] = scales[keep]
        idx = torch.cat([torch.nonzero(keep)[:, 0], torch.nonzero(small)[:, 0], torch.nonzero(big)[:, 0]])

        gaussians.means = torch.cat(new_means).contiguous()
        gaussians.pre_act_scales = torch.cat(new_scales).contiguous()
        gaussians.pre_act_quats = quats[idx].contiguous()
        gaussians.colours = colours[idx].contiguous()
        gaussians.pre_act_opacities = opacities[idx].contiguous()

def reset_opacities(gaussians, max_opacity=0.01):
    # Periodically lower opacities so that floaters can be pruned
    with torch.no_grad():
        cap = math.log(max_opacity / (1 - max_opacity))
        gaussians.pre_act_opacities = torch.minimum(
            gaussians.pre_act_opacities.detach(), torch.full_like(gaussians.pre_act_opacities, cap)
        )

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    suffix = "_baseline" if args.baseline else ""
    viz_gif_path_1 = os.path.join(args.out_path, f"part_7_2_training_progress{suffix}.gif")
    viz_gif_path_2 = os.path.join(args.out_path, f"part_7_2_training_final_renders{suffix}.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).cuda() for i in viz_idxs]

    # Init gaussians and scene
    if args.baseline:
        gaussians = Gaussians(
            num_points=10000, init_type="random",
            device=args.device, isotropic=True
        )
    else:
        gaussians = Gaussians(
            num_points=args.num_init_points, init_type="random",
            device=args.device, isotropic=False
        )
        init_random_in_ball(gaussians, args.num_init_points, radius=1.3)
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians, baseline=args.baseline)

    densify_from, densify_until = int(0.05 * args.num_itrs), int(0.6 * args.num_itrs)
    grad_accum = torch.zeros(len(gaussians), device=args.device)
    grad_count = torch.zeros(len(gaussians), device=args.device)

    # Training loop
    viz_frames = []
    for itr in range(args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        camera = ndc_to_screen_camera(data[0]["camera"]).cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Set img_size to (128, 128)
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img, _, _ = scene.render(
            camera, per_splat=args.gaussians_per_splat,
            img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0),
        )

        # Compute loss
        ### YOUR CODE HERE ###
        if args.baseline:
            loss = F.l1_loss(pred_img, gt_img)
        else:
            loss = 0.8 * F.l1_loss(pred_img, gt_img) + 0.2 * (1.0 - compute_ssim(pred_img, gt_img))

        loss.backward()

        if not args.baseline:
            for group in optimizer.param_groups:
                if group["name"] == "means":
                    group["lr"] = means_lr(itr, args.num_itrs)
            grad_accum += gaussians.means.grad.norm(dim=-1)
            grad_count += 1

        optimizer.step()
        optimizer.zero_grad()

        if not args.baseline and densify_from <= itr < densify_until:
            if itr % 100 == 0 and itr > densify_from:
                densify_and_prune(
                    gaussians, grad_accum, grad_count, args.max_gaussians,
                    grad_thresh=args.densify_grad_thresh, min_opacity=0.005,
                )
                if itr % 1000 == 0:
                    reset_opacities(gaussians)
                make_trainable(gaussians)
                optimizer = setup_optimizer(gaussians)
                grad_accum = torch.zeros(len(gaussians), device=args.device)
                grad_count = torch.zeros(len(gaussians), device=args.device)
                print(f"[*] Itr: {itr:07d} | Num gaussians: {len(gaussians)}")

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, (128, 128)
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera, per_splat=args.gaussians_per_splat,
                img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0),
            )

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera, per_splat=args.gaussians_per_splat,
                img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0),
            )

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Use the training setup from question 6 (isotropic, L1 loss, constant learning rates)."
    )
    parser.add_argument("--num_init_points", default=10000, type=int)
    parser.add_argument("--max_gaussians", default=30000, type=int)
    parser.add_argument("--densify_grad_thresh", default=2e-4, type=float)
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
