
"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import argparse
import itertools
import os
import pathlib
import time
from typing import Callable

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from plyfile import PlyData, PlyElement
from radiance_fields.ngp import NGPDensityField, NGPRadianceField
from datasets.utils import Rays

from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    DL3DV_SCENES,
    render_image_with_propnet,
    set_random_seed,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="140",
    choices=DL3DV_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)


from datasets.nerf_360_v2 import SubjectLoader

# training parameters
max_steps = 5_000
init_batch_size = 4096
weight_decay = 0.0
# scene parameters
unbounded = True
aabb = torch.tensor([-1, -1, -1, 1, 1, 1], device=device) # dl3dv: 1
near_plane = 0.1 # dl3dv: 0.1
far_plane = 30 
# dataset parameters
train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 1}
# model parameters
proposal_networks = [
    NGPDensityField(
        aabb=aabb,
        unbounded=unbounded,
        n_levels=5,
        max_resolution=128,
    ).to(device),
    NGPDensityField(
        aabb=aabb,
        unbounded=unbounded,
        n_levels=5,
        max_resolution=256,
    ).to(device),
]
# render parameters
num_samples = 48
num_samples_per_prop = [256, 96]
sampling_type = "lindisp"
opaque_bkgd = True

proposal_networks = [
    NGPDensityField(
        aabb=aabb,
        unbounded=unbounded,
        n_levels=5,
        max_resolution=128,
    ).to(device),
    NGPDensityField(
        aabb=aabb,
        unbounded=unbounded,
        n_levels=5,
        max_resolution=256,
    ).to(device),
]

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

# setup the radiance field we want to train.
prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            prop_optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            prop_optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
proposal_requires_grad_fn = get_proposal_requires_grad_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    for p in proposal_networks:
        p.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras = render_image_with_propnet(
        radiance_field,
        proposal_networks,
        estimator,
        rays,
        # rendering options
        num_samples=num_samples,
        num_samples_per_prop=num_samples_per_prop,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=sampling_type,
        opaque_bkgd=opaque_bkgd,
        render_bkgd=render_bkgd,
        # train options
        proposal_requires_grad=proposal_requires_grad,
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        if step in [5_000]:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | psnr={psnr:.2f} | "
                f"num_rays={len(pixels):d} | "
                f"max_depth={depth.max():.3f} | "
            )

            os.makedirs(f'{args.output}/ckpt', exist_ok=True)
            
            checkpoint_path_radiance = f'{args.output}/ckpt/ckpt_rf_{step}.pth'
            checkpoint_path_estimator = f'{args.output}/ckpt/ckpt_est_{step}.pth'
            torch.save({
                'step': step,
                'model_state_dict': radiance_field.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path_radiance)
            torch.save({
                'step': step,
                'model_state_dict': estimator.state_dict(),
                'optimizer_state_dict': prop_optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path_estimator)
            
            for idx, p_net in enumerate(proposal_networks):
                checkpoint_path_p_net = f'{args.output}/ckpt/ckpt_net_{idx}_{step}.pth'
                torch.save({
                    'step': step,
                    'model_state_dict': p_net.state_dict(),}, checkpoint_path_p_net)

            # checkpoint_path_radiance = f'{args.output}/ckpt/ckpt_rf_{step}.pth'
            # radiance_field.load_state_dict(torch.load(checkpoint_path_radiance)['model_state_dict'])

            # checkpoint_path_estimator = f'{args.output}/ckpt/ckpt_est_{step}.pth'
            # estimator.load_state_dict(torch.load(checkpoint_path_estimator)['model_state_dict'])

            # for idx, p_net in enumerate(proposal_networks):
            #     checkpoint_path_p_net = f'{args.output}/ckpt/ckpt_net_{idx}_{step}.pth'
            #     p_net.load_state_dict(torch.load(checkpoint_path_p_net)['model_state_dict'])

            train_dataset_frames = SubjectLoader(
                subject_id=args.scene,
                root_fp=args.data_root,
                split=args.train_split,
                num_rays=None,
                device=device,
                **train_dataset_kwargs,
            )

            radiance_field.eval()
            for p in proposal_networks:
                p.eval()
            estimator.eval()

            bkgd = torch.ones(3).to(device)

            T = train_dataset_frames.T
            s = train_dataset_frames.sscale

            translate = T[:3, 3]
            rotate = T[:3, :3]

            num_samples = 2

            all_points = []
            all_rgbs = []
            all_depths = []

            total_points = 50_000
            points_per_image = total_points // len(train_dataset_frames)

            for i in tqdm.trange(len(train_dataset_frames)):
                rays = train_dataset_frames[i]["rays"]
                rays_orig = rays
                n_rays = rays.origins.shape[0] * rays.origins.shape[1]
                indices = torch.randint(0, n_rays, (points_per_image,))

                origins = rays.origins.reshape(-1, 3)[indices]
                viewdirs = rays.viewdirs.reshape(-1, 3)[indices]
                rays = Rays(origins, viewdirs)

                (rgb, acc, depth, _,) = render_image_with_propnet(
                    radiance_field,
                    proposal_networks,
                    estimator,
                    rays=rays,
                    # rendering options
                    num_samples=num_samples,
                    num_samples_per_prop=num_samples_per_prop,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    sampling_type=sampling_type,
                    opaque_bkgd=opaque_bkgd,
                    render_bkgd=bkgd,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                    force_stratified=True,
                )

                points = (rays.origins + rays.viewdirs * depth).reshape(-1, 3)

                points = points.cpu().numpy()
                points = (rotate.T @ (points / s - translate).T).T

                all_points.append(points)
                all_rgbs.append(rgb.reshape(-1, 3).cpu().numpy())

                (_, _, depth, _,) = render_image_with_propnet(
                    radiance_field,
                    proposal_networks,
                    estimator,
                    rays=rays_orig,
                    # rendering options
                    num_samples=48,
                    num_samples_per_prop=num_samples_per_prop,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    sampling_type=sampling_type,
                    opaque_bkgd=opaque_bkgd,
                    render_bkgd=bkgd,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                    force_stratified=True,
                )

                all_depths.append(depth)

            torch.save(all_depths, f'{args.output}/{args.scene}_depths_{step}.pt')

            with open(f'{args.output}/scale.txt', "w") as fd:
                fd.writelines([f'{s}'])

            all_points = np.concatenate(all_points, axis=0)
            all_rgbs = np.concatenate(all_rgbs, axis=0) * 255.0

            def storePly(path, xyz, rgb):
                # Define the dtype for the structured array
                dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
                
                normals = np.zeros_like(xyz)

                elements = np.empty(xyz.shape[0], dtype=dtype)
                attributes = np.concatenate((xyz, normals, rgb), axis=1)
                elements[:] = list(map(tuple, attributes))

                # Create the PlyData object and write to file
                vertex_element = PlyElement.describe(elements, 'vertex')
                ply_data = PlyData([vertex_element])
                ply_data.write(path)

            storePly(f'{args.output}/nerfacc_{step}.ply', all_points, all_rgbs)

            quit()