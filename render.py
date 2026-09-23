#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import numpy as np
from utils.vis_utils import apply_depth_colormap
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

DEPTH_OFFSET = 6
ALPHA_OFFSET = 7
MAX_DEPTH_RANGE_SAMPLES_PER_VIEW = 65_536
MAX_DEPTH_RANGE_SAMPLES = 2_000_000


def uniformly_subsample(values, max_samples):
    if values.numel() <= max_samples:
        return values
    indices = torch.linspace(
        0,
        values.numel() - 1,
        steps=max_samples,
        device=values.device,
    ).long()
    return values[indices]


def compute_depth_range(views, gaussians, pipeline, background, kernel_size, alpha_threshold, low_quantile, high_quantile):
    samples = []
    for view in tqdm(views, desc="Depth range pass"):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        depth = rendering[DEPTH_OFFSET]
        alpha = rendering[ALPHA_OFFSET]
        valid = (alpha > alpha_threshold) & torch.isfinite(depth) & (depth > 0)
        if valid.any():
            valid_depth = uniformly_subsample(
                depth[valid].detach(),
                MAX_DEPTH_RANGE_SAMPLES_PER_VIEW,
            )
            samples.append(valid_depth.cpu())

    if not samples:
        return None, None

    depths = uniformly_subsample(torch.cat(samples), MAX_DEPTH_RANGE_SAMPLES)
    near = torch.quantile(depths, low_quantile).item()
    far = torch.quantile(depths, high_quantile).item()
    if near >= far:
        near = depths.min().item()
        far = depths.max().item()
    return near, far


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    kernel_size,
    scale_factor,
    save_depth,
    depth_global_norm,
    depth_q_low,
    depth_q_high,
    depth_alpha_thresh,
    depth_vis_blend_alpha,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")
    depth_vis_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"depth_vis_{scale_factor}")
    depth_raw_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"depth_raw_{scale_factor}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if save_depth:
        makedirs(depth_vis_path, exist_ok=True)
        makedirs(depth_raw_path, exist_ok=True)

    global_near = None
    global_far = None
    poses = []
    if save_depth and depth_global_norm:
        global_near, global_far = compute_depth_range(
            views,
            gaussians,
            pipeline,
            background,
            kernel_size,
            depth_alpha_thresh,
            depth_q_low,
            depth_q_high,
        )

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_full = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        rendering = rendering_full[:3]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if save_depth:
            depth = rendering_full[DEPTH_OFFSET]
            alpha = rendering_full[ALPHA_OFFSET]
            valid = (alpha > depth_alpha_thresh) & torch.isfinite(depth) & (depth > 0)

            if global_near is not None and global_far is not None:
                near, far = global_near, global_far
            elif valid.any():
                near = depth[valid].min().item()
                far = depth[valid].max().item()
            else:
                near = depth.min().item()
                far = depth.max().item()

            depth_color = apply_depth_colormap(
                depth[..., None],
                alpha[..., None] if depth_vis_blend_alpha else None,
                near_plane=near,
                far_plane=far,
            ).permute(2, 0, 1)
            stem = "{0:05d}".format(idx)
            torchvision.utils.save_image(depth_color, os.path.join(depth_vis_path, stem + ".png"))
            np.save(os.path.join(depth_raw_path, stem + ".npy"), depth.detach().cpu().numpy())
            poses.append(view.world_view_transform.T.inverse().detach().cpu().numpy())

    if poses:
        poses = np.stack(poses)
        np.save(os.path.join(depth_raw_path, "poses.npy"), poses)
        np.savetxt(os.path.join(depth_raw_path, "poses.txt"), poses.reshape(len(poses), -1), fmt="%.8f")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    save_depth: bool,
    depth_global_norm: bool,
    depth_q_low: float,
    depth_q_high: float,
    depth_alpha_thresh: float,
    depth_vis_blend_alpha: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor, save_depth, depth_global_norm, depth_q_low, depth_q_high, depth_alpha_thresh, depth_vis_blend_alpha)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor, save_depth, depth_global_norm, depth_q_low, depth_q_high, depth_alpha_thresh, depth_vis_blend_alpha)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_depth", dest="save_depth", action="store_true")
    parser.add_argument("--no_save_depth", dest="save_depth", action="store_false")
    parser.add_argument("--depth_global_norm", action="store_true")
    parser.add_argument("--depth_q_low", type=float, default=0.01)
    parser.add_argument("--depth_q_high", type=float, default=0.99)
    parser.add_argument("--depth_alpha_thresh", type=float, default=0.05)
    parser.add_argument("--depth_vis_blend_alpha", action="store_true")
    parser.add_argument("--no_depth_vis_blend_alpha", dest="depth_vis_blend_alpha", action="store_false")
    parser.set_defaults(save_depth=True, depth_vis_blend_alpha=True)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.save_depth,
        args.depth_global_norm,
        args.depth_q_low,
        args.depth_q_high,
        args.depth_alpha_thresh,
        args.depth_vis_blend_alpha,
    )
