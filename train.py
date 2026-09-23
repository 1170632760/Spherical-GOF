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

import os
import numpy as np
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim_weighted
import torch.nn.functional as F
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.graphics_utils import equirectangular_rays

def L1_loss_appearance(image, gt_image, gaussians, view_idx):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]

    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]

    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    return l1_loss(transformed_image, crop_gt_image)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    min_scale_ratio = float(getattr(gaussians, "min_scale_ratio", 1e-4))
    gaussians.min_scale = max(float(getattr(gaussians, "min_scale", 1e-4)), scene.cameras_extent * min_scale_ratio)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    trainCameras = scene.getTrainCameras().copy()
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_rgb_for_densify = 0.0
    ray_cache = {}
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image = rendering[:3, :, :]

        # rgb Loss
        gt_image = viewpoint_cam.original_image.cuda()

        _, H_img, W_img = image.shape
        theta_min = float(getattr(viewpoint_cam, "pano_theta_min", -0.5 * np.pi))
        theta_max = float(getattr(viewpoint_cam, "pano_theta_max", 0.5 * np.pi))
        theta_span = max(theta_max - theta_min, 1e-6)
        h_idx = torch.linspace(0, H_img - 1, H_img, device=image.device)
        row_theta = theta_min + ((h_idx + 0.5) / max(H_img, 1)) * theta_span
        row_cos = torch.cos(row_theta)
        weights = row_cos.clamp_min(0.0).view(1, H_img, 1)
        horiz_metric = row_cos.abs().clamp_min(1e-4).view(H_img, 1)
        horiz_metric3 = horiz_metric.view(H_img, 1, 1)

        l1_diff = torch.abs(image - gt_image)
        Ll1 = (l1_diff * weights).sum() / (weights.expand_as(l1_diff).sum() + 1e-8)

        # use L1 loss for the transformed image if using decoupled appearance
        if dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)

        ssim_val = ssim_weighted(image.unsqueeze(0), gt_image.unsqueeze(0), weights.unsqueeze(0).expand(1, 3, H_img, W_img))
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        normal_loss = torch.tensor(0.0, device=image.device)
        depth_loss = torch.tensor(0.0, device=image.device)
        depth_lap_loss = torch.tensor(0.0, device=image.device)
        depth_normal_consistency_loss = torch.tensor(0.0, device=image.device)
        depth_jump_loss = torch.tensor(0.0, device=image.device)
        depth_jump2_loss = torch.tensor(0.0, device=image.device)

        if (
            opt.normal_smoothness_weight > 0
            or opt.depth_grad_weight > 0
            or opt.depth_laplacian_weight > 0
            or opt.depth_normal_consistency_weight > 0
            or opt.depth_jump_weight > 0
            or opt.depth_jump2_weight > 0
        ):
            area_w = weights

            depth = rendering[6, :, :]
            alpha = rendering[7, :, :].clamp(0.0, 1.0)

            if opt.normal_smoothness_weight > 0:
                normal = rendering[3:6, :, :]
                normal = F.normalize(normal, dim=0)
                n_xp = torch.roll(normal, shifts=-1, dims=2)
                n_yp = normal[:, 1:, :]

                alpha_xp = torch.roll(alpha, shifts=-1, dims=1)
                a_x = torch.minimum(alpha, alpha_xp)
                a_y = torch.minimum(alpha[1:, :], alpha[:-1, :])

                dot_x = (normal * n_xp).sum(0)
                dot_y = (normal[:, :-1, :] * n_yp).sum(0)
                diff_x = (1.0 - dot_x) * (a_x > opt.normal_alpha_thresh).float()
                diff_y = (1.0 - dot_y) * (a_y > opt.normal_alpha_thresh).float()

                w_x = area_w * a_x
                w_y = area_w[:, 1:, :] * a_y
                normal_loss = (diff_x * w_x).sum() / (w_x.sum() + 1e-6)
                normal_loss = normal_loss + (diff_y * w_y).sum() / (w_y.sum() + 1e-6)

            if opt.depth_grad_weight > 0:
                depth = depth.clamp_min(1e-4)
                log_depth = torch.log(depth)
                d_x = torch.roll(log_depth, shifts=-1, dims=1) - log_depth
                d_y = log_depth[1:, :] - log_depth[:-1, :]

                d_x = d_x / horiz_metric

                a_x = torch.minimum(alpha, torch.roll(alpha, shifts=-1, dims=1))
                a_y = torch.minimum(alpha[1:, :], alpha[:-1, :])

                w_x = area_w.squeeze(0) * a_x
                w_y = area_w.squeeze(0)[1:, :] * a_y

                eps = float(getattr(opt, "depth_robust_eps", 1e-3))
                d_x_robust = torch.sqrt(d_x * d_x + eps * eps)
                d_y_robust = torch.sqrt(d_y * d_y + eps * eps)
                depth_loss = (d_x_robust * (a_x > opt.depth_alpha_thresh).float() * w_x).sum() / (w_x.sum() + 1e-6)
                depth_loss = depth_loss + (d_y_robust * (a_y > opt.depth_alpha_thresh).float() * w_y).sum() / (w_y.sum() + 1e-6)

            if opt.depth_laplacian_weight > 0:
                depth = depth.clamp_min(1e-4)
                log_depth = torch.log(depth)
                dxx = torch.roll(log_depth, shifts=-1, dims=1) - 2.0 * log_depth + torch.roll(log_depth, shifts=1, dims=1)
                dyy = log_depth[2:, :] - 2.0 * log_depth[1:-1, :] + log_depth[:-2, :]

                inv_sin2 = (1.0 / horiz_metric) ** 2
                dxx = dxx * inv_sin2

                a_x = torch.minimum(alpha, torch.roll(alpha, shifts=-1, dims=1))
                a_x = torch.minimum(a_x, torch.roll(alpha, shifts=1, dims=1))
                a_y = torch.minimum(alpha[1:-1, :], alpha[:-2, :])
                a_y = torch.minimum(a_y, alpha[2:, :])

                w_x = area_w.squeeze(0) * a_x
                w_y = area_w.squeeze(0)[1:-1, :] * a_y

                eps = float(getattr(opt, "depth_robust_eps", 1e-3))
                dxx_robust = torch.sqrt(dxx * dxx + eps * eps)
                dyy_robust = torch.sqrt(dyy * dyy + eps * eps)
                depth_lap_loss = (dxx_robust * (a_x > opt.depth_alpha_thresh).float() * w_x).sum() / (w_x.sum() + 1e-6)
                depth_lap_loss = depth_lap_loss + (dyy_robust * (a_y > opt.depth_alpha_thresh).float() * w_y).sum() / (w_y.sum() + 1e-6)

            dn_start = int(opt.iterations * float(getattr(opt, "depth_normal_start_ratio", 0.55)))
            if opt.depth_normal_consistency_weight > 0 and iteration >= dn_start:
                depth = depth.clamp_min(1e-4)
                normal_pred = F.normalize(rendering[3:6, :, :], dim=0)
                cache_key = (H_img, W_img, str(image.device), round(theta_min, 7), round(theta_max, 7))
                if cache_key not in ray_cache:
                    xs = torch.arange(W_img, device=image.device).float()
                    ys = torch.arange(H_img, device=image.device).float()
                    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
                    rays = equirectangular_rays(
                        W_img,
                        H_img,
                        gx,
                        gy,
                        theta_min=theta_min,
                        theta_max=theta_max,
                    )
                    if rays.shape[0] == H_img and rays.shape[1] == W_img:
                        rays_hw = rays
                    elif rays.shape[0] == W_img and rays.shape[1] == H_img:
                        rays_hw = rays.permute(1, 0, 2).contiguous()
                    else:
                        raise RuntimeError(
                            f"Unexpected ray shape {tuple(rays.shape)} for H={H_img}, W={W_img}"
                        )
                    ray_cache[cache_key] = rays_hw

                rays_cam = ray_cache[cache_key]
                pts_cam = rays_cam * depth.unsqueeze(-1)
                pts_xp = torch.roll(pts_cam, shifts=-1, dims=1)
                pts_xm = torch.roll(pts_cam, shifts=1, dims=1)
                dpx = pts_xp - pts_xm
                pts_yp = pts_cam[2:, :, :]
                pts_ym = pts_cam[:-2, :, :]
                dpy = pts_yp - pts_ym
                dpx = dpx / horiz_metric3
                n_depth = torch.zeros_like(pts_cam)
                n_depth_mid = F.normalize(torch.cross(dpx[1:-1, :, :], dpy, dim=-1), dim=-1)
                n_depth[1:-1, :, :] = n_depth_mid
                n_depth = n_depth.permute(2, 0, 1).contiguous()

                valid_n = torch.zeros_like(alpha, dtype=torch.bool)
                valid_n[1:-1, :] = True
                m_n = valid_n & (alpha > opt.depth_alpha_thresh)
                cos_sim = torch.abs((normal_pred * n_depth).sum(dim=0))
                diff_n = (1.0 - cos_sim) * m_n.float()
                w_n = area_w.squeeze(0) * alpha
                depth_normal_consistency_loss = (diff_n * w_n).sum() / ((w_n * m_n.float()).sum() + 1e-6)

            if opt.depth_jump_weight > 0:
                depth = depth.clamp_min(1e-4)
                log_depth = torch.log(depth)
                d_x = torch.roll(log_depth, shifts=-1, dims=1) - log_depth
                d_y = log_depth[1:, :] - log_depth[:-1, :]

                d_x = d_x / horiz_metric

                gray = 0.2989 * gt_image[0] + 0.5870 * gt_image[1] + 0.1140 * gt_image[2]
                g_x = torch.abs(torch.roll(gray, shifts=-1, dims=1) - gray)
                g_y = torch.abs(gray[1:, :] - gray[:-1, :])
                beta = float(getattr(opt, "depth_jump_edge_beta", 8.0))
                edge_w_x = torch.exp(-beta * g_x)
                edge_w_y = torch.exp(-beta * g_y)

                tau = float(getattr(opt, "depth_jump_thresh", 0.06))
                excess_x = torch.relu(torch.abs(d_x) - tau)
                excess_y = torch.relu(torch.abs(d_y) - tau)

                a_x = torch.minimum(alpha, torch.roll(alpha, shifts=-1, dims=1))
                a_y = torch.minimum(alpha[1:, :], alpha[:-1, :])
                valid_x = (a_x > opt.depth_alpha_thresh).float()
                valid_y = (a_y > opt.depth_alpha_thresh).float()

                w_x = area_w.squeeze(0) * a_x * edge_w_x
                w_y = area_w.squeeze(0)[1:, :] * a_y * edge_w_y

                depth_jump_loss = (excess_x * valid_x * w_x).sum() / (w_x.sum() + 1e-6)
                depth_jump_loss = depth_jump_loss + (excess_y * valid_y * w_y).sum() / (w_y.sum() + 1e-6)

            if opt.depth_jump2_weight > 0:
                depth = depth.clamp_min(1e-4)
                log_depth = torch.log(depth)
                dxx = torch.roll(log_depth, shifts=-1, dims=1) - 2.0 * log_depth + torch.roll(log_depth, shifts=1, dims=1)
                dyy = log_depth[2:, :] - 2.0 * log_depth[1:-1, :] + log_depth[:-2, :]

                inv_sin2 = (1.0 / horiz_metric) ** 2
                dxx = dxx * inv_sin2

                gray = 0.2989 * gt_image[0] + 0.5870 * gt_image[1] + 0.1140 * gt_image[2]
                g_x = torch.abs(torch.roll(gray, shifts=-1, dims=1) - gray)
                g_y = torch.abs(gray[1:, :] - gray[:-1, :])
                beta = float(getattr(opt, "depth_jump_edge_beta", 8.0))
                edge_w_x2 = torch.exp(-beta * g_x)
                edge_w_y2 = torch.exp(-beta * (0.5 * (g_y[:-1, :] + g_y[1:, :])))

                tau2 = float(getattr(opt, "depth_jump2_thresh", 0.03))
                excess_x2 = torch.relu(torch.abs(dxx) - tau2)
                excess_y2 = torch.relu(torch.abs(dyy) - tau2)

                a_x = torch.minimum(alpha, torch.roll(alpha, shifts=-1, dims=1))
                a_x = torch.minimum(a_x, torch.roll(alpha, shifts=1, dims=1))
                a_y = torch.minimum(alpha[1:-1, :], alpha[:-2, :])
                a_y = torch.minimum(a_y, alpha[2:, :])

                valid_x2 = (a_x > opt.depth_alpha_thresh).float()
                valid_y2 = (a_y > opt.depth_alpha_thresh).float()

                w_x2 = area_w.squeeze(0) * a_x * edge_w_x2
                w_y2 = area_w.squeeze(0)[1:-1, :] * a_y * edge_w_y2

                depth_jump2_loss = (excess_x2 * valid_x2 * w_x2).sum() / (w_x2.sum() + 1e-6)
                depth_jump2_loss = depth_jump2_loss + (excess_y2 * valid_y2 * w_y2).sum() / (w_y2.sum() + 1e-6)

        ramp_start = int(opt.iterations * float(getattr(opt, "geometry_ramp_start_ratio", 0.4)))
        ramp_end = int(opt.iterations * float(getattr(opt, "geometry_ramp_end_ratio", 0.9)))
        ramp_min = float(getattr(opt, "geometry_ramp_min_scale", 0.2))
        if ramp_end <= ramp_start:
            geom_scale = 1.0
        elif iteration <= ramp_start:
            geom_scale = ramp_min
        elif iteration >= ramp_end:
            geom_scale = 1.0
        else:
            t = (iteration - ramp_start) / float(ramp_end - ramp_start)
            geom_scale = ramp_min + (1.0 - ramp_min) * t

        dn_start = int(opt.iterations * float(getattr(opt, "depth_normal_start_ratio", 0.55)))
        dn_full = int(opt.iterations * float(getattr(opt, "depth_normal_full_ratio", 0.85)))
        if dn_full <= dn_start:
            dn_scale = 1.0 if iteration >= dn_start else 0.0
        elif iteration <= dn_start:
            dn_scale = 0.0
        elif iteration >= dn_full:
            dn_scale = 1.0
        else:
            dn_scale = (iteration - dn_start) / float(dn_full - dn_start)

        # Final loss
        loss = (
            rgb_loss
            + geom_scale * opt.normal_smoothness_weight * normal_loss
            + geom_scale * opt.depth_grad_weight * depth_loss
            + geom_scale * opt.depth_laplacian_weight * depth_lap_loss
            + dn_scale * opt.depth_normal_consistency_weight * depth_normal_consistency_loss
            + geom_scale * opt.depth_jump_weight * depth_jump_loss
            + geom_scale * opt.depth_jump2_weight * depth_jump2_loss
        )
        loss.backward()

        if getattr(opt, 'grad_clip', 0) and opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gaussians.parameters(), opt.grad_clip)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                point_count = gaussians.get_xyz.shape[0]
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Points": point_count})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                current_rgb = float(rgb_loss.item())
                baseline_rgb = ema_rgb_for_densify if ema_rgb_for_densify > 0 else current_rgb
                error_weight = current_rgb / (baseline_rgb + 1e-6)
                error_weight = max(0.7, min(1.5, error_weight))
                ema_rgb_for_densify = 0.4 * current_rgb + 0.6 * ema_rgb_for_densify
                per_point_weight = None
                if getattr(gaussians, "densify_latitude_power", 0.0) > 0:
                    with torch.no_grad():
                        xyz = gaussians.get_xyz
                        R = torch.tensor(viewpoint_cam.R, device=xyz.device, dtype=xyz.dtype)
                        T = torch.tensor(viewpoint_cam.T, device=xyz.device, dtype=xyz.dtype)
                        xyz_cam = xyz @ R + T[None, :]
                        radii = torch.linalg.norm(xyz_cam, dim=1) + 1e-8
                        y_over_r = torch.clamp(xyz_cam[:, 1] / radii, -1.0, 1.0)
                        cos_theta = torch.sqrt(torch.clamp(1.0 - y_over_r * y_over_r, min=0.0))
                        cos_theta = torch.clamp(cos_theta, min=gaussians.densify_latitude_min)
                        prog = min(1.0, max(0.0, iteration / float(opt.densify_until_iter)))
                        power = gaussians.densify_latitude_power * (1.0 - 0.7 * prog)
                        inv_cos = torch.clamp(1.0 / cos_theta, max=1.0 / gaussians.densify_latitude_min)
                        per_point_weight = inv_cos ** power
                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                    weight=error_weight,
                    per_point_weight=per_point_weight,
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    start_opacity = 0.01
                    end_opacity = 0.08
                    denom = max(1, opt.densify_until_iter - opt.densify_from_iter)
                    t = (iteration - opt.densify_from_iter) / float(denom)
                    t = max(0.0, min(1.0, t))
                    min_opacity = float(start_opacity + t * (end_opacity - start_opacity))

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        min_opacity,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    gaussians.compute_3D_filter(cameras=trainCameras)

                    denom = gaussians.denom.squeeze()
                    if denom.numel() > 0:
                        visible = denom > 0
                        if visible.any():
                            threshold = torch.quantile(denom[visible], 0.2)
                            visibility_mask = visible & (denom <= threshold)
                        else:
                            visibility_mask = torch.zeros_like(denom, dtype=torch.bool)
                    else:
                        visibility_mask = torch.zeros_like(denom, dtype=torch.bool)
                    gaussians.decay_opacity_masked(0.98, visibility_mask)

                reset_interval = 2000
                if iteration % reset_interval == 0 and iteration > opt.densify_from_iter:
                    safe_reset_value = min_opacity * 1.2
                    gaussians.reset_opacity(min_opacity_threshold=safe_reset_value)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if rendering.shape[0] > 7:
                        with torch.no_grad():
                            depth_map = rendering[6]
                            alpha_map = rendering[7]
                            err_map = torch.mean(torch.abs(image - gt_image), dim=0)
                            err_weighted = err_map * weights.squeeze(0)
                            valid = torch.isfinite(depth_map) & (alpha_map > 0.15)
                            if valid.any():
                                seed_ratio = 0.8
                                if iteration <= int(opt.densify_until_iter * seed_ratio):
                                    threshold = torch.quantile(err_weighted[valid], 0.995)
                                    candidate = valid & (err_weighted >= threshold)
                                    candidate_idx = torch.where(candidate.view(-1))[0]
                                    max_new = 512
                                    if gaussians.max_points > 0:
                                        max_new = min(max_new, max(0, gaussians.max_points - gaussians.get_xyz.shape[0]))
                                    if candidate_idx.numel() > 0 and max_new > 0:
                                        scores = err_weighted.view(-1)[candidate_idx]
                                        k = min(int(max_new), int(candidate_idx.numel()))
                                        _, topk = torch.topk(scores, k=k, largest=True)
                                        select_idx = candidate_idx[topk]
                                        seed_scale = max(scene.cameras_extent * 0.003, gaussians.min_scale * 1.2)
                                        depth_offsets = [0.0, scene.cameras_extent * 0.008]
                                        for depth_offset in depth_offsets:
                                            depth_map_offset = torch.clamp(depth_map + float(depth_offset), min=1e-4)
                                            gaussians.add_points_from_pixels(
                                                viewpoint_cam,
                                                select_idx,
                                                depth_map_offset,
                                                gt_image,
                                                seed_scale,
                                                init_opacity=0.10,
                                            )

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration % 100 == 0:
                    gaussians.enforce_scale_floor(min_scale=gaussians.min_scale)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for viewpoint in config['cameras']:
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    image = rendering[:3, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[8_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
