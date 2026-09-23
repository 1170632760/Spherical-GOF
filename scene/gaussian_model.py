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

import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, cartesian_to_equirect
from utils.depth_utils import depths_to_points
from utils.general_utils import strip_symmetric, build_scaling_rotation
import trimesh
from scene.appearance_network import AppearanceNetwork
from scene.cameras import Camera
from einops import einsum
from typing import List

@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    H, W = cameras[0].image_height, cameras[0].image_width
    theta_min = float(getattr(cameras[0], "pano_theta_min", -0.5 * math.pi))
    theta_max = float(getattr(cameras[0], "pano_theta_max", 0.5 * math.pi))

    # full_proj_matrices: (n_view, 4, 4)
    view_matrices = torch.stack(
        [cam.world_view_transform for cam in cameras], dim=0
    ).transpose(1, 2)

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    # homo_points: (N, 4)
    homo_points = torch.cat([points, ones], dim=-1)

    view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    u, v = cartesian_to_equirect(
        view_points,
        W,
        H,
        theta_min=theta_min,
        theta_max=theta_max,
        wrap=False,
        clamp=False,
    )
    valid_coords = torch.isfinite(u) & torch.isfinite(v)
    valid_coords = valid_coords & (v >= 0.0) & (v <= float(max(H - 1, 0)))
    u = torch.remainder(u, W)
    v = torch.clamp(v, 0.0, max(H - 1, 0))

    depth = torch.linalg.norm(view_points, dim=-1)
    cull_near_fars = (depth >= near) & (depth <= far)

    mask = torch.any(cull_near_fars & valid_coords, dim=0)
    return mask


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.split_factor = 2
        self.max_points = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        # appearance network and appearance embedding
        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()

        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        min_scale = getattr(self, 'min_scale', 1e-4)
        return torch.clamp(self.scaling_activation(self._scaling), min=min_scale)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling

        scales = torch.sqrt(torch.square(scales) + torch.square(self.filter_3D))

        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling

        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_view2gaussian(self, viewmatrix):
        r = self._rotation
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)

        rots = R
        xyz = self.get_xyz
        N = xyz.shape[0]
        G2W = torch.zeros((N, 4, 4), device='cuda')
        G2W[:, :3, :3] = rots # TODO check if we need to transpose here
        G2W[:, :3, 3] = xyz
        G2W[:, 3, 3] = 1.0

        viewmatrix = viewmatrix.transpose(0, 1)
        G2V = viewmatrix @ G2W

        R = G2V[:, :3, :3]
        t = G2V[:, :3, 3]

        t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
        V2G = torch.zeros((N, 4, 4), device='cuda')
        V2G[:, :3, :3] = R.transpose(1, 2)
        V2G[:, :3, 3] = t2
        V2G[:, 3, 3] = 1.0

        # transpose view2gaussian to match glm in CUDA code
        V2G = V2G.transpose(2, 1).contiguous()

        # precompute results to reduce computation and IO
        scales = self.get_scaling_with_3D_filter
        S_inv_square = 1.0 / (scales ** 2)
        R = V2G[:, :3, :3].transpose(1, 2)
        t2 = V2G[:, 3:, :3]

        C = torch.sum((t2 ** 2) * S_inv_square[:, None, :], dim=2)
        S_inv_square_R = S_inv_square[:, :, None] * R
        B = t2 @ S_inv_square_R
        Sigma = R.transpose(1, 2) @ S_inv_square_R
        merged = torch.cat([Sigma[:, :, 0], Sigma[:, 1:, 1], Sigma[:, 2:, 2], B.squeeze(), C], dim=1)

        return merged

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        xyz = self.get_xyz

        filter_candidates = []
        min_cos = float(getattr(self, "densify_latitude_min", 0.2))
        min_cos = max(min_cos, 0.05)

        for camera in cameras:
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)

            xyz_cam = xyz @ R + T[None, :]
            radii = torch.linalg.norm(xyz_cam, dim=1)

            theta_min = float(getattr(camera, "pano_theta_min", -0.5 * math.pi))
            theta_max = float(getattr(camera, "pano_theta_max", 0.5 * math.pi))
            theta_span = max(theta_max - theta_min, 1e-6)
            dtheta = theta_span / camera.image_height
            dphi = 2.0 * math.pi / camera.image_width
            y_over_r = torch.clamp(xyz_cam[:, 1] / (radii + 1e-8), -1.0, 1.0)
            cos_theta = torch.sqrt(torch.clamp(1.0 - y_over_r * y_over_r, min=0.0))
            cos_theta = torch.clamp(cos_theta, min=min_cos)
            rad_per_pixel = torch.sqrt(cos_theta * dtheta * dphi)

            filter_candidate = radii * rad_per_pixel

            visible = get_frustum_mask(xyz, [camera])
            if visible.any():
                cand = filter_candidate.clone()
                cand[~visible] = float('nan')
                filter_candidates.append(cand)

        if len(filter_candidates) == 0:
            return
        stacked = torch.stack(filter_candidates, dim=0)
        q = float(getattr(self, "filter_3d_quantile", 0.8))
        q = min(max(q, 0.5), 1.0)

        filter_radius = torch.nanquantile(stacked, q, dim=0)

        filter_radius = torch.nan_to_num(filter_radius, nan=0.0)
        filter_3D = filter_radius * (0.2 ** 0.5)

        self.filter_3D = filter_3D[..., None]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.filter_3D = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    @torch.no_grad()
    def add_points_from_pixels(self, view, pixel_indices, depth_map, rgb_image, base_scale, init_opacity=0.08):
        if pixel_indices is None or pixel_indices.numel() == 0:
            return 0
        H, W = depth_map.shape
        points = depths_to_points(view, depth_map).reshape(-1, 3)
        pixel_indices = pixel_indices.to(points.device)
        points = points[pixel_indices]

        ys = (pixel_indices // W).long()
        xs = (pixel_indices % W).long()
        colors = rgb_image[:, ys, xs].permute(1, 0).contiguous()

        features = torch.zeros((colors.shape[0], 3, (self.max_sh_degree + 1) ** 2), device=colors.device)
        features[:, :3, 0] = RGB2SH(colors)
        features[:, 3:, 1:] = 0.0

        scale_value = float(base_scale)
        new_scaling = self.scaling_inverse_activation(
            torch.full((colors.shape[0], 3), scale_value, device=colors.device)
        )
        new_rotation = torch.zeros((colors.shape[0], 4), device=colors.device)
        new_rotation[:, 0] = 1.0
        new_opacity = self.inverse_opacity_activation(
            torch.full((colors.shape[0], 1), float(init_opacity), device=colors.device)
        )

        self.densification_postfix(
            points,
            features[:, :, 0:1].transpose(1, 2).contiguous(),
            features[:, :, 1:].transpose(1, 2).contiguous(),
            new_opacity,
            new_scaling,
            new_rotation,
        )

        if hasattr(self, "filter_3D") and self.filter_3D is not None:
            filter_value = scale_value * (0.2 ** 0.5)
            pad = torch.full((colors.shape[0], 1), filter_value, device=colors.device)
            self.filter_3D = torch.cat([self.filter_3D, pad], dim=0)
        return colors.shape[0]

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.split_factor = max(0, int(getattr(training_args, "split_factor", 2)))
        self.max_points = max(0, int(getattr(training_args, "max_points", 0)))

        self.min_scale_ratio = float(getattr(training_args, "min_scale_ratio", 5e-5))
        self.min_scale = getattr(training_args, 'min_scale', 1e-4)

        self.small_scale_prune_factor = float(getattr(training_args, "small_scale_prune_factor", 3e-4))
        self.small_scale_prune_grad_ratio = float(getattr(training_args, "small_scale_prune_grad_ratio", 0.35))
        self.small_scale_prune_opacity_ratio = float(getattr(training_args, "small_scale_prune_opacity_ratio", 0.8))
        self.small_scale_prune_cap = float(getattr(training_args, "small_scale_prune_cap", 0.02))

        self.auto_max_points_ratio = float(getattr(training_args, "auto_max_points_ratio", 1.6))
        self.auto_max_points_min = int(getattr(training_args, "auto_max_points_min", 200000))
        if self.max_points <= 0:
            base_count = int(self.get_xyz.shape[0])
            self.max_points = max(int(base_count * self.auto_max_points_ratio), self.auto_max_points_min)

        self.densify_latitude_power = getattr(training_args, "densify_latitude_power", 0.6)
        self.densify_latitude_min = getattr(training_args, "densify_latitude_min", 0.2)
        self.clone_grad_multiplier = getattr(training_args, "clone_grad_multiplier", 1.5)
        self.densify_grad_clip = getattr(training_args, 'densify_grad_clip', 100.0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr, "name": "appearance_embeddings"},
            {'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr, "name": "appearance_network"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def parameters(self):
        if self.optimizer is None:
            return []
        return [parameter for group in self.optimizer.param_groups for parameter in group["params"]]

    def enforce_scale_floor(self, min_scale=1e-4):
        if not hasattr(self, '_scaling') or self._scaling.numel() == 0:
            return
        min_param = self.scaling_inverse_activation(
            torch.tensor(min_scale, device=self._scaling.device)
        )
        with torch.no_grad():
            self._scaling.data = torch.maximum(self._scaling.data, min_param)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_fused_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # fuse opacity and scale
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities = self.inverse_opacity_activation(current_opacity_with_filter).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()

        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @torch.no_grad()
    def get_tetra_points(self, views: List[Camera], near: float = 0.02, far: float = 1e6):
        M = trimesh.creation.box()
        M.vertices *= 2

        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter * 3. # TODO test
        # filter points with small opacity for bicycle scene
        # opacity = self.get_opacity_with_3D_filter
        # mask = (opacity > 0.1).squeeze(-1)
        # xyz = xyz[mask]
        # scale = scale[mask]
        # rots = rots[mask]

        vertices = M.vertices.T
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)

        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)

        # Mask out vertices outside of context views
        vertex_mask = get_frustum_mask(vertices, views, near, far)
        return vertices[vertex_mask], vertices_scale[vertex_mask]

    def decay_opacity_masked(self, factor: float, mask: torch.Tensor):
        if not hasattr(self, '_opacity') or self._opacity.numel() == 0:
            return
        if mask is None or mask.numel() == 0:
            return
        mask = mask.view(-1).to(self._opacity.device)
        if mask.sum().item() == 0:
            return
        with torch.no_grad():
            current_opacity = self.get_opacity
            decayed = current_opacity.clone()
            decayed[mask] = torch.clamp(decayed[mask] * float(factor), min=1e-8, max=1.0)
            new_param = self.inverse_opacity_activation(decayed)
            optimizable_tensors = self.replace_tensor_to_optimizer(new_param, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def reset_opacity(self, min_opacity_threshold=0.01):
        if not hasattr(self, '_opacity') or self._opacity.numel() == 0:
            return

        reset_target = max(0.02, min_opacity_threshold)

        with torch.no_grad():
            current_opacity = self.get_opacity

            reset_value = torch.clamp(current_opacity, max=reset_target)
            new_param = self.inverse_opacity_activation(reset_value)
            optimizable_tensors = self.replace_tensor_to_optimizer(new_param, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if hasattr(self, 'filter_3D') and self.filter_3D is not None and self.filter_3D.numel() > 0:
            if self.filter_3D.shape[0] == valid_points_mask.shape[0]:
                self.filter_3D = self.filter_3D[valid_points_mask]
            else:
                self.filter_3D = torch.zeros(
                    (self.get_xyz.shape[0], 1), device=self.max_radii2D.device
                )

        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        old_size = self.get_xyz.shape[0]
        old_xyz_gradient_accum = self.xyz_gradient_accum.clone()
        old_xyz_gradient_accum_abs = self.xyz_gradient_accum_abs.clone()
        old_xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max.clone()
        old_denom = self.denom.clone()
        old_max_radii2D = self.max_radii2D.clone()

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        new_size = self.get_xyz.shape[0]
        n_new = new_size - old_size

        self.xyz_gradient_accum = torch.cat([
            old_xyz_gradient_accum,
            torch.zeros((n_new, 1), device="cuda")
        ], dim=0)
        self.xyz_gradient_accum_abs = torch.cat([
            old_xyz_gradient_accum_abs,
            torch.zeros((n_new, 1), device="cuda")
        ], dim=0)
        self.xyz_gradient_accum_abs_max = torch.cat([
            old_xyz_gradient_accum_abs_max,
            torch.zeros((n_new, 1), device="cuda")
        ], dim=0)
        self.denom = torch.cat([
            old_denom,
            torch.zeros((n_new, 1), device="cuda")
        ], dim=0)
        self.max_radii2D = torch.cat([
            old_max_radii2D,
            torch.zeros((n_new), device="cuda")
        ], dim=0)

        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, slow_growth=False, budget_pressure=0.0, selected_indices=None):
        if self.max_points > 0 and self.get_xyz.shape[0] >= self.max_points:
            return

        split_factor = max(0, int(self.split_factor))
        if split_factor == 0:
            return

        n_init_points = self.get_xyz.shape[0]

        if selected_indices is not None:
            selected_pts_mask = torch.zeros((n_init_points), device=self._xyz.device, dtype=torch.bool)
            if selected_indices.numel() > 0:
                selected_pts_mask[selected_indices] = True

            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        else:
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            grad_limit = grad_threshold * (1.0 + 0.5 * budget_pressure)
            selected_pts_mask = torch.where(padded_grad >= grad_limit, True, False)
            padded_grad_abs = torch.zeros((n_init_points), device="cuda")
            padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
            abs_limit = grad_abs_threshold * (1.0 + 0.4 * budget_pressure)
            selected_pts_mask_abs = torch.where(padded_grad_abs >= abs_limit, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        selected_count = selected_pts_mask.sum()
        if selected_count == 0:
            return

        if selected_count > 0:
            selected_visibility = self.denom[selected_pts_mask].squeeze()
            avg_visibility = self.denom.mean()

            sparse_boost = (selected_visibility < avg_visibility * 0.7).float() * 0.3

            if selected_indices is not None:
                grads_full = torch.zeros((n_init_points), device="cuda")
                grads_full[:grads.shape[0]] = grads.squeeze()
                selected_grads = grads_full[selected_pts_mask]
            else:
                selected_grads = padded_grad[selected_pts_mask]
            priority_score = selected_grads * (1.0 + sparse_boost)

        if slow_growth:
            n_candidates = selected_count.item()
            n_keep = int(n_candidates * 0.5)
            if n_keep < n_candidates:
                mask_indices = torch.where(selected_pts_mask)[0]
                available = priority_score.numel()
                k_safe = max(0, min(int(n_keep), int(available), int(mask_indices.numel())))
                if k_safe > 0:
                    _, top_indices = torch.topk(priority_score, k=k_safe, largest=True)
                    top_indices = top_indices.to(mask_indices.device)
                    selected_idx = mask_indices[top_indices]
                    new_mask = torch.zeros_like(selected_pts_mask)
                    new_mask[selected_idx] = True
                    selected_pts_mask = new_mask
                    selected_count = selected_pts_mask.sum()

        stds = self.get_scaling[selected_pts_mask]
        means = self.get_xyz[selected_pts_mask]
        rots = self._rotation[selected_pts_mask]

        R = build_rotation(rots)

        max_scale_indices = torch.argmax(stds, dim=1)

        index_expanded = max_scale_indices.view(-1, 1, 1).expand(-1, 3, 1)
        major_axis_dir = torch.gather(R, 2, index_expanded).squeeze(-1)

        split_offset = major_axis_dir * stds.max(dim=1).values.unsqueeze(-1) * 0.5

        if split_factor == 2:
            new_xyz = torch.cat([means + split_offset, means - split_offset], dim=0)
        else:
            new_xyz_list = []
            for i in range(split_factor):
                offset_factor = (i / (split_factor - 1) - 0.5) * 2.0 if split_factor > 1 else 0.0
                new_xyz_list.append(means + split_offset * offset_factor)
            new_xyz = torch.cat(new_xyz_list, dim=0)

        new_scaling = self.scaling_inverse_activation(
            stds.repeat(split_factor, 1) / (0.8 * split_factor)
        )

        new_rotation = rots.repeat(split_factor, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(split_factor, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(split_factor, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(split_factor, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(split_factor * selected_count, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, slow_growth=False, budget_pressure=0.0, selected_indices=None):
        if self.max_points > 0 and self.get_xyz.shape[0] >= self.max_points:
            return

        n_points = self.get_xyz.shape[0]
        if selected_indices is not None:
            selected_pts_mask = torch.zeros((n_points), device=self._xyz.device, dtype=torch.bool)
            if selected_indices.numel() > 0:
                selected_pts_mask[selected_indices] = True
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        else:
            grad_limit = grad_threshold * (1.0 + 0.65 * budget_pressure)
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_limit, True, False)
            abs_limit = grad_abs_threshold * (1.0 + 0.5 * budget_pressure)
            selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= abs_limit, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        selected_count = selected_pts_mask.sum()
        if selected_count == 0:
            return

        if selected_pts_mask.sum() > 0:
            selected_visibility = self.denom[selected_pts_mask].squeeze()
            avg_visibility = self.denom.mean()

            sparse_boost = (selected_visibility < avg_visibility * 0.7).float() * 0.3
            grads_norm = torch.norm(grads, dim=-1)
            selected_grads = grads_norm[selected_pts_mask]
            priority_score = selected_grads * (1.0 + sparse_boost)

        if slow_growth:
            n_candidates = selected_pts_mask.sum().item()
            n_keep = int(n_candidates * 0.5)
            if n_keep < n_candidates:
                mask_indices = torch.where(selected_pts_mask)[0]
                available = priority_score.numel()
                k_safe = max(0, min(int(n_keep), int(available), int(mask_indices.numel())))
                if k_safe > 0:
                    _, top_indices = torch.topk(priority_score, k=k_safe, largest=True)
                    top_indices = top_indices.to(mask_indices.device)
                    selected_idx = mask_indices[top_indices]
                    new_mask = torch.zeros_like(selected_pts_mask)
                    new_mask[selected_idx] = True
                    selected_pts_mask = new_mask

        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, enable_densify=True):
        target_point_count = self.max_points if self.max_points > 0 else 2_000_000

        denom_safe = torch.where(self.denom > 0, self.denom, torch.ones_like(self.denom))
        grads = self.xyz_gradient_accum / denom_safe
        grads[self.denom == 0] = 0.0
        grads[~torch.isfinite(grads)] = 0.0
        grads_abs = self.xyz_gradient_accum_abs / denom_safe
        grads_abs[self.denom == 0] = 0.0
        grads_abs[~torch.isfinite(grads_abs)] = 0.0

        grads_norm = grads_abs.squeeze() if grads_abs.numel() > 0 else torch.norm(grads, dim=-1)

        effective_max_grad = float(max_grad)

        ratio = (grads_norm >= effective_max_grad).float().mean() if grads_norm.numel() > 0 else 0.0
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio) if grads_abs.numel() > 0 else 0.0

        current_count = self.get_xyz.shape[0]
        usage_ratio = current_count / target_point_count
        budget_pressure = max(0.0, min(1.0, (usage_ratio - 0.92) / 0.08))
        clone_candidate_mask = torch.zeros(current_count, dtype=torch.bool, device="cuda")
        split_candidate_mask = torch.zeros_like(clone_candidate_mask)

        budget_threshold = int(target_point_count * 0.85)

        budget_prune_mask = None

        should_slow_growth = False

        if enable_densify and current_count > budget_threshold:
            scales = self.get_scaling
            max_scales = torch.max(scales, dim=1).values

            clone_candidate_mask = (grads_norm > effective_max_grad * self.clone_grad_multiplier) & (max_scales <= self.percent_dense * extent)
            n_clone = clone_candidate_mask.sum().item()

            split_candidate_mask = (grads_norm > effective_max_grad) & (max_scales > self.percent_dense * extent)
            n_split = split_candidate_mask.sum().item()

            split_factor = max(2, int(self.split_factor))
            n_estimated_growth = n_clone + n_split * (split_factor - 1)

            predicted_final = current_count + n_estimated_growth

            max_add_allowed = max(int(current_count * 0.12), 18000)
            if n_estimated_growth > max_add_allowed:
                should_slow_growth = True

            usage_ratio = current_count / target_point_count

            if usage_ratio >= 0.95:
                should_slow_growth = True
                n_to_kill = int(predicted_final - target_point_count)
                n_to_kill = min(n_to_kill, int(current_count * 0.03))

            elif usage_ratio >= 0.90:
                n_to_kill = max(0, int((predicted_final - target_point_count) * 0.6))
                n_to_kill = min(n_to_kill, int(current_count * 0.02))

            elif usage_ratio >= 0.85:
                n_to_kill = max(0, int((predicted_final - target_point_count) * 0.4))
                n_to_kill = min(n_to_kill, int(current_count * 0.015))
            else:
                n_to_kill = 0

            if n_to_kill > 0:
                volumes = torch.prod(scales, dim=1)

                visibility_bonus = self.denom[:current_count].squeeze() / (self.denom.max() + 1e-8)
                importance_score = grads_norm * volumes * (1.0 + visibility_bonus) + volumes * 0.01 + 1e-10

                importance_score[clone_candidate_mask] *= 10.0
                importance_score[split_candidate_mask] *= 10.0

                potential_mask = (grads_norm > max_grad * 0.5) & ~clone_candidate_mask & ~split_candidate_mask
                importance_score[potential_mask] *= 3.0

                sparse_region_mask = (self.denom[:current_count].squeeze() < self.denom.mean() * 0.5)
                importance_score[sparse_region_mask] *= 2.0

                extreme_big = (max_scales > extent * 0.5) & ~clone_candidate_mask & ~split_candidate_mask & ~potential_mask
                small_scale_ref = extent * self.small_scale_prune_factor
                extreme_small = (max_scales < small_scale_ref * 0.2) & ~clone_candidate_mask & ~split_candidate_mask & ~potential_mask
                importance_score[extreme_big] *= 0.3
                importance_score[extreme_small] *= 0.3

                _, kill_indices = torch.topk(importance_score, k=n_to_kill, largest=False)

                budget_prune_mask = torch.zeros(current_count, dtype=torch.bool, device="cuda")
                budget_prune_mask[kill_indices] = True

        before = self._xyz.shape[0]
        clone = before
        split = before

        newly_cloned_indices = None
        newly_split_indices = None

        if enable_densify:
            current_points = self._xyz.shape[0]
            max_scales = self.get_scaling.max(dim=1).values

            clone_mask = (grads_norm > effective_max_grad * self.clone_grad_multiplier) & (max_scales <= self.percent_dense * extent)
            split_mask = (grads_norm > effective_max_grad) & (max_scales > self.percent_dense * extent)

            clone_indices = torch.where(clone_mask)[0]
            split_indices = torch.where(split_mask)[0]

            n_clone_candidates = clone_indices.numel()
            n_split_candidates = split_indices.numel()

            split_factor = max(2, int(self.split_factor))
            net_per_split = split_factor - 1

            global_add_cap = max(int(current_points * 0.18), 18000)

            if should_slow_growth:
                n_clone_candidates = n_clone_candidates // 2
                n_split_candidates = n_split_candidates // 2

            max_split_allowed = min(n_split_candidates, global_add_cap // max(1, net_per_split))
            remaining_cap_after_split = global_add_cap - max_split_allowed * net_per_split
            max_clone_allowed = min(n_clone_candidates, remaining_cap_after_split)

            final_split_indices = None
            if max_split_allowed > 0 and split_indices.numel() > 0:
                split_vis = self.denom[split_indices].squeeze()
                avg_vis = self.denom.mean()
                sparse_boost = (split_vis < avg_vis * 0.7).float() * 0.3
                split_grads = grads_norm[split_indices]
                split_priority = split_grads * (1.0 + sparse_boost)
                k = int(max_split_allowed)
                k_safe = max(0, min(k, int(split_priority.numel()), int(split_indices.numel())))
                if k_safe > 0:
                    _, topk = torch.topk(split_priority, k=k_safe, largest=True)
                    topk = topk.to(split_indices.device)
                    final_split_indices = split_indices[topk]
                else:
                    final_split_indices = torch.tensor([], dtype=torch.long, device=split_indices.device)
            else:
                final_split_indices = torch.tensor([], dtype=torch.long, device=split_indices.device)

            final_clone_indices = None
            if max_clone_allowed > 0 and clone_indices.numel() > 0:
                clone_vis = self.denom[clone_indices].squeeze()
                avg_vis = self.denom.mean()
                sparse_boost = (clone_vis < avg_vis * 0.7).float() * 0.3
                clone_grads = grads_norm[clone_indices]
                clone_priority = clone_grads * (1.0 + sparse_boost)
                k = int(max_clone_allowed)
                k_safe = max(0, min(k, int(clone_priority.numel()), int(clone_indices.numel())))
                if k_safe > 0:
                    _, topk = torch.topk(clone_priority, k=k_safe, largest=True)
                    topk = topk.to(clone_indices.device)
                    final_clone_indices = clone_indices[topk]
                else:
                    final_clone_indices = torch.tensor([], dtype=torch.long, device=clone_indices.device)
            else:
                final_clone_indices = torch.tensor([], dtype=torch.long, device=clone_indices.device)

            before_clone = self._xyz.shape[0]
            if final_clone_indices.numel() > 0:
                self.densify_and_clone(grads, max_grad, grads_abs, Q, extent, slow_growth=should_slow_growth, selected_indices=final_clone_indices)
            clone = self._xyz.shape[0]
            n_cloned = clone - before_clone
            if n_cloned > 0:
                newly_cloned_indices = torch.arange(before_clone, clone, device=self._xyz.device)

            before_split = self._xyz.shape[0]
            if final_split_indices.numel() > 0:
                self.densify_and_split(grads, max_grad, grads_abs, Q, extent, slow_growth=should_slow_growth, selected_indices=final_split_indices)
            split = self._xyz.shape[0]
            n_split = split - before_split
            if n_split > 0:
                newly_split_indices = torch.arange(before_split, split, device=self._xyz.device)

        prune_mask = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        if budget_prune_mask is not None:
            old_count = budget_prune_mask.shape[0]
            if old_count <= prune_mask.shape[0]:
                prune_mask[:old_count] = budget_prune_mask

        opacity_values = self.get_opacity.squeeze()

        denom_normalized = self.denom.squeeze() / (self.denom.mean() + 1e-8)

        adaptive_min_opacity = torch.where(
            denom_normalized < 0.5,
            min_opacity * 0.3,
            min_opacity
        )

        opacity_mask = (opacity_values < adaptive_min_opacity)
        prune_mask = prune_mask | opacity_mask

        xyz_valid = torch.isfinite(self.get_xyz).all(dim=1)
        scale_valid = torch.isfinite(self.get_scaling).all(dim=1)
        opacity_valid = torch.isfinite(self.get_opacity).squeeze()
        valid_mask = xyz_valid & scale_valid & opacity_valid
        prune_mask = prune_mask | ~valid_mask

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = prune_mask | big_points_vs

        current_count_after_grow = self.get_xyz.shape[0]
        usage_ratio_after = current_count_after_grow / max(1.0, float(target_point_count))
        if usage_ratio_after >= 0.90 and current_count_after_grow > int(target_point_count * 0.6):
            scales_current = self.get_scaling.max(dim=1).values
            small_scale_ref = extent * self.small_scale_prune_factor
            small_mask = scales_current < small_scale_ref

            grads_norm_current = torch.zeros((current_count_after_grow,), device=scales_current.device)
            n_grads = grads_norm.shape[0]
            if n_grads > 0:
                grads_norm_current[:n_grads] = grads_norm
            low_grad_mask = grads_norm_current < (max_grad * self.small_scale_prune_grad_ratio)
            low_opacity_mask = opacity_values < (min_opacity * self.small_scale_prune_opacity_ratio)
            denom_flat = self.denom[:current_count_after_grow].squeeze()
            denom_guard = denom_flat >= (denom_flat.mean() * 0.6)
            candidate_mask = small_mask & low_grad_mask & low_opacity_mask & denom_guard
            if newly_cloned_indices is not None:
                candidate_mask[newly_cloned_indices] = False
            if newly_split_indices is not None:
                candidate_mask[newly_split_indices] = False
            n_candidates = int(candidate_mask.sum().item())
            if n_candidates > 0:
                max_prune = int(current_count_after_grow * self.small_scale_prune_cap)
                n_to_prune = min(n_candidates, max_prune)
                if n_to_prune > 0:
                    scores = (opacity_values + 1e-6) * scales_current
                    candidate_indices = torch.where(candidate_mask)[0]
                    candidate_scores = scores[candidate_indices]
                    _, worst_indices = torch.topk(candidate_scores, k=n_to_prune, largest=False)
                    small_prune_mask = torch.zeros_like(prune_mask)
                    small_prune_mask[candidate_indices[worst_indices]] = True
                    prune_mask = prune_mask | small_prune_mask

        if current_count_after_grow > int(target_point_count * 0.7):
            scales_current = self.get_scaling.max(dim=1).values

            extreme_big_mask = (scales_current > extent * 0.5)
            small_scale_ref = extent * self.small_scale_prune_factor
            extreme_small_mask = (scales_current < small_scale_ref * 0.2)
            extreme_mask = extreme_big_mask | extreme_small_mask

            n_extreme = extreme_mask.sum().item()
            if n_extreme > 0:
                max_extreme_prune = int(current_count_after_grow * 0.02)
                if n_extreme > max_extreme_prune:
                    extreme_scores = torch.where(extreme_big_mask, scales_current, 1.0 / (scales_current + 1e-8))
                    _, extreme_indices = torch.topk(extreme_scores, k=max_extreme_prune, largest=True)
                    limited_extreme_mask = torch.zeros_like(extreme_mask)
                    limited_extreme_mask[extreme_indices] = True
                    prune_mask = prune_mask | limited_extreme_mask
                else:
                    prune_mask = prune_mask | extreme_mask

            if usage_ratio >= 0.94 and current_count > 0:
                denom_flat = self.denom[:current_count].squeeze()
                contribution_score = grads_norm[:current_count] * denom_flat
                if contribution_score.numel() > 0:
                    quantile = torch.quantile(contribution_score, 0.35)
                    candidate_mask = (contribution_score <= quantile)
                    candidate_mask = candidate_mask & ~clone_candidate_mask & ~split_candidate_mask
                    candidate_mask = candidate_mask & ~prune_mask[:current_count]
                    candidate_indices = torch.where(candidate_mask)[0]
                    n_candidates = candidate_indices.numel()
                    if n_candidates > 0:
                        max_low_prune = max(1, int(current_count * 0.015 * budget_pressure))
                        n_to_prune = min(max_low_prune, n_candidates)
                        if n_to_prune > 0:
                            candidate_scores = contribution_score[candidate_indices]
                            _, worst_indices = torch.topk(candidate_scores, k=n_to_prune, largest=False)
                            low_contrib_mask = torch.zeros_like(prune_mask)
                            low_contrib_mask[candidate_indices[worst_indices]] = True
                            prune_mask = prune_mask | low_contrib_mask

        final_count = self.get_xyz.shape[0]
        if final_count > target_point_count:
            target_after_prune = int(target_point_count * 0.995)
            n_over = final_count - target_after_prune

            n_over = min(n_over, int(final_count * 0.03))

            if n_over > 0:
                scales = self.get_scaling
                volumes = torch.prod(scales, dim=1)
                grads_norm_safe = grads_norm[:final_count] if grads_norm.shape[0] >= final_count else torch.zeros(final_count, device='cuda')
                importance_score = grads_norm_safe * volumes + volumes * 0.01 + 1e-10

                max_scales = scales.max(dim=1).values
                too_big_mask = (max_scales > extent * 0.5).float()
                small_scale_ref = extent * self.small_scale_prune_factor
                too_small_mask = (max_scales < small_scale_ref * 0.4).float()
                importance_score = importance_score * (1.0 - 0.5 * too_big_mask - 0.5 * too_small_mask)

                newly_added_mask = torch.zeros(final_count, dtype=torch.bool, device="cuda")
                if newly_cloned_indices is not None:
                    newly_added_mask[newly_cloned_indices] = True
                if newly_split_indices is not None:
                    newly_added_mask[newly_split_indices] = True

                importance_score[newly_added_mask] *= 1000.0

                _, kill_indices = torch.topk(importance_score, k=n_over, largest=False)
                emergency_mask = torch.zeros(final_count, dtype=torch.bool, device="cuda")
                emergency_mask[kill_indices] = True
                prune_mask = prune_mask | emergency_mask

        self.prune_points(prune_mask)

        prune = self._xyz.shape[0]

        torch.cuda.empty_cache()

        return clone - before, split - clone, split - prune

    def add_densification_stats(self, viewspace_point_tensor, update_filter, weight: float = 1.0, per_point_weight=None):
        if viewspace_point_tensor.grad is None:
            return
        grad = viewspace_point_tensor.grad[update_filter]
        grad_xy = torch.norm(grad[:, :2], dim=-1, keepdim=True)
        grad_l1 = torch.sum(torch.abs(grad), dim=-1, keepdim=True)

        max_clip = getattr(self, 'densify_grad_clip', 100.0)
        grad_xy = torch.clamp(grad_xy, max=max_clip)
        grad_l1 = torch.clamp(grad_l1, max=max_clip * 10.0)
        if weight != 1.0:
            grad_xy = grad_xy * float(weight)
            grad_l1 = grad_l1 * float(weight)

        if per_point_weight is not None:
            per_point_weight = per_point_weight.view(-1, 1)[update_filter]
            grad_xy = grad_xy * per_point_weight
            grad_l1 = grad_l1 * per_point_weight
            denom_weight = per_point_weight
        else:
            denom_weight = 1.0

        self.xyz_gradient_accum[update_filter] += grad_xy
        self.xyz_gradient_accum_abs[update_filter] += grad_l1
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(
            self.xyz_gradient_accum_abs_max[update_filter],
            grad_l1
        )

        self.denom[update_filter] += denom_weight
