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
import torch.nn.functional as F
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def cartesian_to_equirect(
    points: torch.Tensor,
    width: int,
    height: int,
    *,
    theta_min: float = -0.5 * math.pi,
    theta_max: float = 0.5 * math.pi,
    wrap: bool = True,
    clamp: bool = True,
):
    if points.shape[-1] != 3:
        raise ValueError("Expected (..., 3) points for cartesian_to_equirect")

    eps = 1e-6
    width_t = torch.as_tensor(width, dtype=points.dtype, device=points.device)
    height_t = torch.as_tensor(height, dtype=points.dtype, device=points.device)
    height_span = torch.as_tensor(max(height - 1, 1), dtype=points.dtype, device=points.device)
    theta_min_t = torch.as_tensor(theta_min, dtype=points.dtype, device=points.device)
    theta_span_t = torch.as_tensor(max(theta_max - theta_min, 1e-6), dtype=points.dtype, device=points.device)

    radii = torch.linalg.norm(points, dim=-1, keepdim=True).clamp_min(eps)
    phi = torch.atan2(points[..., 0], points[..., 2])
    theta = torch.asin(torch.clamp(points[..., 1] / radii[..., 0], -1.0, 1.0))

    u = (phi / (2.0 * math.pi) + 0.5) * width_t
    v = ((theta - theta_min_t) / theta_span_t) * height_span

    if wrap:
        u = torch.remainder(u, width_t)
    if clamp:
        v = torch.clamp(v, 0.0, height_span)
    return u, v


def equirectangular_rays(
    width: int,
    height: int,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    *,
    theta_min: float = -0.5 * math.pi,
    theta_max: float = 0.5 * math.pi,
):
    if grid_x.shape != grid_y.shape:
        raise ValueError("grid_x and grid_y must share shape for equirectangular_rays")

    eps = 1e-6
    width_t = torch.as_tensor(width, dtype=grid_x.dtype, device=grid_x.device)
    height_span = torch.as_tensor(max(height - 1, 1), dtype=grid_y.dtype, device=grid_y.device)
    theta_min_t = torch.as_tensor(theta_min, dtype=grid_y.dtype, device=grid_y.device)
    theta_span_t = torch.as_tensor(max(theta_max - theta_min, 1e-6), dtype=grid_y.dtype, device=grid_y.device)

    u = (grid_x + 0.5) / width_t
    v = grid_y / height_span

    phi = (u - 0.5) * 2.0 * math.pi
    theta = theta_min_t + v * theta_span_t

    cos_theta = torch.cos(theta)
    dir_x = torch.sin(phi) * cos_theta
    dir_y = torch.sin(theta)
    dir_z = torch.cos(phi) * cos_theta
    dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)

    return F.normalize(dirs, dim=-1, eps=eps)
