# copy from 2DGS
import torch
import torch.nn.functional as F
from utils.graphics_utils import equirectangular_rays

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    device = depthmap.device
    theta_min = float(getattr(view, "pano_theta_min", -0.5 * torch.pi))
    theta_max = float(getattr(view, "pano_theta_max", 0.5 * torch.pi))

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing='xy'
    )
    rays_cam = equirectangular_rays(
        W,
        H,
        grid_x,
        grid_y,
        theta_min=theta_min,
        theta_max=theta_max,
    ).reshape(-1, 3)
    rays_world = rays_cam @ c2w[:3, :3].T
    rays_world = F.normalize(rays_world, dim=-1)
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_world + rays_o
    return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    H, W = points.shape[:2]

    points_xp = torch.roll(points, shifts=-1, dims=1)
    points_xm = torch.roll(points, shifts=1, dims=1)
    dx = points_xp - points_xm

    points_yp = points[2:, :]
    points_ym = points[:-2, :]
    dy = points_yp - points_ym

    device = points.device
    ys = torch.arange(H, device=device, dtype=points.dtype) + 0.5
    theta_min = torch.as_tensor(float(getattr(view, "pano_theta_min", -0.5 * torch.pi)), device=device, dtype=points.dtype)
    theta_max = torch.as_tensor(float(getattr(view, "pano_theta_max", 0.5 * torch.pi)), device=device, dtype=points.dtype)
    theta = theta_min + (ys / max(H, 1)) * (theta_max - theta_min)
    cos_theta = torch.cos(theta).abs().clamp_min(1e-4)
    dx = dx / cos_theta.view(H, 1, 1)

    normal_map = torch.nn.functional.normalize(
        torch.cross(dx[1:-1, :], dy, dim=-1), dim=-1
    )
    output[1:-1, :, :] = normal_map
    return output, points
