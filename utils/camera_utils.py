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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8, 16, 32, 64]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  pano_theta_min=getattr(cam_info, "pano_theta_min", -0.5 * np.pi),
                  pano_theta_max=getattr(cam_info, "pano_theta_max", 0.5 * np.pi))

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    if hasattr(camera, 'image_width') and hasattr(camera, 'image_height'):
        width = int(camera.image_width)
        height = int(camera.image_height)
    else:
        try:
            width, height = camera.image.size
        except Exception:
            width = 0
            height = 0

    if hasattr(camera, 'FoVy') and hasattr(camera, 'FoVx'):
        fov_y = float(camera.FoVy)
        fov_x = float(camera.FoVx)
    else:
        fov_y = 0.0
        fov_x = 0.0

    img_name = camera.image_name if hasattr(camera, 'image_name') else getattr(camera, 'image_path', '')

    camera_entry = {
        'id': id,
        'img_name': img_name,
        'width': width,
        'height': height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'projection': 'equirectangular',
        'fov_y': fov_y,
        'fov_x': fov_x,
        'pano_theta_min': float(getattr(camera, 'pano_theta_min', -0.5 * np.pi)),
        'pano_theta_max': float(getattr(camera, 'pano_theta_max', 0.5 * np.pi)),
    }
    return camera_entry
