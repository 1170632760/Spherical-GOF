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

from pathlib import Path
import os
from math import exp
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from argparse import ArgumentParser
import json
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS

def iter_image_pairs(renders_dir, gt_dir):
    image_names = sorted([fname for fname in os.listdir(renders_dir) if not fname.startswith(".")])
    for fname in image_names:
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT for {fname} in {gt_dir}")
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        yield fname, render, gt

def psnr(img1, img2, ws_map=None):
    mse_map = (img1 - img2) ** 2
    mse = mse_map.view(img1.shape[0], -1).mean(1, keepdim=True)
    if ws_map is None:
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    ws_mse = (
        (mse_map * ws_map).view(img1.shape[0], -1).mean(1, keepdim=True)
        / ws_map.mean()
    )
    return 20 * torch.log10(1.0 / torch.sqrt(mse)), 20 * torch.log10(1.0 / torch.sqrt(ws_mse))


def ssim(img1, img2, window_size=11, ws_map=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, ws_map)


def create_window(window_size, channel):
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d_window.expand(channel, 1, window_size, window_size).contiguous()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _ssim(img1, img2, window, window_size, channel, ws_map):
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if ws_map is None:
        return ssim_map.mean()
    ws_ssim_map = ssim_map * ws_map
    return ssim_map.mean(), ws_ssim_map.mean() / ws_map.mean()


def est_wsmap(img):
    height, width = img.shape[-2:]
    col = torch.arange(height, device=img.device, dtype=img.dtype)
    ws_map = torch.cos((col + 0.5 - height / 2) * torch.pi / height).reshape(height, 1)
    return ws_map.expand(height, width)


def evaluate(model_paths, scale):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ f"gt_{scale}"
                renders_dir = method_dir / f"test_preds_{scale}"
                ssims = []
                psnrs = []
                lpipss = []
                image_names = []

                with torch.no_grad():
                    for fname, render, gt in tqdm(
                        iter_image_pairs(renders_dir, gt_dir),
                        desc="Metric evaluation progress",
                    ):
                        render_t = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                        gt_t = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                        render_t = torch.clamp(render_t, 0.0, 1.0)
                        gt_t = torch.clamp(gt_t, 0.0, 1.0)
                        ws_map = est_wsmap(render_t)
                        test_psnr, _ = psnr(render_t, gt_t, ws_map)
                        test_ssim, _ = ssim(render_t, gt_t, ws_map=ws_map)
                        ssims.append(float(test_ssim))
                        psnrs.append(float(test_psnr))
                        lpipss.append(float(lpips_fn(render_t, gt_t).detach().cpu()))
                        image_names.append(fname)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as exc:
            print(f"Unable to compute metrics for model {scene_dir}: {exc}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = LPIPS(net_type="alex").to(device)
    lpips_fn.eval()

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=-1)

    args = parser.parse_args()
    evaluate(args.model_paths, args.resolution)
