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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self._kernel_size = 0.0
        # self.use_spatial_gaussian_bias = False
        self.ray_jitter = False
        self.resample_gt_image = False
        self.load_allres = False
        self.sample_more_highres = False
        self.use_decoupled_appearance = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.compute_view2gaussian_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 8_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.appearance_embeddings_lr = 0.001
        self.appearance_network_lr = 0.001
        self.percent_dense = 0.015
        self.lambda_dssim = 0.2
        self.densification_interval = 200
        self.densify_from_iter = 500
        self.densify_until_iter = 4_000
        self.densify_grad_threshold = 0.0002
        self.split_factor = 2
        self.max_points = 2_000_000
        self.clone_grad_multiplier = 1.5
        self.grad_clip = 1.0
        self.min_scale = 0.0001
        self.densify_grad_clip = 100.0
        self.normal_smoothness_weight = 0.09
        self.depth_grad_weight = 0.08
        self.depth_laplacian_weight = 0.05
        self.depth_normal_consistency_weight = 0.02
        self.depth_normal_start_ratio = 0.55
        self.depth_normal_full_ratio = 0.85
        self.depth_jump_weight = 0.32
        self.depth_jump_thresh = 0.018
        self.depth_jump_edge_beta = 8.0
        self.depth_jump2_weight = 0.22
        self.depth_jump2_thresh = 0.010
        self.depth_robust_eps = 1e-3
        self.geometry_ramp_start_ratio = 0.15
        self.geometry_ramp_end_ratio = 0.6
        self.geometry_ramp_min_scale = 0.6
        self.normal_alpha_thresh = 0.05
        self.depth_alpha_thresh = 0.12
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
