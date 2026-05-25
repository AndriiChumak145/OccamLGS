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
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.dlpack
import matplotlib.pyplot as plt
import time


def _filter_views_with_language_features(views, language_feature_dir, max_feature_views=None):
    if max_feature_views is not None and max_feature_views <= 0:
        return []
    filtered = []
    for view in views:
        base_name = view.image_name.split('.')[0]
        base_path = os.path.join(language_feature_dir, base_name)
        if os.path.isfile(base_path + "_s.npy") and os.path.isfile(base_path + "_f.npy"):
            filtered.append(view)
            if max_feature_views is not None and len(filtered) >= max_feature_views:
                break
    return filtered


def extract_gaussian_features(model_path, iteration, views, gaussians, pipeline, background, feature_level, language_feature_dir, feature_dim=None):

    language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')
    
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_pkg= render(view, gaussians, pipeline, background)

        gt_language_feature, gt_mask = view.get_language_feature(language_feature_dir=language_feature_dir, feature_level=feature_level)
        actual_dim = int(gt_language_feature.shape[0])
        if feature_dim is None:
            feature_dim_use = actual_dim
        else:
            if int(feature_dim) != actual_dim:
                raise ValueError(
                    f"feature_dim {feature_dim} does not match loaded feature dim {actual_dim}."
                )
            feature_dim_use = int(feature_dim)
        activated = render_pkg["info"]["activated"]
        significance = render_pkg["info"]["significance"]
        means2D = render_pkg["info"]["means2d"]
        
        mask = activated[0] > 0
        gaussians.accumulate_gaussian_feature_per_view(
            gt_language_feature.permute(1, 2, 0),
            gt_mask.squeeze(0),
            mask,
            significance[0, mask],
            means2D[0, mask],
            feature_dim=feature_dim_use,
        )
        
    gaussians.finalize_gaussian_features()

    torch.save((gaussians.capture_language_feature(), 0), language_feature_save_path)
    print("checkpoint saved to: ", language_feature_save_path)
            
def process_scene_language_features(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int, feature_dim=None, max_feature_views=None):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}.pth')
        (model_params, _) = torch.load(checkpoint, weights_only=False)
        gaussians.restore_rgb(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        language_feature_dir = getattr(dataset, "lf_path", os.path.join(dataset.source_path, dataset.language_features_name))
        train_views = scene.getTrainCameras()
        filtered_views = _filter_views_with_language_features(train_views, language_feature_dir, max_feature_views)
        if len(filtered_views) == 0:
            raise FileNotFoundError(
                f"No language feature files found in {language_feature_dir}."
            )
        if len(filtered_views) < len(train_views):
            print(
                f"Using {len(filtered_views)}/{len(train_views)} cameras with language features."
            )

        extract_gaussian_features(
            args.model_path,
            iteration,
            filtered_views,
            gaussians,
            pipeline,
            background,
            feature_level,
            language_feature_dir,
            feature_dim,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--max_feature_views", type=int, default=None)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    process_scene_language_features(
        model.extract(args),
        opt.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.feature_level,
        args.feature_dim,
        args.max_feature_views,
    )