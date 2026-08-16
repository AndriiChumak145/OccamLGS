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
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import yaml
import numpy as np
from utils.siglip_extractor import OnlineSiglipExtractor


def _select_views_fps(views: list, max_feature_views: int) -> list:
    """Select up to max_feature_views using Farthest Point Sampling on 3D camera centers."""
    if max_feature_views is None or max_feature_views <= 0 or len(views) <= max_feature_views:
        return views

    centers = []
    for v in views:
        c = v.camera_center.detach().cpu().numpy()
        centers.append(c)
    centers = np.array(centers, dtype=np.float32)
    assert centers.ndim == 2 and centers.shape[1] == 3, f"Invalid camera centers shape: {centers.shape}"

    selected_indices = [0]
    distances = np.linalg.norm(centers - centers[0], axis=1)

    for _ in range(1, max_feature_views):
        far_idx = int(np.argmax(distances))
        selected_indices.append(far_idx)
        new_distances = np.linalg.norm(centers - centers[far_idx], axis=1)
        distances = np.minimum(distances, new_distances)

    selected_indices.sort()
    return [views[i] for i in selected_indices]


def _filter_views_with_language_features(views, language_feature_dir, max_feature_views=None):
    if max_feature_views is not None and max_feature_views <= 0:
        return []
    filtered = []
    for view in views:
        base_name = view.image_name.split('.')[0]
        base_path = os.path.join(language_feature_dir, base_name)
        if os.path.isfile(base_path + "_s.npy") and os.path.isfile(base_path + "_f.npy"):
            filtered.append(view)
    if max_feature_views is not None and max_feature_views > 0 and len(filtered) > max_feature_views:
        filtered = _select_views_fps(filtered, max_feature_views)
    return filtered


def extract_gaussian_features(model_path, iteration, views, gaussians, pipeline, background, feature_level, language_feature_dir, 
                              config_path=None, use_online_siglip=False, upsample_method="bilinear"):

    language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')

    # Local resolution from configuration file
    feature_dim = None
    model_id = None
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_spec = yaml.safe_load(f)
        feature_dim = config_spec.get("hidden_dim")
        model_id = config_spec.get("model_id")

    online_extractor = None
    if use_online_siglip:
        online_extractor = OnlineSiglipExtractor(model_id=model_id, upsample_method=upsample_method)
    
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)

        if use_online_siglip:
            view.load_image_data()
            gt_language_feature, gt_mask = online_extractor.extract(view)
            view.unload_image_data()
        else:
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
            
def process_scene_language_features(
        dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int, 
        config_path=None, max_feature_views=None, use_online_siglip=False, geometry_ply=None, upsample_method="bilinear",
        view_sampling_method="fps"):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        if geometry_ply is not None:
            print(f"Bypassing .pth checkpoint. Loading explicit geometry from: {geometry_ply}")
            gaussians.load_ply(geometry_ply)
            print(f"len(gaussians): {len(gaussians._xyz)}")
        else:
            checkpoint = os.path.join(dataset.model_path, f'chkpnt{iteration}.pth')
            print(f"Loading geometry from PyTorch checkpoint: {checkpoint}")
            (model_params, _) = torch.load(checkpoint, weights_only=False)
            gaussians.restore_rgb(model_params, opt)
            
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        language_feature_dir = getattr(dataset, "lf_path", os.path.join(dataset.source_path, dataset.language_features_name))
        train_views = scene.getTrainCameras()

        if use_online_siglip:
            all_valid_views = train_views
        else:
            all_valid_views = _filter_views_with_language_features(train_views, language_feature_dir)
            if len(all_valid_views) == 0:
                raise FileNotFoundError(f"No language feature files found in {language_feature_dir}.")

        method = (view_sampling_method or "fps").lower().strip()
        if max_feature_views is not None and max_feature_views > 0 and len(all_valid_views) > max_feature_views:
            if method == "fps":
                filtered_views = _select_views_fps(all_valid_views, max_feature_views)
            elif method == "uniform":
                skip_step = max(1, len(all_valid_views) // max_feature_views)
                filtered_views = all_valid_views[::skip_step][:max_feature_views]
            elif method == "sequential":
                filtered_views = all_valid_views[:max_feature_views]
            else:
                raise ValueError(f"Unsupported view_sampling_method '{view_sampling_method}'. Choose fps, uniform, or sequential.")
        else:
            filtered_views = all_valid_views

        print(f"Selected {len(filtered_views)}/{len(train_views)} feature views using '{method}' sampling method.")

        print(f"Memory before extract_gaussian_features: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        extract_gaussian_features(
            dataset.model_path,
            iteration,
            filtered_views,
            gaussians,
            pipeline,
            background,
            feature_level,
            language_feature_dir,
            config_path=config_path,
            use_online_siglip=use_online_siglip,
            upsample_method=upsample_method
        )
        print(f"Memory after extract_gaussian_features: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_feature_views", type=int, default=None)
    parser.add_argument("--geometry_ply", type=str, default=None, help="Path to a specific .ply file to use instead of a PyTorch checkpoint")
    parser.add_argument("--online_siglip", action="store_true", help="Generate features in VRAM instead of disk")
    parser.add_argument("--config", type=str, default=None, help="Path to the model profile YAML configuration file")
    parser.add_argument("--upsample_method", type=str, default="bilinear", choices=["bilinear", "naf"], help="Method for upsampling SigLIP patches")
    parser.add_argument("--view_sampling_method", type=str, default="fps", choices=["fps", "uniform", "sequential"], help="Sampling method for feature views when max_feature_views is set")
    
    # Grab the custom CLI arguments BEFORE the config file wipes them
    safe_args, _ = parser.parse_known_args()
    
    # Let OccamLGS load its config file natively
    args = get_combined_args(parser)
    
    # Inject the custom arguments back into the final Namespace
    args.max_feature_views = safe_args.max_feature_views
    args.geometry_ply = safe_args.geometry_ply
    args.online_siglip = safe_args.online_siglip
    args.config = safe_args.config
    args.upsample_method = safe_args.upsample_method
    args.view_sampling_method = safe_args.view_sampling_method

    # Initialize system state (RNG)
    safe_state(args.quiet)

    process_scene_language_features(
        model.extract(args),
        opt.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.feature_level,
        max_feature_views=args.max_feature_views,
        use_online_siglip=args.online_siglip,
        config_path=args.config,
        geometry_ply=args.geometry_ply,
        upsample_method=args.upsample_method,
        view_sampling_method=args.view_sampling_method
    )