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
from utils.siglip_extractor import OnlineSiglipExtractor
import yaml


def _filter_views_with_language_features(views, language_feature_dir, max_feature_views=None):
    print(f"len(views)={len(views)}")
    if max_feature_views is not None and max_feature_views <= 0:
        print("max_feature_views is set to 0 or negative, skipping all views.")
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
            
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, feature_level, language_feature_dir, config_path=None, use_online_siglip=False, upsample_method="bilinear"):
    
    save_path = os.path.join(model_path, name, "ours_{}_langfeat_{}".format(iteration, feature_level))
    render_path = os.path.join(save_path, "renders")
    gts_path = os.path.join(save_path, "gt")
    render_npy_path = os.path.join(save_path, "renders_npy")
    gts_npy_path = os.path.join(save_path,"gt_npy")
    
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_npy_path, exist_ok=True)
    os.makedirs(gts_npy_path, exist_ok=True)
    
    # Resolve parameters locally from YAML right where they are consumed
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

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # Chunked 3D Rasterization to save VRAM
        original_lang_features = gaussians._language_feature
        total_dim = original_lang_features.shape[1]
        render_chunk_size = 64
        
        rendered_chunks = []
        for i in range(0, total_dim, render_chunk_size):
            # Temporarily slice the 3D features to a small block
            gaussians._language_feature = original_lang_features[:, i:i+render_chunk_size]
            
            render_pkg = render(view, gaussians, pipeline, background, include_feature=True)
            chunk_render = render_pkg["render"]
            
            # Instantly move the rendered pixels to System RAM
            rendered_chunks.append(chunk_render.cpu())
            torch.cuda.empty_cache()
            
        # Restore the original full features
        gaussians._language_feature = original_lang_features
        
        # Stitch the final 1536-D render on CPU RAM
        rendering_cpu = torch.cat(rendered_chunks, dim=0) 
        
        if use_online_siglip:
            gt, mask = online_extractor.extract(view)
        else:
            gt, mask = view.get_language_feature(language_feature_dir=language_feature_dir, feature_level=feature_level)
        
        gt_cpu = gt.cpu()
        
        np.save(os.path.join(render_npy_path, view.image_name.split('.')[0] + ".npy"), rendering_cpu.permute(1,2,0).numpy())
        np.save(os.path.join(gts_npy_path, view.image_name.split('.')[0] + ".npy"), gt_cpu.permute(1,2,0).numpy())
        
        D, H, W = gt_cpu.shape
        if feature_dim is not None and int(feature_dim) != int(D):
            raise ValueError(
                f"feature_dim {feature_dim} does not match loaded feature dim {D}."
            )
            
        gt_np = gt_cpu.reshape(D, -1).T.numpy()
        rendering_np = rendering_cpu.reshape(D, -1).T.numpy() # (H*W, D)
        
        pca = PCA(n_components=3)

        combined_np = np.concatenate((gt_np, rendering_np), axis=0)
        combined_features = pca.fit_transform(combined_np) 
        normalized_features = (combined_features - combined_features.min(axis=0)) / (combined_features.max(axis=0) - combined_features.min(axis=0))
        reshaped_combined_features = normalized_features.reshape(2, H, W, 3)
        
        reduced_rendering = reshaped_combined_features[1]
        reduced_gt = reshaped_combined_features[0]
        
        rendering = torch.tensor(reduced_rendering).permute(2, 0, 1)
        gt = torch.tensor(reduced_gt).permute(2, 0, 1)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name ))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name))

def render_sets(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_level : int, config_path=None, max_feature_views=None, use_online_siglip=False, upsample_method="bilinear"):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        # The internal symlink setup inside the output dir ensures the Scene constructor initializes correctly
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        checkpoint = os.path.join(dataset.model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')
        print(f"Loading lifted feature vectors from: {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore_language_features(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        language_feature_dir = getattr(dataset, "lf_path", os.path.join(dataset.source_path, dataset.language_features_name))

        if not skip_train:
             train_views = scene.getTrainCameras()
             if use_online_siglip:
                 if max_feature_views is not None and max_feature_views > 0:
                     skip_step = max(1, len(train_views) // max_feature_views)
                     filtered_train_views = train_views[::skip_step]
                 else:
                     filtered_train_views = train_views
             else:
                 filtered_train_views = _filter_views_with_language_features(train_views, language_feature_dir, max_feature_views)
                 
             if len(filtered_train_views) == 0:
                 raise FileNotFoundError(f"No language feature views found/generated for train set.")
             print(f"Rendering {len(filtered_train_views)} train views.")
             render_set(dataset.model_path, "train", scene.loaded_iter, filtered_train_views, gaussians, pipeline, background, feature_level, language_feature_dir, config_path, use_online_siglip, upsample_method)

        if not skip_test:
             test_views = scene.getTestCameras()
             if use_online_siglip:
                 if max_feature_views is not None and max_feature_views > 0:
                     skip_step = max(1, len(test_views) // max_feature_views)
                     filtered_test_views = test_views[::skip_step]
                 else:
                     filtered_test_views = test_views
             else:
                 filtered_test_views = _filter_views_with_language_features(test_views, language_feature_dir, max_feature_views)
                 
             if len(filtered_test_views) == 0:
                 raise FileNotFoundError(f"No language feature views found/generated for test set.")
             print(f"Rendering {len(filtered_test_views)} test views.")
             render_set(dataset.model_path, "test", scene.loaded_iter, filtered_test_views, gaussians, pipeline, background, feature_level, language_feature_dir, config_path, use_online_siglip, upsample_method)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_feature_views", type=int, default=None)
    parser.add_argument("--online_siglip", action="store_true", help="Generate ground truth comparison views online")
    parser.add_argument("--config", type=str, default=None, help="Path to the model profile YAML configuration file")
    parser.add_argument("--upsample_method", type=str, default="bilinear", choices=["bilinear", "naf"], help="Method for upsampling SigLIP patches")
    
    # Preserve arguments across the config file loading step
    safe_args, _ = parser.parse_known_args()
    args = get_combined_args(parser)
    
    args.max_feature_views = safe_args.max_feature_views
    args.online_siglip = safe_args.online_siglip
    args.config = safe_args.config
    args.upsample_method = safe_args.upsample_method
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        opt.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.feature_level,
        config_path=args.config,
        max_feature_views=args.max_feature_views,
        use_online_siglip=args.online_siglip,
        upsample_method=args.upsample_method
    )