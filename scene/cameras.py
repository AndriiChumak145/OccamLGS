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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import os
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 cam_info=None, args=None, resolution_scale=1.0, is_nerf_synthetic=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        
        self.cam_info = cam_info
        self.args = args
        self.resolution = resolution
        self.resolution_scale = resolution_scale
        self.is_nerf_synthetic = is_nerf_synthetic
        self.train_test_exp = train_test_exp
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view
        self.depth_params = depth_params

        self.original_image = None
        self.alpha_mask = None
        self.invdepthmap = None
        self.depth_mask = None
        self.depth_reliable = False

        self.lazy_load = getattr(args, "lazy_load", False)
        if not self.lazy_load:
            self.load_image_data()

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        self.x = None
        self.y = None

    def load_image_data(self):
        if self.original_image is not None:
            return

        image = Image.open(self.cam_info.image_path)
        mask_paths = getattr(self.cam_info, "mask_paths", None)
        mask_rgb = getattr(self.args, "mask_rgb", False)
        if mask_rgb:
            if not mask_paths:
                raise FileNotFoundError(f"mask_rgb is enabled but no mask paths were provided for {self.cam_info.image_name}.")
            mask_union = None
            for mask_path in mask_paths:
                mask_img = Image.open(mask_path).convert("L")
                mask_arr = np.array(mask_img) > 0
                if mask_union is None:
                    mask_union = mask_arr
                else:
                    mask_union |= mask_arr
            if mask_union is None:
                raise ValueError(f"No valid masks loaded for {self.cam_info.image_name}.")
            if mask_union.shape[0] != image.size[1] or mask_union.shape[1] != image.size[0]:
                raise ValueError(f"Mask size does not match image size.")
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            bg = np.array([1, 1, 1], dtype=np.float32) if self.args.white_background else np.array([0, 0, 0], dtype=np.float32)
            rgb = rgb * mask_union[..., None] + bg * (1.0 - mask_union[..., None])
            alpha = (mask_union.astype(np.uint8) * 255)[..., None]
            rgba = np.concatenate([rgb * 255.0, alpha], axis=-1).astype(np.uint8)
            image = Image.fromarray(rgba, "RGBA")

        invdepthmap = None
        if self.cam_info.depth_path != "":
            if self.is_nerf_synthetic:
                invdepthmap = cv2.imread(self.cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(self.cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        resized_image_rgb = PILtoTorch(image, self.resolution)
        gt_image = resized_image_rgb[:3, ...]
        
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if self.train_test_exp and self.is_test_view:
            if self.is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, self.resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if self.depth_params is not None:
                if self.depth_params["scale"] < 0.2 * self.depth_params["med_scale"] or self.depth_params["scale"] > 5 * self.depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if self.depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * self.depth_params["scale"] + self.depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

    def unload_image_data(self):
        if not getattr(self, "lazy_load", False):
            return
        self.original_image = None
        self.alpha_mask = None
        self.invdepthmap = None
        self.depth_mask = None

        
    def get_language_feature(self, language_feature_dir, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name.split('.')[0])
        
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy')).cuda()
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy')).cuda()
        
        # Implicit, based on tensor dims
        # New dense format (e.g., SigLIP [1536, H, W])
        if feature_map.dim() == 3:
            # We now trust the pre-computed _s file completely
            if seg_map.dim() == 2:
                seg_map = seg_map.unsqueeze(0)
                
            mask = seg_map > 0.5 
            return feature_map, mask

        # Original format (CLIP Dictionary [N, 512])
        else:
            if self.x is None or self.y is None:
                y, x = torch.meshgrid(torch.arange(0, self.image_height, device='cuda'), torch.arange(0, self.image_width, device='cuda'))
                self.x = x.reshape(-1, 1)
                self.y = y.reshape(-1, 1)
                
            seg = seg_map[:, self.y, self.x].squeeze(-1).long()
            mask = seg != -1
            # print(f"Shapes of seg_map: {seg_map.shape}, feature_map: {feature_map.shape} for camera {self.image_name}, seg shape: {seg.shape}, mask shape: {mask.shape}")

            if feature_level == 0: # default
                point_feature1 = feature_map[seg[0:1]].squeeze(0)
                mask = mask[0:1].reshape(1, self.image_height, self.image_width)
            elif feature_level == 1: # s
                point_feature1 = feature_map[seg[1:2]].squeeze(0)
                mask = mask[1:2].reshape(1, self.image_height, self.image_width)
            elif feature_level == 2: # m
                point_feature1 = feature_map[seg[2:3]].squeeze(0)
                mask = mask[2:3].reshape(1, self.image_height, self.image_width)
            elif feature_level == 3: # l
                point_feature1 = feature_map[seg[3:4]].squeeze(0)
                mask = mask[3:4].reshape(1, self.image_height, self.image_width)
            else:
                raise ValueError("feature_level=", feature_level)
            
            point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        
            return point_feature, mask
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

