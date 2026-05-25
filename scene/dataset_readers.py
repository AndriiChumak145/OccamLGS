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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    mask_paths: list

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                      image_path=image_path, image_name=image_name, depth_path=depth_path,
                      width=width, height=height, is_test=image_name in test_cam_names_list,
                      mask_paths=None)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test,
                            mask_paths=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def _ycbv_collect_mask_paths(mask_dir: Path, frame_id: int, mask_instance_idx: int = -1) -> list:
    if mask_dir is None:
        raise FileNotFoundError("mask_dir is None; masks are required for YCBV scenes.")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask directory not found at {mask_dir}")

    paths = sorted(mask_dir.glob(f"{frame_id:06d}_*.png"))
    if not paths:
        raise FileNotFoundError(
            f"No mask files found in {mask_dir} for frame {frame_id:06d}."
        )

    if mask_instance_idx is None or mask_instance_idx < 0:
        return [str(p) for p in paths]

    filtered = []
    for p in paths:
        parts = p.stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Malformed mask filename: {p.name}")
        if not parts[1].isdigit():
            raise ValueError(f"Non-integer instance id in mask filename: {p.name}")
        if int(parts[1]) == mask_instance_idx:
            filtered.append(str(p))

    if not filtered:
        raise FileNotFoundError(
            f"No mask files found for instance {mask_instance_idx} in frame {frame_id:06d}."
        )

    return filtered

def _ycbv_select_mask_dir(ycbv_dir: Path, mask_source: str) -> Path:
    mask_dir = ycbv_dir / "mask"
    mask_visib_dir = ycbv_dir / "mask_visib"

    if mask_source is None or mask_source == "auto":
        if mask_visib_dir.exists():
            return mask_visib_dir
        if mask_dir.exists():
            return mask_dir
        raise FileNotFoundError(
            f"No mask or mask_visib directory found under {ycbv_dir}."
        )

    if mask_source == "mask":
        if mask_dir.exists():
            return mask_dir
        raise FileNotFoundError(f"mask directory not found at {mask_dir}")

    if mask_source == "mask_visib":
        if mask_visib_dir.exists():
            return mask_visib_dir
        raise FileNotFoundError(f"mask_visib directory not found at {mask_visib_dir}")

    raise ValueError(f"Unsupported mask_source: {mask_source}")

def _ycbv_backproject(depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray = None, stride: int = 1):
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    zs = depth_m[ys, xs].astype(np.float32)
    valid = zs > 0
    if mask is not None:
        valid = valid & mask[ys, xs]

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    points_cam = np.stack([x, y, zs], axis=1)
    pixels = np.stack([xs, ys], axis=1).astype(np.int32)
    return points_cam, pixels

def _ycbv_points_to_world(points_cam: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    return (R_w2c.T @ (points_cam.T - t_w2c.reshape(3, 1))).T

def _ycbv_load_mask_union(mask_paths: list, width: int, height: int) -> np.ndarray:
    if not mask_paths:
        raise FileNotFoundError("No mask paths provided for mask union.")
    mask_union = None
    for mask_path in mask_paths:
        mask_img = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask_img) > 0
        if mask_union is None:
            mask_union = mask_arr
        else:
            mask_union |= mask_arr
    if mask_union is None:
        return None
    if mask_union.shape[0] != height or mask_union.shape[1] != width:
        raise ValueError(
            f"Mask size {mask_union.shape[1]}x{mask_union.shape[0]} does not match image size {width}x{height}."
        )
    return mask_union

def readYCBVSceneInfo(
    path,
    images,
    depths,
    eval,
    train_test_exp,
    llffhold=8,
    mask_instance_idx: int = -1,
    mask_source: str = "auto",
):
    ycbv_dir = Path(path)
    scene_camera_path = ycbv_dir / "scene_camera.json"
    if not scene_camera_path.exists():
        raise FileNotFoundError(f"scene_camera.json not found at {scene_camera_path}")

    with open(scene_camera_path, "r") as f:
        scene_camera = json.load(f)

    rgb_dir = ycbv_dir / (images if images is not None else "rgb")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found at {rgb_dir}")

    depth_dir = None
    if depths:
        depth_dir = ycbv_dir / depths
        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found at {depth_dir}")
    else:
        depth_dir = ycbv_dir / "depth"
        if not depth_dir.exists():
            depth_dir = None

    mask_dir = _ycbv_select_mask_dir(ycbv_dir, mask_source)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")

    frame_ids = [int(p.stem) for p in rgb_files]
    test_frame_ids = []
    if eval and llffhold:
        test_frame_ids = [fid for idx, fid in enumerate(frame_ids) if idx % llffhold == 0]

    first_image = Image.open(rgb_files[0])
    width, height = first_image.size

    cam_infos = []
    for idx, frame_id in enumerate(frame_ids):
        cam = scene_camera.get(str(frame_id))
        if cam is None:
            raise KeyError(f"scene_camera.json missing entry for frame {frame_id:06d}")

        K = np.array(cam["cam_K"], dtype=np.float32).reshape(3, 3)
        R_w2c = np.array(cam["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
        t_w2c = np.array(cam["cam_t_w2c"], dtype=np.float32).reshape(3)

        R = R_w2c.T
        T = t_w2c * 0.001

        fx = K[0, 0]
        fy = K[1, 1]
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        image_path = str(rgb_dir / f"{frame_id:06d}.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"RGB image not found at {image_path}")
        image_name = f"{frame_id:06d}.png"
        mask_paths = _ycbv_collect_mask_paths(mask_dir, frame_id, mask_instance_idx=mask_instance_idx)

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                depth_params=None,
                image_path=image_path,
                image_name=image_name,
                depth_path="",
                width=width,
                height=height,
                is_test=frame_id in test_frame_ids,
                mask_paths=mask_paths,
            )
        )

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "ycbv_points3d.ply")
    if not os.path.exists(ply_path):
        if depth_dir is None or not depth_dir.exists():
            raise FileNotFoundError(
                "Depth directory is required to build the YCBV point cloud, but was not found."
            )
        points_all = []
        colors_all = []
        frame_stride = 5
        point_stride = 4
        max_points = 200_000
        for frame_id in frame_ids[::frame_stride]:
            cam = scene_camera.get(str(frame_id))
            if cam is None:
                raise KeyError(f"scene_camera.json missing entry for frame {frame_id:06d}")

            depth_path = depth_dir / f"{frame_id:06d}.png"
            rgb_path = rgb_dir / f"{frame_id:06d}.png"
            if not depth_path.exists():
                raise FileNotFoundError(f"Depth image not found at {depth_path}")
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB image not found at {rgb_path}")

            depth_raw = np.array(Image.open(depth_path))
            if depth_raw.ndim != 2:
                raise ValueError(
                    f"Depth image at {depth_path} is not single-channel."
                )

            depth_scale = float(cam.get("depth_scale", 1.0))
            depth_m = depth_raw.astype(np.float32) * depth_scale * 0.001

            K = np.array(cam["cam_K"], dtype=np.float32).reshape(3, 3)
            R_w2c = np.array(cam["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
            t_w2c = np.array(cam["cam_t_w2c"], dtype=np.float32).reshape(3) * 0.001

            mask_paths = _ycbv_collect_mask_paths(mask_dir, frame_id, mask_instance_idx=mask_instance_idx)
            mask_union = _ycbv_load_mask_union(mask_paths, depth_m.shape[1], depth_m.shape[0])

            points_cam, pixels = _ycbv_backproject(depth_m, K, mask=mask_union, stride=point_stride)
            if points_cam.shape[0] == 0:
                raise ValueError(f"No valid depth points for frame {frame_id:06d}")

            points_world = _ycbv_points_to_world(points_cam, R_w2c, t_w2c)
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            colors = rgb[pixels[:, 1], pixels[:, 0]].astype(np.float32) / 255.0

            points_all.append(points_world)
            colors_all.append(colors)

        if not points_all:
            raise ValueError("No points collected for YCBV point cloud generation.")

        points_all = np.concatenate(points_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if points_all.shape[0] > max_points:
            rng = np.random.default_rng(42)
            keep_idx = rng.choice(points_all.shape[0], size=max_points, replace=False)
            points_all = points_all[keep_idx]
            colors_all = colors_all[keep_idx]
        pcd = BasicPointCloud(points=points_all, colors=colors_all, normals=np.zeros_like(points_all))
        storePly(ply_path, points_all, (colors_all * 255).astype(np.uint8))

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "YCBV": readYCBVSceneInfo
}