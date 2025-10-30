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
import sys
import torch
import math
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.arguments import PipelineParams


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Query a volume with voxelization.
    """
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def query_masked(
    pc: GaussianModel,
    center,
    nVoxel,          # (nx, ny, nz)
    sVoxel,          # (sx, sy, sz)  [world units / voxel]
    mask_npy,        # 形状 (nx, ny, nz) の 0/1 npy ファイル or ndarray
    pipe: PipelineParams,
    scaling_modifier=1.0,
):  
    import numpy as np
    """
    マスク領域(1)に中心が入るGaussiansのみからvoxelを生成。
    """
    # ---- Voxelizer 設定 ----
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    # ---- 元のパラメタ取得 ----
    means3D   = pc.get_xyz            # (N,3) world coords
    density   = pc.get_density        # (N,1) or (N,)
    device    = means3D.device

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales    = pc.get_scaling
        rotations = pc.get_rotation

    # ---- マスクの読み込み（Tensor化）----
    if isinstance(mask_npy, str):
        mask_np = np.load(mask_npy)
    elif isinstance(mask_npy, np.ndarray):
        mask_np = mask_npy
    else:
        raise TypeError("mask_npy must be a path or numpy.ndarray of shape (nx, ny, nz)")
    assert mask_np.shape == (int(nVoxel[0]), int(nVoxel[1]), int(nVoxel[2])), \
        f"mask shape {mask_np.shape} != nVoxel {tuple(nVoxel)}"

    binary_mask = torch.from_numpy(mask_np.astype(np.uint8)).to(device=device)  # (nx,ny,nz)



    # ---- Gaussian中心をvoxelインデックスへ変換 (torch, on-device) ----
    # Convert nVoxel/sVoxel to torch tensors on the same device as means3D
    nV = torch.tensor(nVoxel, device=device, dtype=torch.float32)
    sV = torch.tensor(sVoxel, device=device, dtype=torch.float32)

    # means3D is already a torch tensor on `device`
    # compute voxel indices (floor via .long())
    means3D_ids = ((means3D / (sV / nV)) + (nV / 2.0)).long()  # (N,3) long on device

    # 範囲外を除外 (torch boolean mask)
    valid_mask = (
        (means3D_ids[:, 0] >= 0) & (means3D_ids[:, 0] < binary_mask.shape[0]) &
        (means3D_ids[:, 1] >= 0) & (means3D_ids[:, 1] < binary_mask.shape[1]) &
        (means3D_ids[:, 2] >= 0) & (means3D_ids[:, 2] < binary_mask.shape[2])
    )

    # Initialize keep_mask (False by default)
    keep_mask = torch.zeros((means3D.shape[0],), dtype=torch.bool, device=device)

    # Only index binary_mask with valid indices to avoid out-of-bounds GPU indexing
    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() > 0:
        ids_valid = means3D_ids[valid_indices]  # (M,3)
        vals = binary_mask[ids_valid[:, 0], ids_valid[:, 1], ids_valid[:, 2]]
        # Ensure boolean
        keep_mask[valid_indices] = vals.to(dtype=torch.bool)

    # invert keep_mask (we keep voxels where mask == 0)
    keep_mask = ~keep_mask
    # combine with valid_mask (both are on device)
    keep_mask &= valid_mask

    # debug prints
    print(keep_mask.shape)
    # filter ids/means using keep_mask
    means3D_ids = means3D_ids[keep_mask]
    if means3D_ids.numel() > 0:
        print(means3D_ids.max().cpu().item(), means3D_ids.min().cpu().item())
    print(keep_mask.sum().cpu().item())


    means3D_f = means3D[keep_mask]
    density_f = density[keep_mask]
    scales_f  = scales[keep_mask] if scales is not None else None
    rots_f    = rotations[keep_mask] if rotations is not None else None
    cov_f     = cov3D_precomp[keep_mask] if cov3D_precomp is not None else None

    # ---- voxelize（マスク後のGaussiansのみ）----
    vol_pred, radii = voxelizer(
        means3D=means3D_f,
        opacities=density_f,
        scales=scales_f,
        rotations=rots_f,
        cov3D_precomp=cov_f,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
