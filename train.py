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
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian, query_masked, render_with_mask
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.dataset.dataset_readers import angle2pose

from r2_gaussian.utils.loss_utils import ecc_loss_for_pair, l1_loss, l2_loss, ssim, tv_3d_loss, voxel_empty_loss, smoothness_loss_knn, pseudo_gt_loss_step, normal_smoothness_loss_knn, smoothness_loss_knn_layered_allpairs, compute_layer_indices_from_z, smoothness_loss_knn_consensus
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice
from r2_gaussian.utils.graphics_utils import fov2focal
from r2_gaussian.utils.loss_utils import generate_random_RT_pairs_pm1deg, generate_random_RT_pairs_translate_only, ray_entropy_loss_from_camera, tv_l1

def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1



    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )


        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()

        frame_idx = int(viewpoint_cam.uid)

        loss = {"total": 0.0}
        if True:
        #if iteration %2 == 0:
            render_loss = l1_loss(image, gt_image)
            loss["render"] = render_loss
            loss["total"] += loss["render"]
            if opt.lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(image, gt_image)
                loss["dssim"] = loss_dssim
                loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
            if use_tv:
                # Randomly get the tiny volume center
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                    bbox[1] - tv_vol_sVoxel - bbox[0]
                ) * torch.rand(3)
                vol_pred = query(
                    gaussians,
                    tv_vol_center,
                    tv_vol_nVoxel,
                    tv_vol_sVoxel,
                    pipe,
                )["vol"]
                loss_tv = tv_3d_loss(vol_pred, reduction="mean")
                loss["tv"] = loss_tv
                loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
        
        # smoothness loss
        smoothness = False
        if smoothness:
            import math
            import matplotlib.pyplot as plt
    
            curr_angle = float(viewpoint_cam.angle)  # 現在角度（rad）
            
            dtheta = math.radians(1)  # 1° = π/180 rad
            angle_minus = (curr_angle - dtheta) % (2 * math.pi)
            angle_plus  = (curr_angle + dtheta) % (2 * math.pi)


            angles = [angle_plus, angle_minus]
            angle_loss = 0.0

            # CT 'transform_matrix' is a camera-to-world transform

            for angle_plus in angles:
                c2w = angle2pose(5, angle_plus)  # c2w
                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(
                    w2c[:3, :3]
                )  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                viewpoint_cam_angle_plus = Camera(
                    colmap_id=viewpoint_cam.colmap_id,
                    scanner_cfg=None,
                    R=R,
                    T=T,
                    angle=angle_plus,
                    mode=viewpoint_cam.mode,
                    FoVx=viewpoint_cam.FoVx,
                    FoVy=viewpoint_cam.FoVy,
                    image=torch.zeros((1, 512, 512)),
                    image_name="none",
                    uid=1,
                )

                # --- ここから可視化用 ---
                render_pkg = render(viewpoint_cam_angle_plus, gaussians, pipe)
                image_shift, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                # --------- マスク付き L1 loss ---------
                #image_shift と gt_image の shape を合わせておくこと（例: (1,1,H,W) → squeeze など）
                # diff = torch.abs(image_shift - gt_image) * mask
                # masked_l1 = diff.sum() / (mask.sum() + 1e-6)
                # angle_loss += masked_l1
                angle_loss += l1_loss(image_shift, gt_image) * 0.1

            loss["angle_smoothnses"] = angle_loss 
            loss["total"] += loss["angle_smoothnses"] 

        # random_smoothness
        random_smoothness = False
        edge_aware = False  # ← ここでON/OFF切り替え # Falseのほうがいい

        from PIL import Image
        import matplotlib.pyplot as plt


        if random_smoothness:
            import math
            import random
            import torch.nn.functional as F

            curr_angle = float(viewpoint_cam.angle)
            delta_deg = 1     # 🔹0.1度刻みに変更
            delta_rad = math.radians(delta_deg)
            random_smooth_loss = 0

            for i in range(3): # sampling数が多いほうがよい？
                # 0~60°までのrandomな値
                base_deg = random.randint(-19, 19)

                angle_minus = (curr_angle + math.radians(base_deg)) % (2 * math.pi)
                angle_plus  = (curr_angle + math.radians(base_deg + delta_deg)) % (2 * math.pi)

                images_shifts = []
                for angle in [angle_minus, angle_plus]:
                    c2w = angle2pose(5, angle)
                    w2c = np.linalg.inv(c2w)
                    R = np.transpose(w2c[:3, :3])
                    T = w2c[:3, 3]

                    viewpoint_cam_angle = Camera(
                        colmap_id=viewpoint_cam.colmap_id,
                        scanner_cfg=None,
                        R=R,
                        T=T,
                        angle=angle,
                        mode=viewpoint_cam.mode,
                        FoVx=viewpoint_cam.FoVx,
                        FoVy=viewpoint_cam.FoVy,
                        image=torch.zeros((1, 512, 512)),
                        image_name="none",
                        uid=1,
                    )
                    render_pkg = render(viewpoint_cam_angle, gaussians, pipe)
                    image_shift = render_pkg["render"]
                    if image_shift.ndim == 3:
                        image_shift = image_shift.unsqueeze(1)  # (B,1,H,W)
                    images_shifts.append(image_shift)

                img0, img1 = images_shifts

                # ===================================================
                # 🟦 Edge-aware / Non edge-aware 切り替え
                # ===================================================
                if edge_aware:
                    # ---- edge-aware weight 作成 ----
                    with torch.no_grad():
                        base_img = 0.5 * (img0 + img1)
                        dx = base_img[..., :, 1:] - base_img[..., :, :-1]
                        dy = base_img[..., 1:, :] - base_img[..., :-1, :]
                        dx = F.pad(dx, (0, 1, 0, 0))  # (B,1,H,W)
                        dy = F.pad(dy, (0, 0, 0, 1))

                        grad_mag = torch.sqrt(dx * dx + dy * dy + 1e-6)
                        gmax = grad_mag.max()
                        if gmax > 0:
                            grad_norm = grad_mag / gmax
                        else:
                            grad_norm = grad_mag

                        alpha = 5.0  # エッジ抑制強度
                        weight = torch.exp(-alpha * grad_norm).detach()

                    diff = torch.abs(img0 - img1) * weight
                    random_smooth_loss = diff.mean()

                else:
                    # ---- 通常のL1 smoothness ----
                    # abs_dP = torch.abs(img0 - img1)
                    # # threshold
                    # tau = 0.01  # 好きな値に調整

                    # # --- soft threshold weight ---
                    # weight = torch.clamp(1.0 - abs_dP / tau, min=0.0, max=1.0)
                    # weight = weight.detach()
                    # random_smooth_loss = weight * abs_dP
                    # # abs_dP を保存（curr_angle を渡す）
                    # if iteration % 100 == 0:
                    #     save_abs_dP_hist(abs_dP, curr_angle, iteration, save_dir="./debug_abs_no_loss")



                    random_smooth_loss += torch.mean(torch.abs(img0 - img1))

            # ===================================================
            loss["random_smoothness"] = random_smooth_loss.mean() * 0.1
            loss["total"] += loss["random_smoothness"]

        random_sample_smoothness = False

        
        if random_sample_smoothness:
            import math
            import random
            import torch.nn.functional as F
            import copy
                        
            xyz = gaussians.get_xyz 
            #grad_mask = (xyz >= -0.5).all(dim=1) & (xyz <= 0.5).all(dim=1) 
            def gaussian_blur2d(img: torch.Tensor, sigma: float = 1.5, ksize: int = 9):
                """
                img: (H,W) or (B,1,H,W) or (B,C,H,W)
                return: same shape
                """
                if img.dim() == 2:
                    img = img[None, None, ...]
                elif img.dim() == 3:
                    img = img[:, None, ...]  # (B,1,H,W)

                B, C, H, W = img.shape
                device = img.device
                dtype = img.dtype

                # 1D Gaussian kernel
                x = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
                g = torch.exp(-(x**2) / (2 * sigma**2))
                g = g / g.sum()

                # separable conv: horizontal then vertical
                g_x = g.view(1, 1, 1, ksize).repeat(C, 1, 1, 1)
                g_y = g.view(1, 1, ksize, 1).repeat(C, 1, 1, 1)

                pad = ksize // 2
                out = F.conv2d(img, g_x, padding=(0, pad), groups=C)
                out = F.conv2d(out, g_y, padding=(pad, 0), groups=C)
                return out

            def save_tensor_as_png(x: torch.Tensor, path: str, clamp_percentile: float = 0.0):
                """
                x: (H,W) or (1,H,W) or (B,H,W) - float tensor
                保存前に0-255へ正規化（可視化用）
                """
                import imageio.v2 as imageio
                os.makedirs(os.path.dirname(path), exist_ok=True)

                # 1枚だけにする
                if x.dim() == 3:
                    x = x[0]
                x = x.detach().float().cpu()

                # 可視化用に正規化（外れ値があると潰れるので任意でpercentileカット）
                if clamp_percentile > 0.0:
                    lo = torch.quantile(x, clamp_percentile)
                    hi = torch.quantile(x, 1.0 - clamp_percentile)
                    x = x.clamp(lo.item(), hi.item())

                mn, mx = x.min(), x.max()
                if (mx - mn) < 1e-8:
                    img = torch.zeros_like(x)
                else:
                    img = (x - mn) / (mx - mn)

                img_u8 = (img * 255.0).round().to(torch.uint8).numpy()
                imageio.imwrite(path, img_u8)


                        
            random_sample_smoothness_loss = 0
            R0, T0, R1, T1 = generate_random_RT_pairs_pm1deg(
                iteration,
                n_poses=3,
                bbox_min=-5.0,
                bbox_max= 5.0,
                up=np.array([0., 0., 1.], dtype=np.float32),
                seed=None,
            )
            
            
            for i in range(len(R0)):
                viewpoint_cam_0 = Camera(
                    colmap_id=viewpoint_cam.colmap_id,
                    scanner_cfg=None,
                    R=R0[i],
                    T=T0[i],
                    angle=viewpoint_cam.angle,
                    mode=viewpoint_cam.mode,
                    FoVx=viewpoint_cam.FoVx,
                    FoVy=viewpoint_cam.FoVy,
                    image=torch.zeros((1, 512, 512)),
                    image_name="none",
                    uid=1,
                )
                 

                render_cam_0 = render(viewpoint_cam_0, gaussians, pipe)["render"]
                #render_cam_0 = render_with_mask(viewpoint_cam_0, gaussians, pipe, grad_mask=grad_mask)["render"]
                viewpoint_cam_1 = Camera(
                    colmap_id=viewpoint_cam.colmap_id,
                    scanner_cfg=None,
                    R=R1[i],
                    T=T1[i],
                    angle=viewpoint_cam.angle,
                    mode=viewpoint_cam.mode,
                    FoVx=viewpoint_cam.FoVx,
                    FoVy=viewpoint_cam.FoVy,
                    image=torch.zeros((1, 512, 512)),
                    image_name="none",
                    uid=1,
                )
                render_cam_1 = render(viewpoint_cam_1, gaussians, pipe)["render"]
                #render_cam_1 = render_with_mask(viewpoint_cam_1, gaussians, pipe, grad_mask=grad_mask)["render"]

                H, W = 512, 512
                i, j = H // 2, W // 2

                blur_render_cam_0 = gaussian_blur2d(render_cam_0, sigma=1.5, ksize=9)
                blur_render_cam_1 = gaussian_blur2d(render_cam_1, sigma=1.5, ksize=9)
                blur_gt_image = gaussian_blur2d(gt_image, sigma=1.5, ksize=9)
                hf_gt_image = gt_image - blur_gt_image

                hf_cam_0 = render_cam_0 - blur_render_cam_0
                hf_cam_1 = render_cam_1 - blur_render_cam_1
                if iteration < 1000:
                    random_sample_smoothness_loss += torch.abs(render_cam_0[..., :, :]  - render_cam_1[..., :, :]).mean()
                else:
                    random_sample_smoothness_loss += torch.abs(blur_render_cam_0[..., :, :]  - blur_render_cam_1[..., :, :]).mean()
            
                # if (iteration % 100) == 0:
                #     out_dir = os.path.join(scene.model_path, "hf_smooth")
                #     # save_tensor_as_png(gt_image[0],  os.path.join(out_dir, f"{iteration:06d}_gt.png"), clamp_percentile=0.01)
                #     # save_tensor_as_png(blur_gt_image[0],  os.path.join(out_dir, f"{iteration:06d}_blur.png"), clamp_percentile=0.01)
                #     # save_tensor_as_png(hf_gt_image[0], os.path.join(out_dir, f"{iteration:06d}_hf.png"), clamp_percentile=0.01)
                #     save_tensor_as_png(render_cam_0[0],  os.path.join(out_dir, f"{iteration:06d}_render0.png"), clamp_percentile=0.01)
                #     save_tensor_as_png(blur_render_cam_0[0],  os.path.join(out_dir, f"{iteration:06d}_blur0.png"), clamp_percentile=0.01)
                #     save_tensor_as_png(hf_cam_0[0], os.path.join(out_dir, f"{iteration:06d}_hf0.png"), clamp_percentile=0.01)
                


            loss["random_sample_smoothness"] = random_sample_smoothness_loss * 0.1
              
            loss["total"] += loss["random_sample_smoothness"]



        random_sample_smoothness_translate = False # なぜかangularのほうがうまくいく # gaussianの学習中に生じるノイズがtv lossでは、消すことができない
        if random_sample_smoothness_translate:
            # ===================================================

            random_sample_smoothness_loss = 0
            R0, T0, R1, T1 = generate_random_RT_pairs_translate_only(
                n_poses=3,
                bbox_min=-5.0,
                bbox_max= 5.0,
            )
            
            
            for i in range(len(R0)):
                viewpoint_cam_0 = Camera(
                    colmap_id=viewpoint_cam.colmap_id,
                    scanner_cfg=None,
                    R=R0[i],
                    T=T0[i],
                    angle=viewpoint_cam.angle,
                    mode=viewpoint_cam.mode,
                    FoVx=viewpoint_cam.FoVx,
                    FoVy=viewpoint_cam.FoVy,
                    image=torch.zeros((1, 512, 512)),
                    image_name="none",
                    uid=1,
                )

                render_cam_0 = render(viewpoint_cam_0, gaussians, pipe)["render"]
                viewpoint_cam_1 = Camera(
                    colmap_id=viewpoint_cam.colmap_id,
                    scanner_cfg=None,
                    R=R1[i],
                    T=T1[i],
                    angle=viewpoint_cam.angle,
                    mode=viewpoint_cam.mode,
                    FoVx=viewpoint_cam.FoVx,
                    FoVy=viewpoint_cam.FoVy,
                    image=torch.zeros((1, 512, 512)),
                    image_name="none",
                    uid=1,
                )
                render_cam_1 = render(viewpoint_cam_1, gaussians, pipe)["render"]

                random_sample_smoothness_loss += torch.abs(render_cam_0 - render_cam_1)


            #print(random_sample_smoothness_loss.mean())
            loss["random_sample_smoothness"] = random_sample_smoothness_loss.mean() * 0.01
            loss["total"] += loss["random_sample_smoothness"]




        smoothness_dP_dtheta = False

        if smoothness_dP_dtheta:
            import math
            import random
            import torch.nn.functional as F

            curr_angle = float(viewpoint_cam.angle)

            # ---- ランダムな基準角度 & Δθ ----
            base_deg = random.randint(0, 59)   # 0〜59°
            delta_deg = 1                    # 1° 間隔
            delta_rad = math.radians(delta_deg)

            # 3 つの角度: θ0, θ1 = θ0+Δ, θ2 = θ0+2Δ
            angles = [
                (curr_angle + math.radians(base_deg + 0 * delta_deg)) % (2 * math.pi),
                (curr_angle + math.radians(base_deg + 1 * delta_deg)) % (2 * math.pi),
                (curr_angle + math.radians(base_deg + 2 * delta_deg)) % (2 * math.pi),
            ]

            images_shifts = []
            for angle in angles:
                c2w = angle2pose(5, angle)
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]

                viewpoint_cam_angle = Camera(
                    colmap_id=viewpoint_cam.colmap_id,
                    scanner_cfg=None,
                    R=R,
                    T=T,
                    angle=angle,
                    mode=viewpoint_cam.mode,
                    FoVx=viewpoint_cam.FoVx,
                    FoVy=viewpoint_cam.FoVy,
                    image=torch.zeros((1, 512, 512)),
                    image_name="none",
                    uid=1,
                )
                render_pkg = render(viewpoint_cam_angle, gaussians, pipe)
                image_shift = render_pkg["render"]
                if image_shift.ndim == 3:
                    image_shift = image_shift.unsqueeze(1)  # (B,1,H,W)
                images_shifts.append(image_shift)

            img0, img1, img2 = images_shifts  # P(θ0), P(θ1), P(θ2)

            # ===================================================
            # 🔹 dP/dθ を 2 点で計算して、その差分を L1 で抑える
            #     dP0 ≈ (P1 - P0)/Δθ
            #     dP1 ≈ (P2 - P1)/Δθ
            # ===================================================
            dP0 = (img1 - img0) / delta_rad
            dP1 = (img2 - img1) / delta_rad

            diff_dP = torch.abs(dP1 - dP0)  # = 離散二階微分に対応
            random_smooth_loss = diff_dP.mean()

            loss["smoothness_dP_dtheta"] = random_smooth_loss * 0.1
            loss["total"] += loss["smoothness_dP_dtheta"]

            # ================================
            # iteration 100毎に PNG 保存
            # ================================
            # if iteration % 100 == 0:
            #     smooth_save_path = osp.join(scene.model_path, "smooth")
            #     os.makedirs(smooth_save_path, exist_ok=True)

            #     out_path = osp.join(smooth_save_path, f"iter_{iteration:06d}_angle_plus.png")
            #     plt.imsave(out_path, img_plus, cmap="gray")

            #     # ログ
            #     print(f"[iter {iteration}] Saved:", out_path)
        
        # localization loss
        # https://chatgpt.com/s/t_691ad54ee9008191926ae297404dd944
        gaussian_localization = False
        # 細部の構造が失われないよう
        if gaussian_localization:
            xyz     = gaussians.get_xyz[:, :3]
            scales  = gaussians.get_scaling[:, :3]
            opacity = gaussians.get_density[:, 0]
            opacity = opacity.view(opacity.shape[0], -1)  # (N,1)
            theta = torch.cat([opacity], dim=-1)

            param_smooth = smoothness_loss_knn(
                xyz=xyz,
                theta=theta,
            )
     

            # xyz: (N, 3), theta: (N, D)

            loss["param_smooth"] = param_smooth * 1000

            loss["total"] = loss["total"] + loss["param_smooth"]

        normal_smoothness_knn = False
        if normal_smoothness_knn:
            xyz     = gaussians.get_xyz[:, :3]
            scales  = gaussians.get_scaling[:, :3]
            rotation = gaussians.get_rotation[:, :4]
            opacity = gaussians.get_density[:, 0]
            opacity = opacity.view(opacity.shape[0], -1)  # (N,1)
            theta = torch.cat([opacity], dim=-1)


            normal_smooth_loss = normal_smoothness_loss_knn(
                xyz=xyz,
                rotation=rotation,
                scale=scales
            )

            loss["normal_smooth_loss"] = normal_smooth_loss
            loss["total"] += loss["normal_smooth_loss"] 



        # ecc_loss_for_pair
        ecc_loss = False
        if ecc_loss:
            import math
            import matplotlib.pyplot as plt
            curr_angle = float(viewpoint_cam.angle)  # 現在角度（rad）
            
            dtheta = math.radians(3)  # 1° = π/180 rad
            angle_plus  = (curr_angle + dtheta) % (2 * math.pi)


            #angles = [angle_plus]


            # CT 'transform_matrix' is a camera-to-world transform
            loss["ecc_loss"] = 0.0
            c2w = angle2pose(5, angle_plus)  # c2w
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            viewpoint_cam_angle_plus = Camera(
                colmap_id=viewpoint_cam.colmap_id,
                scanner_cfg=None,
                R=R,
                T=T,
                angle=angle_plus,
                mode=viewpoint_cam.mode,
                FoVx=viewpoint_cam.FoVx,
                FoVy=viewpoint_cam.FoVy,
                image=torch.zeros((1, 512, 512)),
                image_name="none",
                uid=1,
            )

            # --- ここから可視化用 ---
            render_pkg = render(viewpoint_cam_angle_plus, gaussians, pipe)
            image_shift, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            
            K = torch.tensor(
                [
                    [fov2focal(viewpoint_cam.FoVx, gt_image[0].shape[1]), 0, gt_image[0].shape[1] / 2],
                    [0, fov2focal(viewpoint_cam.FoVy, gt_image[0].shape[0]), gt_image[0].shape[0] / 2],
                    [0, 0, 1],
                ]
            ).to(device=gt_image.device, dtype=gt_image.dtype)

            for i in range(1):
                ecc_loss_value = ecc_loss_for_pair(
                    img0=gt_image,
                    img1=image_shift,
                    P0=viewpoint_cam.world_view_transform.T, # w2c
                    P1=viewpoint_cam_angle_plus.world_view_transform.T, # w2c 
                    K = K,
                    c0=viewpoint_cam.camera_center,
                    c1=viewpoint_cam_angle_plus.camera_center,
                    points3d=torch.zeros((3,), device=gt_image.device),  # ダミー)
                    #points3d = torch.rand((3,), device=gt_image.device),
                    global_iter=iteration,
                    curr_angle=curr_angle,
                )
            
                loss["ecc_loss"] += ecc_loss_value 
            loss["total"] += loss["ecc_loss"] * 0.01
        symmetry_loss =False
        # https://chatgpt.com/c/691ad40f-32e4-8324-97fe-a1b455f5d86f
        if symmetry_loss and iteration < 10000:
            curr_angle = float(viewpoint_cam.angle)  # 現在角度（rad）
            
            dtheta = math.radians(180)  # 1° = π/180 rad
            flip_angle = (curr_angle + dtheta) % (2 * math.pi)

            # CT 'transform_matrix' is a camera-to-world transform
            c2w = angle2pose(5, flip_angle)  # c2w
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            viewpoint_cam_angle_flip = Camera(
                colmap_id=viewpoint_cam.colmap_id,
                scanner_cfg=None,
                R=R,
                T=T,
                angle=flip_angle,
                mode=viewpoint_cam.mode,
                FoVx=viewpoint_cam.FoVx,
                FoVy=viewpoint_cam.FoVy,
                image=torch.zeros((1, 512, 512)),
                image_name="none",
                uid=1,
            )

            # --- ここから可視化用 ---
            render_pkg = render(viewpoint_cam_angle_flip, gaussians, pipe)
            image, _, _, _ = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            img_plus = render_pkg["render"][0].detach().cpu().numpy()
            # X線なら [1,1,H,W] or [1,H,W,1] の可能性があるので squeeze
            img_plus = np.squeeze(img_plus)


            img_theta_flipped = torch.flip(image, dims=[-1])

            loss["symmetry"] = l1_loss(img_theta_flipped, gt_image)
            loss["total"] = loss["total"] + loss["symmetry"] * 0.1
        

        # sinograom loss
        sinogram = False
        if sinogram:
            from pathlib import Path
            interp_pred_root = Path("/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_38_angle_360/1_pepper_cone/proj_train_interp_epipolar")

            def load_interp_pred(frame_index: int) -> torch.Tensor:
                """保存済みの補間画像 pred を読み込んで torch.Tensor[1, H, W] で返す"""
                p = interp_pred_root / f"proj_pred_{frame_index:04d}.npy"
                arr = np.load(p).astype(np.float32)
                ten = torch.from_numpy(arr)[None].cuda()  # [1,H,W] に合わせる（あなたのrender出力に合わせて次元は調整）
                return ten

            
            lam_interp = 0.1  # 補間GTロスの重み（0.05〜0.2から開始がおすすめ）
            # ---- 追加：補間GTとの L1 ----
            # 例: viewpoint_cam.frame_index (0..74) が取れる場合
            # 偶数でも奇数でも「補間版」を用意して比較したいなら常にOK。
            # 奇数だけに限定したいなら: 
            if frame_idx % 2 == 1:
                interp_gt = load_interp_pred(frame_idx)      # [1,H,W] を想定
                loss_interp = l1_loss(image, interp_gt)      # 形状が合わない場合は squeeze/unsqueeze 調整

                loss["interp"] = lam_interp * loss_interp
                loss["total"] += loss["interp"]

                # if opt.lambda_dssim > 0:
                # loss_dssim = 1.0 - ssim(image, interp_gt)
                # loss["dssim"] = loss_dssim
                # loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim



        # 3D depth loss
        use_depth = False
        if use_depth:
            # Randomly get the tiny volume center
            tv_vol_center = torch.tensor([0, 0, 0])
            tv_vol_nVoxel = torch.tensor([256, 256, 256])
            tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel
            mask_npy = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_3_angle_360/teapot/vol_binary.npy"
            vol_pred = query_masked(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                mask_npy,
                pipe,
            )["vol"]
            #loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss_depth = voxel_empty_loss(vol_pred) / 1000000
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_depth



        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1_000, 5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1_000, 5_000, 10_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
