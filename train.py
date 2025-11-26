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
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian, query_masked
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.dataset.dataset_readers import angle2pose

from r2_gaussian.utils.loss_utils import ecc_loss_for_pair, l1_loss, l2_loss, ssim, tv_3d_loss, voxel_empty_loss, smoothness_loss_knn
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice
from r2_gaussian.utils.graphics_utils import fov2focal



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
        #if frame_idx % 2 == 0:
        if True:
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
            
            dtheta = math.radians(1.0)  # 1° = π/180 rad
            angle_minus = (curr_angle - dtheta) % (2 * math.pi)
            angle_plus  = (curr_angle + dtheta) % (2 * math.pi)


            angles = [angle_plus]
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

                angle_loss += l1_loss(image_shift, gt_image) * 0.01

            loss["angle_smoothnses"] = angle_loss
            loss["total"] += loss["angle_smoothnses"] 




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

            # lambda_smooth = 10000  # ハイパーパラメータ
            # loss["param_smooth"] = param_smooth * lambda_smooth
            # loss["total"] = loss["total"] + loss["param_smooth"]


            base_total = loss["total"].detach()
            # 「smoothness を total の何割くらいにしたいか」
            target_ratio = 0.1  # 例: 全体の 20% くらいの寄与にしたい

            # lambda_smooth を loss スケールから自動決定
            with torch.no_grad():
                ps = param_smooth.detach()
                if ps > 0:
                    raw_lambda = target_ratio * base_total / (ps + 1e-8)
                else:
                    raw_lambda = torch.tensor(0.0, device=param_smooth.device)

                # 極端な値はクリップ（必要に応じて調整）
                min_lambda = 1e-4
                max_lambda = 1e4
                raw_lambda = torch.clamp(raw_lambda, min_lambda, max_lambda)

            lambda_smooth = raw_lambda  # tensor のままで OK

            loss["param_smooth"] = param_smooth * lambda_smooth
            loss["total"] = loss["total"] + loss["param_smooth"]


        # ecc_loss_for_pair
        ecc_loss = True
        if ecc_loss:
            import math
            import matplotlib.pyplot as plt
            curr_angle = float(viewpoint_cam.angle)  # 現在角度（rad）
            
            dtheta = math.radians(3)  # 1° = π/180 rad
            angle_plus  = (curr_angle + dtheta) % (2 * math.pi)


            angles = [angle_plus]


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
                
                K = torch.tensor(
                    [
                        [fov2focal(viewpoint_cam.FoVx, gt_image[0].shape[1]), 0, gt_image[0].shape[1] / 2],
                        [0, fov2focal(viewpoint_cam.FoVy, gt_image[0].shape[0]), gt_image[0].shape[0] / 2],
                        [0, 0, 1],
                    ]
                ).to(device=gt_image.device, dtype=gt_image.dtype)
                ecc_loss_value = ecc_loss_for_pair(
                    img0=gt_image,
                    img1=image_shift,
                    P0=viewpoint_cam.world_view_transform.T, # w2c
                    P1=viewpoint_cam_angle_plus.world_view_transform.T, # w2c 
                    K = K,
                    c0=viewpoint_cam.camera_center,
                    c1=viewpoint_cam_angle_plus.camera_center,
                    points3d=torch.zeros((3,), device=gt_image.device),  # ダミー
                    global_iter=iteration,
                    curr_angle=curr_angle,
                )
                # print("ecc_loss_value:", ecc_loss_value,
                #     "requires_grad:", ecc_loss_value.requires_grad,
                #     "grad_fn:", ecc_loss_value.grad_fn)
                

                loss["ecc_loss"] = ecc_loss_value 
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
        
        # scale entropy loss
        use_scale_entropy = False # 全然よくない
        if use_scale_entropy and iter > 10000:
            densities = gaussians.get_density

            scale_scalar = torch.max(densities, dim=1).values  # (N,)
            eps = 1e-8
            scale_scalar = torch.clamp(scale_scalar, min=eps)
            # ヒストグラムをsoft binningで近似して、その分布のエントロピーを計算する
            # バケット中心をあらかじめ決める
            num_bins = 16
            s_min = torch.min(scale_scalar).detach()
            s_max = torch.max(scale_scalar).detach()
            bin_centers = torch.linspace(s_min, s_max + eps, steps=num_bins, device=scale_scalar.device)

            # 各点を各binに割り当てる確率 (Gaussian kernel でsoft assignment)
            # これでone-hotの代わりにスムーズな分布にするから勾配が通る
            bandwidth = 0.1 * (s_max - s_min + eps)  # カーネル幅
            diff = scale_scalar[:, None] - bin_centers[None, :]          # (N, num_bins)
            weights = torch.exp(-0.5 * (diff / (bandwidth + eps)) ** 2)  # Gaussian kernel
            # 正規化: 各点の重みが1になるように
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)


            # すべての点を足し合わせてヒストグラムっぽい分布に
            hist = torch.sum(weights, dim=0)  # (num_bins,)
            # 確率分布に正規化
            p = hist / (torch.sum(hist) + eps)  # (num_bins,)

            # 分布のエントロピー H = -Σ p log p
            entropy = -torch.sum(p * torch.log(p + eps)) 
            print(entropy)
            loss_scale_entropy = entropy / 10000
            loss["scale_entropy"] = loss_scale_entropy
            loss["total"] = loss["total"] + opt.lambda_tv * loss_scale_entropy



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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 20_000])
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
