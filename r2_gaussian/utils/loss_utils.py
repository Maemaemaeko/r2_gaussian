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

import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
from pathlib import Path
import os
from r2_gaussian.utils.epinaf_loss import (
    plane_points_on_Ephi,
    epipolar_line_on_view,
    clip_line_to_image,
    sample_points_on_segment,
    compute_normal_derivative,
)


def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def voxel_empty_loss(vol):
    loss = torch.sum(vol)
    return loss


def compute_layer_indices_from_z(xyz, num_layers, z_min=None, z_max=None):
    """
    xyz: (N, 3)
    num_layers: レイヤー数（例：CTのスライス数や、任意に分割したい数）

    戻り値:
        layer_ids: (N,) int64, [0, num_layers-1]
    """
    device = xyz.device
    z = xyz[:, 2]

    if z_min is None:
        z_min = z.min()
    if z_max is None:
        z_max = z.max()

    if num_layers <= 1 or (z_max - z_min).abs() < 1e-8:
        return torch.zeros_like(z, dtype=torch.long, device=device)

    normalized = (z - z_min) / (z_max - z_min + 1e-8)
    layer_ids = torch.floor(normalized * num_layers).long()
    layer_ids = torch.clamp(layer_ids, 0, num_layers - 1)
    return layer_ids


def smoothness_loss_knn_layered_allpairs(
    xyz,
    theta,
    layer_ids,
    num_layers_sample=8,
    num_centers_per_layer=200,  # 各レイヤーからサンプルする center 数
    k=8,
    radius=None,                # 一定 radius 内だけで smoothness を計算したい場合 (None なら制限なし)
    sigma=0.03,
):
    """
    xyz:       (N, 3)
    theta:     (N, D)
    layer_ids: (N,) それぞれの点が属するレイヤーID (0,1,...,L-1)

    各レイヤーごとに:
      - レイヤー内から center をランダムサンプリング
      - レイヤー内の「全ての Gaussian」との距離を計算
      - その中から最近傍 k 個をとり、(距離 <= radius) だけで smoothness を計算
    """
    device = xyz.device
    N = xyz.shape[0]

    # ---- 全レイヤー数 ----
    L = int(layer_ids.max().item()) + 1

    # ---- レイヤーをランダムサンプル ----
    num_layers_sample = min(num_layers_sample, L)
    sampled_layers = torch.randperm(L, device=device)[:num_layers_sample]

    total_weighted_loss = xyz.new_tensor(0.0)
    total_weight = xyz.new_tensor(0.0)

    for layer in sampled_layers:
        # ---- このレイヤーに属する点 ----
        mask = (layer_ids == layer)
        idx_layer = mask.nonzero(as_tuple=True)[0]
        n_L = idx_layer.numel()

        if n_L <= 1:
            continue

        # ---- center サンプル ----
        num_centers = min(num_centers_per_layer, n_L)
        perm_local = torch.randperm(n_L, device=device)[:num_centers]

        center_idx_global = idx_layer[perm_local]   # (C,)
        center_xyz = xyz[center_idx_global]         # (C, 3)
        center_theta = theta[center_idx_global]     # (C, D)

        # ---- レイヤー内全点 ----
        xyz_layer = xyz[idx_layer]                  # (n_L, 3)
        theta_layer = theta[idx_layer]              # (n_L, D)


        # ---- 距離計算 (全点) ----
        pts_i = center_xyz.unsqueeze(1)             # (C, 1, 3)
        pts_j = xyz_layer.unsqueeze(0)              # (1, n_L, 3)

        # XY 距離 or 3D 距離
        #dists = (pts_i - pts_j).norm(dim=-1)       # 3D
        dists = (pts_i[..., :2] - pts_j[..., :2]).norm(dim=-1)  # XY距離

        # ---- 自己マスク ----
        self_mask = (center_idx_global.unsqueeze(1) == idx_layer.unsqueeze(0))
        dists = dists + self_mask * 1e6

        # ---- kNN ----
        k_eff = min(k, n_L - 1)
        knn_dists, knn_local_idx = torch.topk(dists, k_eff, dim=-1, largest=False)
        knn_idx_global = idx_layer[knn_local_idx]

        # ---- theta 差分 ----
        theta_i = center_theta.unsqueeze(1)
        theta_j = theta[knn_idx_global]
        diff = theta_i - theta_j
        sq = (diff * diff).sum(dim=-1)

        # ---- 距離重み ----
        weights = torch.exp(-(knn_dists ** 2) / (sigma ** 2))

        # ---- radius 制限 ----
        if radius is not None:
            radius_mask = (knn_dists <= radius)
            weights = weights * radius_mask

        weighted_sq_sum = (weights * sq).sum()
        weight_sum = weights.sum()

        if weight_sum > 0:
            total_weighted_loss += weighted_sq_sum
            total_weight += weight_sum

    if total_weight == 0:
        return xyz.new_tensor(0.0)

    return total_weighted_loss / total_weight
    

def smoothness_loss_knn(
    xyz,
    theta,
    num_centers=50000,  # loss 計算に使う Gaussians の数
    k=8,
    M=320,
    sigma=0.03,
):
    """
    xyz:   (N, 3)
    theta: (N, D)
    num_centers: smoothness の中心にする点の数（N からランダムサンプリング）
    k:     その中で最も近い近傍数
    M:     各 center あたりの候補点数（M >> k）
    """
    N = xyz.shape[0]
    device = xyz.device

    # N が小さいときは、そのまま全点で計算
    if N <= num_centers:
        num_centers = N

    # ---- 1) center となる点をランダムサンプリング ----
    # shape: (num_centers,)
    center_idx = torch.randperm(N, device=device)[:num_centers]
    center_xyz = xyz[center_idx]      # (num_centers, 3)
    center_theta = theta[center_idx]  # (num_centers, D)

    # ---- 2) 各 center に対して、ランダム候補 M 個を全体 N から選ぶ ----
    # shape: (num_centers, M)
    rand_idx = torch.randint(0, N, (num_centers, M), device=device)

    # ---- 3) 距離を計算 ----
    pts_i = center_xyz.unsqueeze(1)    # (num_centers, 1, 3)
    pts_j = xyz[rand_idx]              # (num_centers, M, 3)
    dists = (pts_i - pts_j).norm(dim=-1)  # (num_centers, M)
    #print(dists.max())

    # 自分自身を候補から消したい場合（オプション）
    # 同じ index が入っているとき、距離を大きくして弾く
    self_mask = (rand_idx == center_idx.unsqueeze(1))  # (num_centers, M)
    dists = dists + self_mask * 1e6

    # ---- 4) 一番近い k 個を選ぶ ----
    knn_dists, knn_local_idx = torch.topk(dists, k, dim=-1, largest=False)  # (num_centers, k)
    knn_idx = torch.gather(rand_idx, 1, knn_local_idx)                      # (num_centers, k)

    # ---- 5) theta の差分 ----
    theta_i = center_theta.unsqueeze(1)     # (num_centers, 1, D)
    theta_j = theta[knn_idx]               # (num_centers, k, D)
    diff = theta_i - theta_j               # (num_centers, k, D)
    sq = (diff * diff).sum(dim=-1)         # (num_centers, k)

    # ---- 6) 距離重みつき平均 ----
    weights = torch.exp(-(knn_dists ** 2) / (sigma ** 2))  # (num_centers, k)

    radius = 0.2
    radius_mask = (knn_dists <= radius).float()
    weights = weights * radius_mask
    sq = sq * radius_mask

    loss = (weights * sq).mean()
    return loss

import torch
import torch.nn.functional as F

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    q: (N, 4) quaternion
    ※ (w,x,y,z) 前提。実装が (x,y,z,w) なら並び替えてください。
    return: (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    N = q.shape[0]
    R = torch.zeros(N, 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def compute_gaussian_normals(rotation: torch.Tensor,
                             scale: torch.Tensor) -> torch.Tensor:
    """
    rotation: (N, 4)
    scale   : (N, 3)
    return  : (N, 3) unit normals
    """
    R = quaternion_to_rotation_matrix(rotation)  # (N,3,3)

    idx_min = torch.argmin(scale, dim=-1)       # (N,)
    N = rotation.shape[0]
    idx_batch = torch.arange(N, device=rotation.device)

    normals = R[idx_batch, :, idx_min]          # (N,3)
    normals = F.normalize(normals, dim=-1)
    return normals


def compute_gaussian_normals_xy(rotation: torch.Tensor,
                                scale: torch.Tensor):
    """
    rotation: (N, 4)
    scale   : (N, 3)
    return  : (N, 2)   XY 平面に投影した 2D normal（単位ベクトル）
    """
    # --- 3D の normal 計算 ---
    R = quaternion_to_rotation_matrix(rotation)  # (N,3,3)

    idx_min = torch.argmin(scale, dim=-1)        # (N,)  最も小さい軸を normal とする
    N = rotation.shape[0]
    idx_batch = torch.arange(N, device=rotation.device)

    normals_3d = R[idx_batch, :, idx_min]        # (N,3)
    normals_3d = F.normalize(normals_3d, dim=-1)

    # --- XY へ落とす ---
    normals_xy = normals_3d[:, :2]               # (N,2) → x,y 成分だけ抜く

    # --- XY 平面で正規化（長さ1の2Dベクトルに） ---
    normals_xy = F.normalize(normals_xy, dim=-1)

    return normals_xy


def normal_smoothness_loss_knn(
    xyz: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    num_centers: int = 1000,
    k: int = 8,
    M: int = 320,
    sigma: float = 0.03,
):
    """
    近傍 Gaussian 同士の normal 方向が揃うようにする loss
    （normal と -normal を同一視）
    """
    device = xyz.device
    N = xyz.shape[0]

    # ---- normal 計算 ----
    normals = compute_gaussian_normals(rotation, scale)  # (N,3)

    if N <= num_centers:
        num_centers = N

    # 1) center サンプリング
    center_idx = torch.randperm(N, device=device)[:num_centers]
    center_xyz = xyz[center_idx]        # (num_centers, 3)
    center_normal = normals[center_idx] # (num_centers, 3)

    # 2) ランダム候補 M 個
    rand_idx = torch.randint(0, N, (num_centers, M), device=device)

    # 3) 距離計算
    pts_i = center_xyz.unsqueeze(1)   # (num_centers, 1, 3)
    pts_j = xyz[rand_idx]            # (num_centers, M, 3)
    dists = (pts_i - pts_j).norm(dim=-1)  # (num_centers, M)

    self_mask = (rand_idx == center_idx.unsqueeze(1))
    dists = dists + self_mask * 1e6

    # 4) 近い k 個
    knn_dists, knn_local_idx = torch.topk(dists, k, dim=-1, largest=False)
    knn_idx = torch.gather(rand_idx, 1, knn_local_idx)  # (num_centers, k)

    # 5) normal の cos 類似度（±同一視）
    n_i = center_normal.unsqueeze(1)  # (num_centers, 1, 3)
    n_j = normals[knn_idx]           # (num_centers, k, 3)

    # dot = cosθ（unit 正規化済み前提）
    dot = (n_i * n_j).sum(dim=-1)    # (num_centers, k)
    dot = torch.clamp(dot, -1.0, 1.0)

    # 向き ± を同一視 → |dot|
    # 完全に揃う or 逆向き: |dot|=1 → loss=0
    # 直交: |dot|=0 → loss=1（最大）
    sq = 1.0 - dot.abs()             # (num_centers, k)

    # 6) 距離重み付き平均
    weights = torch.exp(-(knn_dists ** 2) / (sigma ** 2))
    loss = (weights * sq).mean()

    return loss




def ecc_loss_for_pair(
    img0: torch.Tensor, img1: torch.Tensor,
    P0: torch.Tensor, P1: torch.Tensor, # 
    K: torch.Tensor,
    c0: torch.Tensor, c1: torch.Tensor,
    points3d: torch.Tensor,
    Ns: int = 256,
    L_world: float = 200.0,
    eps_normal: float = 1.0,
    global_iter: int = 0,
    curr_angle: float = 0.0,
) -> torch.Tensor:
    """
    Epi-NAF風 Epipolar Consistency Loss for one pair of views.

    img0, img1: (1,1,H,W) or (H,W) predicted projections (cos-weighting なし版)
    P0, P1: (3,4) projection matrices
    num_planes: number of epipolar planes (stochastic approx)
    Ns: number of samples per epipolar line
    Returns:
        loss: scalar tensor
    """
    device = img0.device
    dtype = img0.dtype

    # cast shapes
    if img0.dim() == 2:
        H, W = img0.shape
    else:
        H, W = img0.shape[-2], img0.shape[-1]


    e_b = c1 - c0  # baseline vector
    e_b = e_b / torch.norm(e_b)

    # φ を 0..π から num_planes 個サンプル
    #phis = torch.linspace(0.0, math.pi, steps=num_planes, device=device, dtype=dtype)

    # Compute plane normal for plane through c0, c1 and points3d using torch
    # Ensure tensors are on same device/dtype
    v1 = (c1 - c0).to(device=device, dtype=dtype)
    v2 = (points3d - c0).to(device=device, dtype=dtype)
    n0 = torch.cross(v1, v2)
    n0 = n0 / (n0.norm() + 1e-8)



    # 平面上の2点
    X1, X2 = plane_points_on_Ephi(n0, c0, c1, L=L_world)

    # 各ビューでエピポーラ線
    l0 = epipolar_line_on_view(P0.to(device=device, dtype=dtype), K, X1, X2)
    l1 = epipolar_line_on_view(P1.to(device=device, dtype=dtype), K, X1, X2)

    # 各画像内で線分をクリップ
    p0_start, p0_end = clip_line_to_image(l0, H, W)
    p1_start, p1_end = clip_line_to_image(l1, H, W)

    if p0_start is None or p1_start is None:
        # 画像内にほぼ通っていない
        return torch.zeros(1, device=img0.device, dtype=img0.dtype).sum()
    
    DEBUG_SAVE = False
    DEBUG_DIR = Path("./vis_ecc")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    import cv2
    import numpy as np
    # === デバッグ用に img0 と img1 に線を描画して保存 ===
    if DEBUG_SAVE:
        file_prefix = f"iter_{global_iter:06d}_angle{int(math.degrees(curr_angle))}"
        # img0
        img0_vis = img0.clone().detach().cpu()
        if img0_vis.dim() == 3:
            img0_vis = img0_vis[0]  # (1,1,H,W) → (H,W)
        img0_vis = (img0_vis * 255).clamp(0, 255).numpy().astype(np.uint8)
        img0_vis = cv2.cvtColor(img0_vis, cv2.COLOR_GRAY2BGR)

        cv2.line(
            img0_vis,
            (int(p0_start[0].item()), int(p0_start[1].item())),
            (int(p0_end[0].item()), int(p0_end[1].item())),
            (0, 0, 255), 2
        )
        cv2.imwrite(str(DEBUG_DIR / f"{file_prefix}_img0.png"), img0_vis)
   
        # img1
        img1_vis = img1.clone().detach().cpu()
        if img1_vis.dim() == 3:
            img1_vis = img1_vis[0]  # (1,1,H,W) → (H,W)
        img1_vis = (img1_vis * 255).clamp(0, 255).numpy().astype(np.uint8)
        img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_GRAY2BGR)

        cv2.line(
            img1_vis,
            (int(p1_start[0].item()), int(p1_start[1].item())),
            (int(p1_end[0].item()), int(p1_end[1].item())),
            (0, 255, 0), 2
        )
        
        cv2.imwrite(str(DEBUG_DIR / f"{file_prefix}_img1.png"), img1_vis)

    # 各線分上を Ns点サンプル
    us0, vs0, delta0 = sample_points_on_segment(p0_start, p0_end, Ns)
    us1, vs1, delta1 = sample_points_on_segment(p1_start, p1_end, Ns)

    # 法線方向微分
    d0 = compute_normal_derivative(img0, us0, vs0, l0, eps=eps_normal)  # (Ns,)

    d1 = compute_normal_derivative(img1, us1, vs1, l1, eps=eps_normal)  # (Ns,)

    # derivs_0, derivs_1 がどれだけ似ているか確認
    # ECC の L1 diff 出力
    # Tensor → numpy (GPU → CPU)
    # d0_np = d0.detach().cpu().numpy()
    # d1_np = d1.detach().cpu().numpy()
    # l1_diff = np.mean(np.abs(d0_np - d1_np))
    # angle_deg = math.degrees(curr_angle)
    # print(f"ECC Loss angle {angle_deg:.1f} deg: L1 diff = {l1_diff:.6f}")

    # # プロット保存
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6,4))
    # plt.plot(d0_np, label="cam 0")
    # plt.plot(d1_np, label="cam 1")
    # plt.title(f"Normal Derivative along Epipolar Line (θ={angle_deg:.1f}°)\nL1 diff={l1_diff:.6f}")
    # plt.xlabel("Sample Index along Line")
    # plt.ylabel("dI/dn")
    # plt.legend()
    # plt.tight_layout()

    # # 保存 (角度で保存名を変える・iteration 入れることも推奨)
    # save_path = DEBUG_DIR / f"ecc_derivative_phi_{angle_deg:.1f}.png"
    # plt.savefig(save_path, dpi=300)
    # plt.close()

    # print(f"Saved derivative plot to {save_path}")


    loss = l1_loss(d0, d1)

    return loss

def pseudo_gt_loss_step(
    dtheta_rad: float,
    gt_image,
    pred_minus,
    pred_plus,
    pred_plus_for_loss,
):
    """
    1ステップ分の loss を計算する例。
    - theta_deg: 中心角度（GT がある角度）
    - gt_proj_dict: {角度: GT投影Tensor}
    """

    # --------- 2) 視点微分ベースの pseudo-GT 損失 ---------
    #   θ の近傍 ±Δθ でモデルを動かして dP/dθ を推定


    with torch.no_grad():
        # 数値微分で dP/dθ ≈ (P(θ+Δ) - P(θ-Δ)) / (2Δ)
        dP_dtheta  = (pred_plus - pred_minus) / (2.0 * dtheta_rad)

        # GT(θ) に微分項を足して「擬似GT(θ+Δ)」を作る
        pseudo_gt_plus = gt_image + dtheta_rad * dP_dtheta
        pseudo_gt_plus = pseudo_gt_plus.detach()  # 勾配を流さない


    # pseudo-GT loss
    L_pseudo = F.l1_loss(pred_plus_for_loss, pseudo_gt_plus)

    return L_pseudo



def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
