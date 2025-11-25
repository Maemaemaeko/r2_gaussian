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
    

def smoothness_loss_knn(
    xyz,
    theta,
    num_centers=1000,  # loss 計算に使う Gaussians の数
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
        return torch.tensor(0.0, device=device, dtype=dtype)

    # 各線分上を Ns点サンプル
    us0, vs0, delta0 = sample_points_on_segment(p0_start, p0_end, Ns)
    us1, vs1, delta1 = sample_points_on_segment(p1_start, p1_end, Ns)

    # 法線方向微分
    d0 = compute_normal_derivative(img0, us0, vs0, l0, eps=eps_normal)  # (Ns,)

    d1 = compute_normal_derivative(img1, us1, vs1, l1, eps=eps_normal)  # (Ns,)


    loss = l1_loss(d0, d1)
    import numpy as np
    d0 = d0.detach().cpu().numpy()
    d1 = d1.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.plot(d0)
    plt.plot(d1)
    plt.title("Normal Direction Derivatives along Epipolar Line")
    plt.xlabel("Sample Index along Line")
    plt.ylabel("dI/dn")
    plt.legend()
    plt.show()
        


    # # ∑ Δ_j δ の形に近づける
    # S0 = d0 * delta0
    # S1 = d1 * delta1

    # # Epi-NAF の Eq.(4) 的な残差
    # residual = S0.sum() - S1.sum()  # scalar

    
    return loss



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
