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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn


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
    num_centers=5000,  # loss 計算に使う Gaussians の数
    k=8,
    M=64,
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
