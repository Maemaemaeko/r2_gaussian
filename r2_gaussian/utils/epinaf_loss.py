import os
import os.path as osp

from click import Tuple
import torch
import sys
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import open3d as o3d
import matplotlib

sys.path.append("./")
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.plot_utils import create_textured_camera, create_vol_mesh
from r2_gaussian.utils.graphics_utils import fov2focal
from r2_gaussian.utils.general_utils import t2a


import numpy as np
import open3d as o3d
from typing import Optional, Tuple


def plane_points_on_Ephi(n: torch.Tensor, c0: torch.Tensor, c1: torch.Tensor, L: float = 200.0):
    """
    n:  plane normal (3,)
    c0: source of view 0 (3,)
    c1: source of view 1 (3,)
    L:  スケール [mm] など
    """
    b = c1 - c0
    e_b = b / torch.norm(b)
    t = torch.cross(n, e_b)
    t = t / torch.norm(t)

    X1 = c0 + L * t
    X2 = c0 + L * (t + e_b)
    return X1, X2  # (3,), (3,)


def epipolar_line_on_view(w2c: torch.Tensor, K: torch.Tensor, X1: torch.Tensor, X2: torch.Tensor):
    """
    w2c: (3,4) world to camera projection matrix
    X1, X2: (3,)  plane上の2点
    Returns:
      (u1, v1, u2, v2): エピポーラ線のパラメータ
    """
    X1_h = torch.cat([X1, torch.ones(1, device=X1.device, dtype=X1.dtype)])  # (4,)
    X2_h = torch.cat([X2, torch.ones(1, device=X2.device, dtype=X2.dtype)])  # (4,)

    x1_c = (w2c @ X1_h)[:3] # (3,)
    x2_c = (w2c @ X2_h)[:3] # (3,)

    x1_img = K @ x1_c  # (3,)
    x2_img = K @ x2_c  # (3,)
    u1, v1 = x1_img[0] / x1_img[2], x1_img[1] / x1_img[2]
    u2, v2 = x2_img[0] / x2_img[2], x2_img[1] / x2_img[2]

    return (u1, v1, u2, v2)


def clip_line_to_image(line: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], H: int, W: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Clip an infinite line defined by two points to the image rectangle [0,W]x[0,H].
    Works with torch tensors and returns two 2D points as torch tensors on the same device/dtype.
    """
    x1, y1, x2, y2 = line
    # Ensure scalar tensors
    x1 = torch.as_tensor(x1)
    y1 = torch.as_tensor(y1)
    x2 = torch.as_tensor(x2)
    y2 = torch.as_tensor(y2)

    dx = x2 - x1
    dy = y2 - y1

    # early exit for degenerate line
    if torch.abs(dx) < 1e-12 and torch.abs(dy) < 1e-12:
        return None, None

    device = x1.device
    dtype = x1.dtype
    xmin = torch.as_tensor(0.0, device=device, dtype=dtype)
    xmax = torch.as_tensor(float(W), device=device, dtype=dtype)
    ymin = torch.as_tensor(0.0, device=device, dtype=dtype)
    ymax = torch.as_tensor(float(H), device=device, dtype=dtype)

    intersections = []  # list of torch tensors shape (2,)

    # --- x = xmin ---
    if torch.abs(dx) > 1e-12:
        t = (xmin - x1) / dx
        y = y1 + t * dy
        if (y >= ymin) and (y <= ymax):
            intersections.append(torch.stack([xmin, y]))

    # --- x = xmax ---
    if torch.abs(dx) > 1e-12:
        t = (xmax - x1) / dx
        y = y1 + t * dy
        if (y >= ymin) and (y <= ymax):
            intersections.append(torch.stack([xmax, y]))

    # --- y = ymin ---
    if torch.abs(dy) > 1e-12:
        t = (ymin - y1) / dy
        x = x1 + t * dx
        if (x >= xmin) and (x <= xmax):
            intersections.append(torch.stack([x, ymin]))

    # --- y = ymax ---
    if torch.abs(dy) > 1e-12:
        t = (ymax - y1) / dy
        x = x1 + t * dx
        if (x >= xmin) and (x <= xmax):
            intersections.append(torch.stack([x, ymax]))

    if len(intersections) < 2:
        return None, None

    # deduplicate near-equal points (tolerance)
    uniq = []
    tol = 1e-6
    for pt in intersections:
        keep = True
        for q in uniq:
            if torch.norm(pt - q) < tol:
                keep = False
                break
        if keep:
            uniq.append(pt)

    if len(uniq) < 2:
        return None, None

    # If more than two, pick the farthest pair
    if len(uniq) > 2:
        max_d = -1.0
        best_i, best_j = 0, 1
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                d = torch.norm(uniq[i] - uniq[j]).item()
                if d > max_d:
                    max_d = d
                    best_i, best_j = i, j
        q1, q2 = uniq[best_i], uniq[best_j]
    else:
        q1, q2 = uniq[0], uniq[1]

    return q1, q2


def sample_points_on_segment(p_start: torch.Tensor, p_end: torch.Tensor, Ns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    p_start, p_end: (2,) 線分の端点
    Ns: サンプル数
    Returns:
      us, vs: (Ns,) 線分上のサンプル点座標
      delta: (Ns,) 各点の間隔
    """
    us = torch.linspace(p_start[0], p_end[0], steps=Ns, device=p_start.device, dtype=p_start.dtype)
    vs = torch.linspace(p_start[1], p_end[1], steps=Ns, device=p_start.device, dtype=p_start.dtype)

    delta = torch.sqrt((us[1:] - us[:-1]) ** 2 + (vs[1:] - vs[:-1]) ** 2)
    delta = torch.cat([delta, delta[-1:].clone()])  # 最後の点は前の点と同じ間隔とする

    return us, vs, delta

def bilinear_sample_gray(image, x, y):
    """
    グレースケール画像 image[y, x] から (x, y) をバイリニア補間でサンプリング。
    image: (H, W)
    x, y: float, 画像座標 (0 <= x <= W-1, 0 <= y <= H-1)

    画像外の場合は 0.0 を返す。
    """
    # Support both numpy arrays and torch tensors; if torch, return torch tensor
    if isinstance(image, torch.Tensor):
        H, W = image.shape[-2], image.shape[-1]
        device = image.device
        dtype = image.dtype
        x_t = torch.as_tensor(x, device=device, dtype=dtype)
        y_t = torch.as_tensor(y, device=device, dtype=dtype)

        # Broadcast scalars to 1D tensors
        scalar_input = False
        if x_t.dim() == 0:
            x_t = x_t.unsqueeze(0)
            scalar_input = True
        if y_t.dim() == 0:
            y_t = y_t.unsqueeze(0)
            scalar_input = scalar_input and True

        # Mask invalid coords (outside image)
        valid = (x_t >= 0) & (x_t <= (W - 1)) & (y_t >= 0) & (y_t <= (H - 1))

        # Normalize coordinates for grid_sample (x: [-1,1], y: [-1,1])
        x_norm = (x_t / (W - 1)) * 2.0 - 1.0
        y_norm = (y_t / (H - 1)) * 2.0 - 1.0

        # grid_sample expects grid in shape (N, H_out, W_out, 2)
        # We'll set N=1, H_out=1, W_out=Ns, grid coords ordered (x, y)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(1)  # (1, 1, Ns, 2)

        # Prepare image as (N, C, H, W)
        if image.dim() == 2:
            img_t = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            # shape (C, H, W) or (1, H, W). Ensure batch dim
            img_t = image.unsqueeze(0)
        elif image.dim() == 4:
            img_t = image
        else:
            raise ValueError(f"Unsupported image dims: {image.shape}")

        # Use grid_sample to bilinearly sample points; padding_mode='zeros' returns 0 for out-of-bounds
        sampled = F.grid_sample(img_t, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        # sampled shape: (N, C, 1, Ns) -> squeeze to (C, Ns)
        sampled = sampled.squeeze(2).squeeze(0)

        # If image had channels > 1, reduce to single channel by averaging
        if sampled.dim() == 2 and sampled.shape[0] > 1:
            sampled = sampled.mean(dim=0)

        # sampled is now (Ns,) or (C,Ns) with C==1
        if scalar_input:
            return sampled.squeeze(0)
        # reshape to the original x shape
        return sampled.reshape(x_t.shape)

    # Fallback numpy implementation (for legacy callers)
    H, W = image.shape[:2]
    if np.isscalar(x):
        xs = np.array([x])
        ys = np.array([y])
        scalar = True
    else:
        xs = np.asarray(x)
        ys = np.asarray(y)
        scalar = False

    xs_clamped = np.clip(xs, 0.0, W - 1.0)
    ys_clamped = np.clip(ys, 0.0, H - 1.0)

    x0 = np.floor(xs_clamped).astype(np.int64)
    y0 = np.floor(ys_clamped).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = xs_clamped - x0
    wy = ys_clamped - y0

    I00 = image[y0, x0]
    I01 = image[y0, x1]
    I10 = image[y1, x0]
    I11 = image[y1, x1]

    I0 = (1.0 - wx) * I00 + wx * I01
    I1 = (1.0 - wx) * I10 + wx * I11
    I = (1.0 - wy) * I0 + wy * I1

    if scalar:
        return float(I[0])
    return I



def compute_normal_derivative(img: torch.Tensor, us: torch.Tensor, vs: torch.Tensor, line: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], eps: float = 1.0) -> torch.Tensor:
    """
    img: (H,W) 画像
    us, vs: (Ns,) 線分上のサンプル点座標
    line: (u1,v1,u2,v2) エピポーラ線のパラメータ
    eps: 微小変位 [pixel]
    Returns:
      d: (Ns,) 法線方向微分
    """
    H, W = img.shape[-2], img.shape[-1]
    u1, v1, u2, v2 = line
    dx = u2 - u1
    dy = v2 - v1
    length = torch.sqrt(dx**2 + dy**2)
    n_dx = -dy / length  # 法線ベクトルのx成分
    n_dy = dx / length   # 法線ベクトルのy成分

    us_pos = us + eps * n_dx
    vs_pos = vs + eps * n_dy
    us_neg = us - eps * n_dx
    vs_neg = vs - eps * n_dy

    us_pos_clipped = torch.clamp(us_pos, 0.0, W - 1.0)
    vs_pos_clipped = torch.clamp(vs_pos, 0.0, H - 1.0)
    us_neg_clipped = torch.clamp(us_neg, 0.0, W - 1.0)
    vs_neg_clipped = torch.clamp(vs_neg, 0.0, H - 1.0)

    vals_pos = bilinear_sample_gray(img, us_pos_clipped, vs_pos_clipped)
    vals_neg = bilinear_sample_gray(img, us_neg_clipped, vs_neg_clipped)

    d = (vals_pos - vals_neg) / (2 * eps)
    return d  # (Ns,)


def main(dataset: ModelParams, args):
    # Set up dataset
    scene = Scene(dataset, shuffle=False)
