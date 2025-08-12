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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


import numpy as np

import numpy as np

def getWorld2View3(
    R: np.ndarray,
    t: np.ndarray,
    translate: np.ndarray      = np.array([0.0, 0.0, 0.0]),
    scale:   float              = 1.0,
    rotation_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    R:                (3,3) カメラ→ワールド回転行列（c2w）
    t:                (3,)   ワールド→カメラ平行移動ベクトル（w2c の並進成分）
    translate:        (3,)   カメラ中心に追加でかける平行移動
    scale:            float  カメラ中心にかけるスケール
    rotation_matrix: (3,3)   カメラ姿勢のオフセット回転（ワールド軸まわり）
    戻り値:           (4,4)  更新後のワールド→カメラ同次変換行列
    """

    # --- ステップ 1: 元の w2c を組み立てて逆行列化し、C2W を得る ---
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.T     # w2c の回転部
    Rt[:3,  3] = t       # w2c の並進部
    C2W = np.linalg.inv(Rt)  # カメラ→ワールド行列

    # --- ステップ 2: 回転オフセットを c2w の回転部に Left-multiplication ---
    if rotation_matrix is not None:
        # C2W[:3,:3] は c2w の回転部 R_cw
        # オフセット回転 R_off を左からかけると「まず旧姿勢→つづいてオフセット」が合成される
        C2W[:3, :3] = rotation_matrix @ C2W[:3, :3]

    # --- ステップ 3: translate → rotation → scale をカメラ中心に適用 ---
    cam_center = C2W[:3, 3]           # (x, y, z) of camera center in world
    cam_center = cam_center + translate
    if rotation_matrix is not None:
        cam_center = rotation_matrix @ cam_center
    cam_center = cam_center * scale
    C2W[:3, 3] = cam_center            # 書き戻し

    # --- ステップ 4: 最後に再度逆行列化して w2c を得る ---
    Rt_new = np.linalg.inv(C2W)
    return Rt_new.astype(np.float32)



def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))