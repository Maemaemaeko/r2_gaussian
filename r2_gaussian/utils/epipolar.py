import os
import os.path as osp

from click import Tuple
import torch
import sys
from argparse import ArgumentParser
import numpy as np
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


def rotation_matrix_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    ベクトル a を b に回す 3x3 回転行列 R を返す（両方とも3次元）。
    Rodrigues の回転公式で実装。
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = np.dot(a, b)

    # ほぼ同じ向き → 単位行列
    if np.isclose(dot, 1.0):
        return np.eye(3, dtype=np.float64)

    # ほぼ逆向き → a に直交する任意軸まわりに 180°回転
    if np.isclose(dot, -1.0):
        # a と直交するベクトルを1本取る
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if np.isclose(np.abs(np.dot(a, tmp)), 1.0):
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = tmp - np.dot(tmp, a) * a
        v = v / np.linalg.norm(v)

        # 180°回転のRodrigues: R = I + 2*K^2（sinπ=0, 1-cosπ=2）
        K = np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ],
            dtype=np.float64,
        )
        R = np.eye(3, dtype=np.float64) + 2 * (K @ K)
        return R

    # 一般の場合
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    v = v / s

    K = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + K + K @ K * ((1.0 - dot) / (s * s))
    return R


def create_arrow(origin, direction, length=50.0, color=(1, 0, 0)):
    """
    origin: (3,) np.array, 開始点
    direction: (3,) np.array, 向きベクトル
    length: 矢印の長さ
    color: RGB
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None

    direction = direction / norm  # まず正規化
    direction_scaled = direction * length

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.2,
        cone_radius=0.2,
        cylinder_height=0.8 * length,
        cone_height=0.2 * length,
    )
    arrow.paint_uniform_color(np.array(color, dtype=np.float64))

    # Open3D の矢印は Z+ 方向を向いているので、Z軸→direction の回転を作って適用
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R = rotation_matrix_from_two_vectors(z_axis, direction)

    arrow.rotate(R, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    arrow.translate(origin)

    return arrow


def plane_from_phi_np(phi: float, c0: np.ndarray, e_b: np.ndarray,
                      v0: np.ndarray, v1: np.ndarray): #-> Tuple[np.ndarray, float]:
    """
    phi: angle parameter in [0, pi]
    c0: (3,) source position
    e_b, v0, v1: (3,) basis vectors (eb は使わないが型だけ合わせておく)
    Returns:
        n: (3,) plane normal (unit)
        d: scalar, plane is n·x + d = 0
    """
    n = np.cos(phi) * v0 + np.sin(phi) * v1
    n = n / np.linalg.norm(n)
    d = -np.dot(n, c0)  # c0 を通る平面
    return n, d


# def plane_points_on_Ephi_np(
#     n: np.ndarray,
#     obj_center: np.ndarray,
#     e_b: np.ndarray,
#     L: float,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     エピポーラ平面 E_phi 上の 2 点 X1, X2 を返す。

#     Parameters
#     ----------
#     n : (3,) np.ndarray
#         エピポーラ平面の法線ベクトル（単位ベクトルを想定）
#     obj_center : (3,) np.ndarray
#         原点として使う平面上の点（例: ボリューム中心）
#     e_b : (3,) np.ndarray
#         ベースラインの単位ベクトル（c0→c1 方向など）
#     L : float
#         平面内にとる線分の長さ（world座標系のスケール）

#     Returns
#     -------
#     X1, X2 : (3,) np.ndarray
#         X1 = obj_center
#         X2 = obj_center + L * t （t は平面内の単位ベクトル）
#     """
#     n = np.asarray(n, dtype=np.float64)
#     obj_center = np.asarray(obj_center, dtype=np.float64)
#     e_b = np.asarray(e_b, dtype=np.float64)

#     # 平面内の方向ベクトル t = n × e_b
#     t = np.cross(n, e_b)
#     t_norm = np.linalg.norm(t) 

#     if t_norm < 1e-8:
#         # n と e_b がほぼ平行 → 平面の定義が壊れるので、obj_center を2点返しておく
#         X1 = obj_center.copy()
#         X2 = obj_center.copy()
#         return X1, X2

#     t = t / t_norm  # 正規化（単位ベクトル）

#     X1 = obj_center.copy()
#     X2 = obj_center + L * t

#     return X1, X2


def plane_points_on_Ephi_np(n: torch.Tensor, c0: torch.Tensor, c1: torch.Tensor, L: float = 200.0):
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

def create_plane_patch(point: np.ndarray, normal: np.ndarray,
                       e_b: np.ndarray, size: float,
                       color=(0.8, 0.8, 0.8), alpha: float = 0.3):
    """
    point: (3,) 平面上の1点
    normal: (3,) 平面法線（unit）
    e_b: (3,) 平面内にある方向（baseline） ※ normal ⟂ e_b の想定
    size: パッチの一辺の長さ（だいたいのスケール）
    color: RGB
    alpha: 無視（Open3D TriangleMesh は透過混ざらないので色だけ）

    Returns:
        plane_mesh: o3d.geometry.TriangleMesh
    """
    normal = normal / np.linalg.norm(normal)
    u = e_b / np.linalg.norm(e_b)         # 平面内の1軸
    v = np.cross(normal, u)               # もう1軸（normal, u に直交）
    v = v / np.linalg.norm(v)

    half = size * 0.5
    p0 = point + half * u + half * v
    p1 = point - half * u + half * v
    p2 = point - half * u - half * v
    p3 = point + half * u - half * v

    verts = np.stack([p0, p1, p2, p3], axis=0)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    return mesh

def project_point_np(X: np.ndarray, w2c: np.ndarray, K: np.ndarray,
                     image_size):
    """
    X: (3,) world coordinates
    w2c: (4,4) world-to-camera transform
    K: (3,3) intrinsic matrix
    image_size: (width, height)

    Returns:
        (u, v): pixel coordinates
        depth: Z_cam (distance in camera space)
    """
    # 1. 世界→カメラ変換
    X_h = np.append(X, 1.0)  # (4,)
    X_cam = (w2c @ X_h)[:3]  # (3,)

    depth = X_cam[2]  # カメラからの距離

    if depth <= 0:
        return None, depth  # カメラ裏側の場合

    # 2. 透視投影
    x = X_cam / depth  # (x/z, y/z, 1)

    pixel = K @ x  # (u, v, 1)
    u = pixel[0]
    v = pixel[1]

    width, height = image_size

    return np.array([u, v]), depth


def clip_infinite_line_to_image(
    p1: np.ndarray,
    p2: np.ndarray,
    width: int,
    height: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    p1, p2 を通る“無限直線”と画像矩形との交差部分を求める。
    画像矩形内に収まる区間の両端点を返す。

    画像矩形は [0, width) x [0, height) を想定。

    Parameters
    ----------
    p1, p2 : (2,) np.ndarray
        画像平面上の2点 (x, y)
    width, height : int
        画像サイズ

    Returns
    -------
    q1, q2 : Optional[(2,) np.ndarray]
        画像内に収まる直線区間の端点。
        交差が 2 点未満の場合は (None, None)。
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        # 2点がほぼ同じ
        return None, None

    xmin, xmax = 0.0, float(width)
    ymin, ymax = 0.0, float(height)

    intersections = []

    # --- x = xmin ---
    if abs(dx) > 1e-12:
        t = (xmin - x1) / dx
        y = y1 + t * dy
        if ymin <= y <= ymax:
            intersections.append(np.array([xmin, y], dtype=np.float64))

    # --- x = xmax ---
    if abs(dx) > 1e-12:
        t = (xmax - x1) / dx
        y = y1 + t * dy
        if ymin <= y <= ymax:
            intersections.append(np.array([xmax, y], dtype=np.float64))

    # --- y = ymin ---
    if abs(dy) > 1e-12:
        t = (ymin - y1) / dy
        x = x1 + t * dx
        if xmin <= x <= xmax:
            intersections.append(np.array([x, ymin], dtype=np.float64))

    # --- y = ymax ---
    if abs(dy) > 1e-12:
        t = (ymax - y1) / dy
        x = x1 + t * dx
        if xmin <= x <= xmax:
            intersections.append(np.array([x, ymax], dtype=np.float64))

    if len(intersections) < 2:
        return None, None

    # 角で重複する場合があるので、近い点はまとめる
    uniq = []
    for pt in intersections:
        if not any(np.linalg.norm(pt - q) < 1e-6 for q in uniq):
            uniq.append(pt)

    if len(uniq) < 2:
        return None, None

    # 2点より多い場合は、一番遠い2点を選ぶ
    if len(uniq) > 2:
        max_d = -1.0
        best_i, best_j = 0, 1
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                d = np.linalg.norm(uniq[i] - uniq[j])
                if d > max_d:
                    max_d = d
                    best_i, best_j = i, j
        q1, q2 = uniq[best_i], uniq[best_j]
    else:
        q1, q2 = uniq[0], uniq[1]

    return q1, q2


def bilinear_sample_gray(image: np.ndarray, x: float, y: float) -> float:
    """
    グレースケール画像 image[y, x] から (x, y) をバイリニア補間でサンプリング。
    image: (H, W)
    x, y: float, 画像座標 (0 <= x <= W-1, 0 <= y <= H-1)

    画像外の場合は 0.0 を返す。
    """
    H, W = image.shape[:2]

    if x < 0 or x > W - 1 or y < 0 or y > H - 1:
        return 0.0

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, W - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, H - 1)

    wx = x - x0
    wy = y - y0

    I00 = float(image[y0, x0])
    I01 = float(image[y0, x1])
    I10 = float(image[y1, x0])
    I11 = float(image[y1, x1])

    I0 = (1.0 - wx) * I00 + wx * I01
    I1 = (1.0 - wx) * I10 + wx * I11
    I = (1.0 - wy) * I0 + wy * I1

    return I

def sample_along_segment_with_normal_derivative(
    image: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    Ns: int,
    eps: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    画像上の線分 q1-q2 を Ns 点サンプリングし、
    各点において「線の法線方向」の中央差分を計算する。

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        グレースケール画像（cosine-weighted projection など）
    q1, q2 : np.ndarray, shape (2,)
        画像上の2点 [x, y] （たとえば clip_infinite_line_to_image の結果）
    Ns : int
        線分上のサンプル点数
    eps : float
        法線方向の中央差分ステップサイズ（ピクセル単位）

    Returns
    -------
    coords : np.ndarray, shape (Ns, 2)
        各サンプル点の (x, y) 座標
    values : np.ndarray, shape (Ns,)
        各サンプル点の画像値 I(x, y)
    derivs : np.ndarray, shape (Ns,)
        各サンプル点における法線方向の中央差分
        dI/dn ≈ (I(p+eps*n) - I(p-eps*n)) / (2*eps)
    """
    H, W = image.shape[:2]

    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    # 線分方向ベクトルと長さ
    d = q2 - q1
    length = np.linalg.norm(d)
    if length < 1e-8:
        # 退避：長さゼロ
        coords = np.tile(q1[None, :], (Ns, 1))
        values = np.zeros(Ns, dtype=np.float64)
        derivs = np.zeros(Ns, dtype=np.float64)
        return coords, values, derivs

    # 接線方向（単位ベクトル）
    t = d / length

    # 法線方向（右手系）: t = (tx, ty) → n = (-ty, tx)
    n = np.array([-t[1], t[0]], dtype=np.float64)

    # Ns 点を線分上に等間隔サンプリング
    # t_param in [0, 1]
    t_params = np.linspace(0.0, 1.0, Ns)
    coords = np.zeros((Ns, 2), dtype=np.float64)
    values = np.zeros(Ns, dtype=np.float64)
    derivs = np.zeros(Ns, dtype=np.float64)

    for i, tau in enumerate(t_params):
        # 線分上の点 p = q1 + tau * (q2 - q1)
        p = q1 + tau * d
        x, y = p[0], p[1]
        coords[i] = p

        # 元の位置での値
        I0 = bilinear_sample_gray(image, x, y)
        values[i] = I0

        # 法線方向に eps だけずらした2点でサンプル
        xp = x + eps * n[0]
        yp = y + eps * n[1]
        xm = x - eps * n[0]
        ym = y - eps * n[1]

        Ip = bilinear_sample_gray(image, xp, yp)
        Im = bilinear_sample_gray(image, xm, ym)

        derivs[i] = (Ip - Im) / (2.0 * eps)

    return coords, values, derivs


def cosine_weight_image(image: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    画像に cosine-weight を適用する関数。
    「source → pixel方向」と「source →中心」のcosineを重みとして掛ける。
    厳密には投影行列依存だが、一般的な CBCT 近似として、
    下面の単純な方向ベクトルから算出する。

    Parameters
    ----------
    image : (H, W) np.ndarray
        元の投影画像
    K : (3, 3) np.ndarray
        カメラの内部パラメータ（fx, fy, cx, cy）

    Returns
    -------
    weighted : (H, W) np.ndarray
        cosine-weight 適用後の画像
    """
    H, W = image.shape[:2]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # pixel grid
    y, x = np.mgrid[0:H, 0:W]

    # direction vector in camera space
    dx = (x - cx) / fx
    dy = (y - cy) / fy
    dz = 1.0  # focalを1とした方向ベクトル

    # norm
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)

    # cosine weighting (dot[d, (0,0,1)]) == dz
    cos_theta = dz / norm

    # apply weight
    weighted = image * cos_theta
    return weighted


def main(dataset: ModelParams, args):
    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    scanner_cfg = scene.scanner_cfg

    vol_mesh = create_vol_mesh(
        np.load(osp.join(dataset.source_path, "vol_gt.npy")),
        np.array(scanner_cfg["offOrigin"]),
        np.array(scanner_cfg["dVoxel"]),
        np.eye(3),
        level=args.mc_thresh,
    )

    vol_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scanner_cfg["sVoxel"][0] / 2,
        origin=scanner_cfg["offOrigin"],
    )
    vol_bbox = o3d.geometry.OrientedBoundingBox(
        center=scanner_cfg["offOrigin"],
        R=np.eye(3),
        extent=scanner_cfg["sVoxel"],
    )
    vol_bbox.color = np.array([1, 0, 0])

    unit_bbox = o3d.geometry.OrientedBoundingBox(
        center=[0, 0, 0], R=np.eye(3), extent=[2, 2, 2]
    )
    unit_bbox.color = np.array([0, 0, 1])

    cams = []
    cmap = matplotlib.colormaps["viridis"]
    cam_scale = args.cam_scale
    n_proj = len(scene.train_cameras)

    # ==== added: カメラ中心（＝X線源位置）を保存 ====
    cam_centers = []
    cam_w2c = []
    proj_images = []

    for i_proj, camera in enumerate(scene.train_cameras):
        proj_name = camera.image_name
        proj_id = i_proj
        proj = t2a(camera.original_image)[0]
        image_size = (proj.shape[1], proj.shape[0])  # (width, height)
        K = np.array(
            [
                [fov2focal(camera.FoVx, proj.shape[1]), 0, proj.shape[1] / 2],
                [0, fov2focal(camera.FoVy, proj.shape[0]), proj.shape[0] / 2],
                [0, 0, 1],
            ]
        )
        w2c = np.eye(4)
        w2c[:3, :3] = t2a(camera.R.T)
        w2c[:3, 3] = t2a(camera.T)
        c2w = np.linalg.inv(w2c)

        # ==== added: カメラ中心 c = c2w[:3,3] を保存 ====
        cam_centers.append(c2w[:3, 3].copy())

        # w2cを保存
        cam_w2c.append(w2c.copy())

        # projを保存
        proj_images.append(proj.copy())
        

        DSO = np.linalg.norm(c2w[:3, 3] - np.array(scanner_cfg["offOrigin"]))
        cam = create_textured_camera(
            K,
            w2c,
            cam_scale,
            cmap(i_proj / n_proj)[:3],
            proj.shape[1],
            proj.shape[0],
            f"{proj_id:03d}",
            proj,
        )
        cams += cam

    # ==== added: ベースラインと直交ベクトル v0, v1 を計算して可視化 ====
    # ==== ベースラインと直交ベクトル v0, v1 を計算して可視化 ====
    extra_geoms = []
    if len(cam_centers) >= 2:
        # ここでは例として 0番目 と n_proj//8 番目のカメラを使う
        idx0 = 0
        idx1 = max(1, n_proj // 8)

        c0 = np.array(cam_centers[idx0])
        c1 = np.array(cam_centers[idx1])

        # baseline
        b = c1 - c0
        eb = b / np.linalg.norm(b)

        # eb に直交する v0, v1 を作る（1-2. の内容）
        tmp = np.array([1.0, 0.0, 0.0])
        if np.isclose(np.abs(np.dot(eb, tmp)), 1.0):
            tmp = np.array([0.0, 1.0, 0.0])
        v0 = tmp - np.dot(tmp, eb) * eb
        v0 = v0 / np.linalg.norm(v0)
        v1 = np.cross(eb, v0)
        v1 = v1 / np.linalg.norm(v1)

        # ---- オブジェクト中心 & スケール ----
        obj_center = np.array(scanner_cfg["offOrigin"], dtype=np.float64)  # 体積の中心
        s_voxel = np.array(scanner_cfg["sVoxel"], dtype=np.float64)        # 体積の物理サイズ
        max_extent = float(np.max(s_voxel))

        sphere_radius = 0.1 * max_extent
        arrow_len    = 0.1 * max_extent
        plane_size   = 0.8 * max_extent

        # 1) source位置を小球で表示
        sph0 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sph0.translate(c0)
        sph0.paint_uniform_color([1, 0, 0])  # 赤

        sph1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sph1.translate(c1)
        sph1.paint_uniform_color([0, 1, 0])  # 緑

        # 2) baseline を LineSet で表示
        line_points = o3d.utility.Vector3dVector(np.stack([c0, c1], axis=0))
        line_indices = o3d.utility.Vector2iVector([[0, 1]])
        baseline_line = o3d.geometry.LineSet(points=line_points, lines=line_indices)
        baseline_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 赤

        # 3) eb, v0, v1 を「オブジェクト中心」から出す矢印として表示
        arrow_eb = create_arrow(obj_center, eb, length=arrow_len,  color=(1, 0.5, 0.0))  # オレンジ
        arrow_v0 = create_arrow(obj_center, v0, length=arrow_len,  color=(0, 0, 1.0))   # 青
        arrow_v1 = create_arrow(obj_center, v1, length=arrow_len,  color=(0, 1.0, 1.0)) # シアン

        # 4) plane_from_phi の n(φ) を可視化
        #    例として φ = 0, π/4, π/2 の3つを表示
        # 4) Now use ONLY ONE φ (plane through c0, c1, and obj_center)
        obj_center = np.array(scanner_cfg["offOrigin"], dtype=np.float64)

        # 平面法線ベクトル n0 = (c1 - c0) × (obj_center - c0)
        n0 = np.cross(c1 - c0, obj_center - c0)
        n0 = n0 / np.linalg.norm(n0)


        # ---- X1, X2 を epipolar 平面上に配置 ----
        # L はオブジェクトサイズに応じたスケールで決める
        L_for_X = 0.5 * max_extent  # 体積サイズの半分くらい

        #X1, X2 = plane_points_on_Ephi_np(n0, obj_center, eb, L=L_for_X)
        X1, X2 = plane_points_on_Ephi_np(
            torch.from_numpy(n0).float(),
            torch.from_numpy(c0).float(),
            torch.from_numpy(c1).float(),
            L=L_for_X,
        )


        # X1, X2をカメラ座標系に投影
        p1, d1 = project_point_np(X1, cam_w2c[idx0], K, image_size)
        print(p1, d1)
        p2, d2 = project_point_np(X2, cam_w2c[idx0], K, image_size)
        print(p2, d2)
        if p1 is not None and p2 is not None:
            q1, q2 = clip_infinite_line_to_image(p1, p2, image_size[0], image_size[1])
            #cos_weighted_img = cosine_weight_image(proj_images[idx0], K)
            cos_weighted_img = proj_images[idx0]
            
            # 3) その q1–q2 線分上で ECC 用のサンプルと ∂/∂t を計算
            coords, values, derivs_0 = sample_along_segment_with_normal_derivative(
                image=cos_weighted_img,  # cosine-weighted projection image
                q1=q1,
                q2=q2,
                Ns=64,
                eps=2.0,
            )
            # proj_images[idx0] に線分 q1-q2 を描画
            if q1 is not None and q2 is not None:
                import cv2
                img0 = proj_images[idx0].copy()
                #img1 = proj_images[idx1].copy()
                cv2.line(
                    img0,
                    (int(q1[0]), int(q1[1])),
                    (int(q2[0]), int(q2[1])),
                    color=(255, 0, 0),
                    thickness=2,
                )
                cv2.imshow(f"Projected Line on Cam {idx0}", img0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        # X1, X2をカメラ座標系に投影
        p1, d1 = project_point_np(X1, cam_w2c[idx1], K, image_size)
        print(p1, d1)
        p2, d2 = project_point_np(X2, cam_w2c[idx1], K, image_size)
        print(p2, d2)
        if p1 is not None and p2 is not None:
            q1, q2 = clip_infinite_line_to_image(p1, p2, image_size[0], image_size[1])
            # cosine-weighted projection image
            #cos_weighted_img = cosine_weight_image(proj_images[idx1], K)
            cos_weighted_img = proj_images[idx1].copy()
            # 3) その q1–q2 線分上で ECC 用のサンプルと ∂/∂t を計算
            coords, values, derivs_1 = sample_along_segment_with_normal_derivative(
                image=cos_weighted_img,  # cosine-weighted projection image
                q1=q1,
                q2=q2,
                Ns=64,
                eps=2.0,
            )
            # proj_images[idx0] に線分 q1-q2 を描画
            if q1 is not None and q2 is not None:
                import cv2
                img1 = proj_images[idx1].copy()
                cv2.line(
                    img1,
                    (int(q1[0]), int(q1[1])),
                    (int(q2[0]), int(q2[1])),
                    color=(255, 0, 0),
                    thickness=2,
                )
                cv2.imshow(f"Projected Line on Cam {idx1}", img1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # derivs_0, derivs_1 がどれだけ似ているか確認
        l1_diff = np.mean(np.abs(derivs_0 - derivs_1))
        print(f"L1 difference of normal derivatives between Cam {idx0} and Cam {idx1}: {l1_diff:.6f}")
        if p1 is not None and p2 is not None and q1 is not None:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(derivs_0, label=f"Cam {idx0}")
            plt.plot(derivs_1, label=f"Cam {idx1}")
            plt.title("Normal Direction Derivatives along Epipolar Line")
            plt.xlabel("Sample Index along Line")
            plt.ylabel("dI/dn")
            plt.legend()
            plt.show()
        


        # X1, X2 を小さい球で表示
        X_radius = 0.015 * max_extent

        X1_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=X_radius)
        X1_sphere.translate(X1)
        X1_sphere.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色

        X2_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=X_radius)
        X2_sphere.translate(X2)
        X2_sphere.paint_uniform_color([1.0, 0.6, 0.0])  # オレンジ寄り

        # X1-X2 を結ぶ線も描いておくと「平面内の方向」が見えやすい
        X_points = o3d.utility.Vector3dVector(np.stack([X1, X2], axis=0))
        X_lines = o3d.utility.Vector2iVector([[0, 1]])
        X_line = o3d.geometry.LineSet(points=X_points, lines=X_lines)
        X_line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # 黄色ライン

        # 平面法線ベクトルの矢印
        arrow_n0 = create_arrow(obj_center, n0, length=0.3 * max_extent, color=(1, 0, 1))  # magenta

        # baseline ベクトルの矢印
        arrow_eb = create_arrow(obj_center, eb, length=0.3 * max_extent, color=(1, 0.5, 0))  # orange

        # エピポーラ平面パッチ
        plane_mesh = create_plane_patch(
            point=obj_center,
            normal=n0,
            e_b=eb,           # 平面内の1軸として使う
            size=plane_size,
            color=(0.8, 0.8, 0.8),
        )

        extra_geoms.extend([
            X1_sphere, X2_sphere, X_line,
            arrow_n0, arrow_eb, plane_mesh
        ])

        for g in [sph0, sph1, baseline_line, arrow_eb, arrow_v0, arrow_v1]:
            print(g)
            if g is not None:
                extra_geoms.append(g)

        print("c0 =", c0)
        print("c1 =", c1)
        print("eb =", eb)
        print("v0 =", v0, "dot(eb,v0) =", np.dot(eb, v0))
        print("v1 =", v1, "dot(eb,v1) =", np.dot(eb, v1), "dot(v0,v1) =", np.dot(v0, v1))

    vis_assets = [vol_bbox, vol_coord, unit_bbox] + extra_geoms
    o3d.visualization.draw_geometries(vis_assets, mesh_show_back_face=True)

if __name__ == "__main__":
    # fmt: off
    parser = ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument(
        "--mc_thresh",
        type=float,
        default=0.1,
        help="Marching cubes threshold for volume mesh generation.",
    )
    parser.add_argument(
        "--cam_scale",
        type=float,
        default=100.0,
        help="Scale for camera frustum visualization.",
    )
    args = parser.parse_args()
    # fmt: on

    args = parser.parse_args(sys.argv[1:])
    main(lp.extract(args), args)