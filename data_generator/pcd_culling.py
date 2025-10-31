from importlib.resources import path
import json
import math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn.functional as F
import os

def visualize_point_cloud(pts):
    xyz = pts[:, :3]
    val = pts[:, 3]

    # 0–1に正規化してカラーマップ
    #v = (val - val.min()) / (val.max() - val.min() + 1e-8)
    colors = plt.get_cmap("viridis")(val*10)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # === 軸を追加 ===
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1,      # 軸の長さ（スケール）
        origin=[0, 0, 0]
    )

    # 可視化
    o3d.visualization.draw_geometries([pcd, axis])


def angle2pose(DSO, angle):
    """Transfer angle to pose (c2w) based on scanner geometry.
    1. rotate -90 degree around x-axis (fixed axis),
    2. rotate 90 degree around z-axis  (fixed axis),
    3. rotate angle degree around z axis  (fixed axis)"""

    phi1 = -np.pi / 2
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ]
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R3 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return transform

def point_culling_based_on_depth(
    pts,
    R,
    T,
    FoVx: float,
    FoVy: float,
    W: int,
    H: int,
    gs_depth_npy: str,
    device
):  
    def to_torch_f32(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device, dtype=torch.float32)
        elif torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")
    pts = to_torch_f32(pts)
    xyz = pts[:, :3] 

    R = to_torch_f32(R)
    T = to_torch_f32(T)
    x_cam = xyz @ R + T[None, :]
    Xc, Yc, Zc = x_cam[:, 0], x_cam[:, 1], x_cam[:, 2]

    # 画面内に投影される点のみ
    fx = 0.5 * W / math.tan(FoVx * 0.5)
    fy = 0.5 * H / math.tan(FoVy * 0.5)
    cx, cy = W * 0.5, H * 0.5
    u = (fx * (Xc / Zc)) + cx
    v = (fy * (Yc / Zc)) + cy
    in_img = (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))
    

    fdk_depths = torch.full((xyz.shape[0], 1), float('nan'), dtype=torch.float32, device=device)
    fdk_depths[:, 0] = Zc
    gs_depth = np.load(gs_depth_npy)
    gs_depth = torch.from_numpy(gs_depth).to(device=device, dtype=torch.float32)
    gs_depth = 1.0 / (gs_depth + 1e-8)

    N = xyz.shape[0]
    valid_mask = torch.ones((N,), dtype=torch.bool, device=device)

    if in_img.any():
        ui = u[in_img]
        vi = v[in_img]

        # Clamp indices to valid image bounds
        ui_clamped = torch.clamp(ui.long(), 0, gs_depth.shape[1] - 1)
        vi_clamped = torch.clamp(vi.long(), 0, gs_depth.shape[0] - 1)

        valid_depth = gs_depth[vi_clamped, ui_clamped] < fdk_depths[in_img, 0] + 1.5
        invalid_depth = ~valid_depth


    valid_mask[in_img] = valid_depth
    return valid_mask.detach().cpu().numpy()

# def point_culling_based_on_depth(
#     pts,            # (N,4) or (N,3) NumPy/Torch
#     R, T,           # (3,3), (3,)
#     FoVx: float,
#     FoVy: float,
#     W: int,
#     H: int,
#     gs_depth_npy: str,
#     device,
#     margin: float = 1.5,    # 許容マージン
# ):
#     # ---- to torch float32 (GPU) ----
#     def to_torch_f32(x):
#         if isinstance(x, np.ndarray):
#             return torch.from_numpy(x).to(device=device, dtype=torch.float32)
#         elif torch.is_tensor(x):
#             return x.to(device=device, dtype=torch.float32)
#         else:
#             raise TypeError(f"Unsupported type: {type(x)}")

#     pts_t = to_torch_f32(pts)
#     if pts_t.shape[1] >= 3:
#         xyz = pts_t[:, :3]
#     else:
#         raise ValueError(f"pts must have at least 3 columns, got {pts_t.shape}")

#     R_t = to_torch_f32(R)
#     T_t = to_torch_f32(T)
#     if R_t.shape == (4,4):
#         R_t = R_t[:3, :3]
#     if T_t.numel() == 4:
#         T_t = T_t[:3]
#     T_t = T_t.view(3)

#     # ---- world -> cam  (assume R,T are world->cam) ----
#     x_cam = (xyz @ R_t.t()) + T_t[None, :]
#     Xc, Yc, Zc = x_cam[:, 0], x_cam[:, 1], x_cam[:, 2]

#     # ---- projection ----
#     fx = 0.5 * W / math.tan(FoVx * 0.5)
#     fy = 0.5 * H / math.tan(FoVy * 0.5)
#     cx, cy = W * 0.5, H * 0.5
#     # Z で割る前に有限チェック
#     eps = 1e-8
#     denom = torch.where(torch.isfinite(Zc) & (Zc.abs() > eps), Zc, torch.full_like(Zc, float('nan')))
#     u = fx * (Xc / denom) + cx
#     v = fy * (Yc / denom) + cy

#     # ---- load GS depth (saved as inverse depth -> convert to depth) ----
#     gs_depth_np = np.load(gs_depth_npy)          # dtype=float32 の想定
#     gs_depth_t = torch.from_numpy(gs_depth_np).to(device=device, dtype=torch.float32)
#     # もし保存が inverse depth なら depth へ戻す
#     gs_depth_t = 1.0 / (gs_depth_t + 1e-8)       # 逆深度→深度

#     # ---- output mask (N,) 既定 True: このビューで判定しない点は温存 ----
#     N = xyz.shape[0]
#     valid_mask = torch.zeros((N,), dtype=torch.bool, device=device)

#     # ---- 判定に使う有効画素: 画像内 & 前方Z & 有限 ----
#     u0 = torch.floor(u)
#     v0 = torch.floor(v)
#     m = (
#         torch.isfinite(u0) & torch.isfinite(v0) & torch.isfinite(Zc) &
#         (Zc > eps) &
#         (u0 >= 0) & (u0 <= (W - 1)) &
#         (v0 >= 0) & (v0 <= (H - 1))
#     )

#     if m.any():
#         ui = u0[m].to(torch.long)
#         vi = v0[m].to(torch.long)

#         # FDK(=点のカメラ深度)
#         z_pts = Zc[m]  # (M,)

#         # GS depth サンプル（NaN対策）
#         z_img = gs_depth_t[vi, ui]  # (M,)
#         #inite = torch.isfinite(z_img)
#         # 画像に深度が無い(NaN)画素は判定しない(=Trueのまま)
#         #_eff = m.clone()
#         #_eff[m] = finite

#         # あなたの元のロジックを踏襲： z_img < z_pts + margin を「有効」とする
#         # （意味：画像の方が手前＝点は面の“後ろ側”→ここを無効にしたいなら反転してください）
#         #valid_here = torch.ones_like(z_pts, dtype=torch.bool)
#         valid_here = z_img < z_pts + 1.5

#         # m の位置にのみ反映（それ以外は True のまま）
#         valid_mask[m] = valid_here

#     # ---- 戻りは NumPy bool (上位の & と互換) ----
#     return valid_mask.detach().cpu().numpy()

def get_camera_params_from_scanner_cfg(meta_data_path, angle):
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
    cam_cfg = meta_data["scanner"]
    c2w = angle2pose(cam_cfg["DSO"], angle)
    w2c = np.linalg.inv(c2w)
    R = np.transpose(
        w2c[:3, :3]
    )  # R is stored transposed due to 'glm' in CUDA code
    print(R)
    T = w2c[:3, 3]
    print(T)
    FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
    FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2
    W = cam_cfg["nDetector"][1]
    H = cam_cfg["nDetector"][0]
    return R, T, FovX, FovY, W, H


def get_camera_params_from_gs_settings(camera_params_dir, idx):
    filename = f"{idx:05d}.npy"
    camera_params_npy = os.path.join(camera_params_dir, filename)
    
    camera_params = np.load(camera_params_npy, allow_pickle=True).item()
    R, T, FovX, FovY, W, H = camera_params["R"], camera_params["T"], camera_params["FovX"], camera_params["FovY"], camera_params["W"], camera_params["H"]
    return R, T, FovX, FovY, W, H


if __name__ == "__main__":
    src_path = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_10_angle_360/teapot/init_teapot.npy"
    pts = np.load(src_path)  
    visualize_point_cloud(pts)
    gs_depth_npy_dir = "/home/maemaeko/imari_lab/gaussian-splatting/output/teapot-aligned-rescale/train/ours_30000/renders"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    #get_camera_params_from_gs_settings("/home/maemaeko/imari_lab/gaussian-splatting/output/teapot-aligned-rescale/train/ours_30000/renders/camera_params", 0)
    valid_depth = np.ones(len(pts), dtype=bool)

    for idx in range(80):
        #R, T, FovX, FovY, W, H = get_camera_params_from_scanner_cfg("/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_75_angle_360/teapot/meta_data.json", idx)
        R, T, FovX, FovY, W, H = get_camera_params_from_gs_settings("/home/maemaeko/imari_lab/gaussian-splatting/output/teapot-aligned-rescale/train/ours_30000/renders/camera_params", idx)
        filename = f"{idx:05d}.npy"
        gs_depth_npy = os.path.join(gs_depth_npy_dir, filename)
       
        
        valid_depth &= point_culling_based_on_depth(pts, R, T, FovX, FovY, W, H, gs_depth_npy, device)
    

    pts_valid = pts[valid_depth, :]

    
    # 保存先を決める（元データファイルと同じディレクトリ）
    out_dir = os.path.dirname(src_path)
    out_name = "init_teapot_valid.npy"
    out_path = os.path.join(out_dir, out_name)
    np.save(out_path, pts_valid)
    print(f"Saved filtered points to: {out_path}")

    visualize_point_cloud(pts_valid)

    # ptc_valid[:, 3]のヒストグラムを作成
    
    densities = pts_valid[:, 3]

    plt.figure(figsize=(6,4))
    plt.hist(densities, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("Density Histogram")
    plt.xlabel("Density value * 10")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()
    # N_total = pts.shape[0]
    # n_valid = pts_valid.shape[0]

    # if n_valid == 0:
    #     raise RuntimeError("validな点が0です。しきい値が厳しすぎる可能性があります。")

    # if n_valid < N_total:
    #     deficit = N_total - n_valid  # 何個足りないか

    #     # 追加分の行を、validな点からランダムサンプリングで複製
    #     rand_idx = np.random.randint(low=0, high=n_valid, size=deficit, dtype=np.int64)
    #     duplicated = pts_valid[rand_idx].copy()
    #     duplicated[:, 3] *= 0.5
    #     pts_valid[rand_idx, 3] *= 0.5



    #     pts_final = np.concatenate([pts_valid, duplicated], axis=0)
    # else:
    #     # そもそも減ってないならそのまま
    #     pts_final = pts_valid

    # assert pts_final.shape[0] == N_total, f"{pts_final.shape[0]} != {N_total}"

    # print("Original N:", N_total)
    # print("After cull n_valid:", n_valid)
    # print("After fill pts_final.shape[0]:", pts_final.shape[0])

    # # 保存先を決める（元データファイルと同じディレクトリ）
    # out_dir = os.path.dirname(src_path)
    # out_name = "init_teapot_final.npy"
    # out_path = os.path.join(out_dir, out_name)
    # np.save(out_path, pts_final)
    # print(f"Saved filtered points to: {out_path}")

    # visualize_point_cloud(pts_final)


    





