from importlib.resources import path
import json
import math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn.functional as F
 
# shape: (N, 4)

# def visualize_point_cloud(pts):
#     xyz = pts[:, :3]
#     val = pts[:, 3]

#     # 0–1に正規化してカラーマップ
#     v = (val - val.min()) / (val.max() - val.min() + 1e-8)
#     colors = plt.get_cmap("viridis")(v)[:, :3]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # 必要ならダウンサンプル
#     # pcd = pcd.voxel_down_sample(voxel_size=0.01)

#     o3d.visualization.draw_geometries([pcd])


def visualize_point_cloud(pts):
    xyz = pts[:, :3]
    val = pts[:, 3]

    # 0–1に正規化してカラーマップ
    v = (val - val.min()) / (val.max() - val.min() + 1e-8)
    colors = plt.get_cmap("viridis")(v)[:, :3]

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


def point_depths_from_camera(
        pts: np.ndarray,
        R,
        T,
        FoVx: float,
        FoVy: float,
        W: int,
        H: int,
):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    valid = in_img


    # === 5) 出力 (N,1) で順序保持。無効は NaN ===
    fdk_depths = torch.full((xyz.shape[0], 1), float('nan'), dtype=torch.float32, device=device)
    fdk_depths[valid, 0] = Zc[valid]
    
    
    gt_depth_npy = "/home/maemaeko/imari_lab/gaussian-splatting/depth.npy"
    gt_depth = np.load(gt_depth_npy)
    gt_depth = torch.from_numpy(gt_depth).to(device=device, dtype=torch.float32)
    gt_depth = F.interpolate(
        gt_depth.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)
    # 逆数をとる
    gt_depth = 1.0 / (gt_depth + 1e-8)

    
    #　正規化
    # gt_depth_image = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    # plt.imshow(gt_depth_image.cpu().numpy(), cmap="gray")
    # plt.show()
    # gt_depthのヒスとグラムを表示
    # plt.hist(gt_depth.cpu().numpy(), bins=50)
    # plt.xlabel("Depth")
    # plt.ylabel("Frequency")
    # plt.show()
    #print(gt_depth.min(), gt_depth.max()) # 4~5あたりのはず
    # 正規化する
    #gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    #plt.imshow(gt_depth.cpu().numpy(), cmap="gray")
    #plt.show()

    valid_depth = gt_depth[v.long(), u.long()] < fdk_depths[:, 0]
    print(valid_depth.sum())
    valid_depth = valid_depth & valid

    depths_img = torch.zeros((H, W, 1), dtype=torch.float32, device=device)
    depths_img[v[valid].long(), u[valid].long(), 0] = 1 # depths[valid, 0]
    return depths_img

if __name__ == "__main__":
    pts = np.load("/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_75_angle_360/teapot/init_teapot.npy")  
    #visualize_point_cloud(pts)


    meta_data_path = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_75_angle_360/teapot/meta_data.json"
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    cam_cfg = meta_data["scanner"]
    frame_angle = 0
    c2w = angle2pose(cam_cfg["DSO"], frame_angle)
    print("camera center:", c2w[:3, 3])
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(
        w2c[:3, :3]
    )  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
    FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2

    depths_img  = point_depths_from_camera(pts, R, T, FovX, FovY, cam_cfg["nDetector"][1], cam_cfg["nDetector"][0])
    # visualize the depth image
    plt.imshow(depths_img.squeeze().cpu().numpy(), cmap="gray")
    plt.show()
