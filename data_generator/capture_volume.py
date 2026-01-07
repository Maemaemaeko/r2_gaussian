import numpy as np
import pyvista as pv
from math import cos, sin, pi

vol_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_360_angle_360/2_teapot_cone/vol_gt.npy"
vol = np.load(vol_path)

# ---- 表示設定（必要に応じて調整） ----
window_size = (800, 800)
cmap = "viridis"
opacity = "linear"

# ---- 軌道パラメータ ----
seconds = 6            # 動画の長さ（秒）
fps = 30               # フレームレート
n_frames = seconds * fps
z_offset = 0.0         # 中心からの高さ（Z 方向のオフセット）。水平なら 0
radius_scale = 4     # XY の大きさに対する半径倍率（外側から見たいなら >1）

# ---- Plotter 準備 ----
pl = pv.Plotter(window_size=window_size, line_smoothing=True, off_screen=True)
actor = pl.add_volume(vol, cmap=cmap, opacity=opacity)

# 体積のバウンディングボックスから、中心と推奨半径を計算
xmin, xmax, ymin, ymax, zmin, zmax = actor.GetBounds()
cx = 0.5 * (xmin + xmax)
cy = 0.5 * (ymin + ymax)
cz = 0.5 * (zmin + zmax)

# XY の最大寸法に基づいて半径を決める（ちょい外側から眺める）
r_base = 0.5 * max(xmax - xmin, ymax - ymin)
radius = radius_scale * r_base

# カメラの注視点と上方向
pl.camera.focal_point = (cx, cy, cz)
pl.camera.view_up = (0.0, 0.0, 1.0)

# ---- 動画を書き出し ----
out_path = "orbit_z.mp4"   # "orbit_z.gif" にすれば GIF も可（環境による）
pl.open_movie(out_path, framerate=fps)
pl.show(auto_close=False)  # ウィンドウは off_screen=True なので表示されません

#0 → 360 度を等間隔に回転（Z 軸周り：XY 平面で円運動）
for i, deg in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
    theta = deg * pi / 180.0
    cam_pos = (cx + radius * cos(theta),
               cy + radius * sin(theta),
               cz + z_offset)
    pl.camera.position = cam_pos
    pl.render()
    pl.write_frame()  # 1フレーム保存


# # 0 → 360 度を等間隔に回転（Y 軸周り：XZ 平面で円運動）
# for i, deg in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
#     theta = deg * pi / 180.0
#     cam_pos = (cx + radius * cos(theta),  # X 成分を回す
#                cy,                        # Yは中心で固定
#                cz + radius * sin(theta))  # Z 成分を回す
#     pl.camera.position = cam_pos
#     pl.camera.focal_point = (cx, cy, cz)  # 注視点は常にシーン中心
#     pl.camera.view_up = (0, 1, 0)   # 上方向を常に+Zに固定
#     pl.render()
#     pl.write_frame()

pl.close()
print(f"Saved to {out_path}")
