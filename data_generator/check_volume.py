import math
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

#vol_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/2_teapot_cone/vol_gt.npy"
#vol_path = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_3_angle_360/teapot/vol_gt.npy"
#vol_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/aEupholus_A_CT_cone/vol_gt.npy"
vol_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/0_foot_cone/vol_gt.npy"
binary_save_path = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_3_angle_360/teapot/vol_binary.npy"


vol = np.load(vol_path)

D, H, W = vol.shape

# 中央のスライスを表示する


# 閾値を決める（例：0.01 以下を 0 とする）
threshold = 0.2

# # 閾値処理
# binary_voxel = np.where(vol > threshold, 1, 0).astype(np.uint8)

# print(binary_voxel.sum())



# 中央の sagittal スライス（x 方向）
mid_x = W // 2
slice_x = vol[:, :, mid_x]  # shape = (D, H)

D, H_img = slice_x.shape  # D = 縦方向の長さ, H_img = 横方向

plt.figure(figsize=(5,5))
plt.title(f"Sagittal slice (x={mid_x})")
plt.imshow(slice_x, cmap="gray")

# -------------------------------------------------
# ① 縦の赤線（ray 位置）
# y range
y_min, y_max = -1.0, 1.0
step = 0.5

y_values = np.arange(y_min, y_max + 1e-6, step)
y_values = np.array([0.15, 0, -0.15])
for y in y_values:
    # y → col index（float のまま OK）
    col = (y - y_min) / (y_max - y_min) * (H_img - 1)

    plt.axvline(x=col, color='red', linestyle='--', linewidth=1)
    plt.text(col, 0, f"y={y:.1f}", color='red', fontsize=8,
             rotation=90, verticalalignment='bottom')

# -------------------------------------------------
# ② t=0〜2 を上→下に対応させた水平ライン
t_min, t_max = 0.0, 2.0
step = 0.5

#t_values = np.arange(t_min, t_max + 1e-6, step)
t_values = np.array([1.1055, 0.7236, 1.095, 0.6633])

for t in t_values:
    # t → z index に変換
    z = int((t - t_min) / (t_max - t_min) * (D - 1))
    plt.axhline(y=z, color='yellow', linestyle='--', linewidth=0.8)
    plt.text(0, z, f"t={t:.1f}", color='yellow', fontsize=8,
             verticalalignment='bottom')

# -------------------------------------------------
plt.axis("off")
plt.show()

# # 断面を表示していく
# # === 断面を順に表示 ===
# for i in range(vol.shape[0]):  # z軸方向にスライス
#     plt.imshow(vol[i, :, :], cmap='gray', vmin=0, vmax=1)
#     plt.title(f"Slice {i+1}/{vol.shape[0]}")
#     plt.axis('off')
#     plt.pause(0.05)  # スライス間の表示間隔（0.05秒）
#     plt.clf()

# plt.close()


# 保存

#np.save(binary_save_path, binary_voxel)

plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(binary_voxel, cmap="viridis", opacity="linear")
plotter.show_axes()  # ← これだけ！
plotter.show()

