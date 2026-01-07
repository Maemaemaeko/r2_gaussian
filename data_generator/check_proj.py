import os
import os.path as osp
import numpy as np
import sys

import imageio.v3 as iio
from pathlib import Path
import matplotlib.cm as cm


sys.path.append("./")
from r2_gaussian.utils.plot_utils import show_one_volume

#proj_path = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_75_angle_360/teapot/proj_train"
proj_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/0_foot_cone/proj_train"
#proj_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/aEupholus_A_CT_cone/proj_train/interp"
proj_list = [f for f in sorted(os.listdir(proj_path)) if f.endswith(".npy")]


projs = np.stack(
    [np.load(osp.join(proj_path, proj_id)) for proj_id in proj_list], axis=-1
)


show_one_volume(projs)


# ----- Parameter -----
num_frames = 60
fps = 10
output_path = Path("./projection60_viridis.gif")

# ----- Load projections (HxWxN assumed) -----
H, W, N = projs.shape

# ----- 等間隔で60枚抽出 -----
frame_indices = np.linspace(0, N - 1, num_frames, endpoint=True, dtype=int)
frames = projs[:, :, frame_indices]  # shape = (H, W, 60)

# ----- 正規化（float） -----
frames_norm = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)

# ----- カラーマップ適用（viridis） -----
cmap = cm.get_cmap("viridis")
frames_color = []
for i in range(num_frames):
    color_frame = cmap(frames_norm[:, :, i])[:, :, :3]  # RGBA → RGB
    frames_color.append((color_frame * 255).astype(np.uint8))

# ----- GIF 保存 -----
duration = 1.0 / fps
iio.imwrite(output_path, frames_color, duration=duration, loop=0)

# ----- ログ出力 -----
print(f"🎨 GIF saved with viridis colormap: {output_path}")
print(f"- Total original frames : {N}")
print(f"- Extracted frames      : {num_frames}")
print(f"- Sampling indices      : {frame_indices}")
print(f"- fps                  : {fps} (duration={duration:.3f}s)")
print(f"- Total duration       : {num_frames / fps:.2f} sec")