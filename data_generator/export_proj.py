import os
import os.path as osp
import numpy as np
import sys
import imageio.v3 as iio

sys.path.append("./")
from r2_gaussian.utils.plot_utils import show_one_volume

proj_path = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_75_angle_360/teapot/proj_train"
proj_list = sorted(os.listdir(proj_path))

projs = np.stack(
    [np.load(osp.join(proj_path, proj_id)) for proj_id in proj_list], axis=-1
)

arr = projs.astype(np.float32)

# 1) 全フレーム共通レンジで正規化（外れ値を抑えるため百分位でクリップ）
lo, hi = np.percentile(arr, (1.0, 99.5))
arr = np.clip(arr, lo, hi)
arr = ((arr - lo) / (hi - lo + 1e-7) * 255.0).astype(np.uint8)  # -> 8bit

frames = arr.transpose(2, 0, 1)

print(frames.shape)

fps = 15
iio.imwrite("projs.gif", frames, duration=1.0/fps, loop=0)
print("Saved: projs.gif, frames:", frames.shape[0], "fps:", fps)