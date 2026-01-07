import os
import glob
import numpy as np
import tigre
from tigre.algorithms import fdk
from tigre.geometry import Geometry
from tigre.utilities import sample_loader

import imageio.v2 as imageio  # for PNG/JPG 読み込み

# -----------------------------
# 1. 入力データの設定
# -----------------------------
projs_dir = "/path/to/xray_images"   # X線投影画像のフォルダ
file_list = sorted(glob.glob(os.path.join(projs_dir, "*.png")))

# 画像を読み込み → numpy 配列化
projs = []
for f in file_list:
    img = imageio.imread(f).astype(np.float32)
    img = img / 65535.0 if img.max() > 1 else img / img.max()  # 正規化
    img = -np.log(np.clip(img, 1e-6, None))  # 吸収変換
    projs.append(img)

# shape: (n_proj, nDetectorY, nDetectorX)
projs = np.stack(projs, axis=0)

# -----------------------------
# 2. 幾何設定 (例: コーンビーム)
# -----------------------------
geo = Geometry()
geo.mode = 'cone'

# 投影画像サイズ
geo.nDetector = np.array([projs.shape[2], projs.shape[1]])   # [nDetectorX, nDetectorY]
geo.dDetector = np.array([1.0, 1.0])  # [sizeX, sizeY] mm/ピクセル (実データに合わせて設定)
geo.sDetector = geo.nDetector * geo.dDetector

# 体積サイズ
geo.nVoxel = np.array([256, 256, 256])     # 再構成ボクセル数
geo.sVoxel = np.array([256, 256, 256])     # mm 単位の体積サイズ

geo.DSD = 1000  # source-detector distance [mm]
geo.DSO = 500   # source-object distance [mm]

# -----------------------------
# 3. 角度の設定
# -----------------------------
angles = np.linspace(0, 2 * np.pi, projs.shape[0], endpoint=False)

# -----------------------------
# 4. FDK 再構成
# -----------------------------
print("Start FDK reconstruction...")
recon = fdk(projs, geo, angles)
print("Reconstruction done:", recon.shape)

# 保存
np.save("ct_volume.npy", recon)
