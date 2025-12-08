import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact
import imageio.v2 as imageio


import numpy as np
import imageio.v2 as imageio
import matplotlib.cm as cm
import cv2


mode = "pred" # or gt

# 画像が入っているフォルダ
img_dir = Path("/home/maemaeko/imari_lab/r2_gaussian/output/synthetic_dataset/cone_ntrain_3_angle_360-wo-densification/2_backpack_cone/test/iter_10000/reconstruction")

# 出力ファイル
output_video = img_dir / f"reconstruction_{mode}.mp4"

# 画像リストをソートして取得
images = sorted([f for f in img_dir.glob(f"*_{mode}.png")])

# 最初の画像でサイズ取得
frame = cv2.imread(str(images[0]))
h, w, _ = frame.shape


import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
# 画像フォルダ
# 画像を読み込んでTkinter用に変換する関数
def load_image(idx):
    path = os.path.join(img_dir, f"{idx:05}_{mode}.png")
    if os.path.exists(path):
        img = Image.open(path)
        img = img.resize((512, 512))  # 表示サイズを適宜調整
        return ImageTk.PhotoImage(img)
    else:
        return None
# スライダーが動いたときに画像を更新
def update_image(val):
    idx = int(float(val))
    img = load_image(idx)
    if img:
        label.config(image=img)
        label.image = img
        status_label.config(text=f"表示中: {idx}_pred.png")
    else:
        label.config(image="")
        label.image = None
        status_label.config(text=f"{idx}_pred.png は存在しません")
# Tkinterウィンドウ作成
root = tk.Tk()
root.title("画像スライダー")
# 最初の画像
initial_idx = 0
photo = load_image(initial_idx)
# 画像表示ラベル
label = tk.Label(root, image=photo)
label.pack(padx=10, pady=10)
# スライダー
slider = ttk.Scale(root, from_=0, to=100, orient="horizontal", command=update_image)
slider.pack(fill="x", padx=20)
# ステータス表示
status_label = tk.Label(root, text=f"表示中: {initial_idx}_pred.png")
status_label.pack(pady=5)
root.mainloop()


def save_as_gif_with_cmap(images, output_gif, fps=10, cmap_name="viridis"):
    frames = []

    # matplotlib の colormap を取得
    cmap = cm.get_cmap(cmap_name)

    for img_path in images:
        # 画像読み込み（カラー or グレースケール OK）
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # 0〜255 → 0〜1 正規化
        img_norm = img.astype(np.float32) / 255.0

        # カラーマップ適用 → RGBA（0〜1）
        img_cmap = cmap(img_norm)

        # RGBA → RGB（不要なalphaを削除） + 0〜255 変換
        img_rgb = (img_cmap[:, :, :3] * 255).astype(np.uint8)

        frames.append(img_rgb)

    # GIF 保存
    imageio.mimsave(
        output_gif,
        frames,
        duration=1.0 / fps,
    )
    print(f"🎨 GIF を保存しました（cmap={cmap_name}）: {output_gif}")



# 出力ファイル
output_video = img_dir.parent / f"reconstruction_{mode}.gif"

save_as_gif_with_cmap(
    images,
    output_video,
    fps=90,
    cmap_name="viridis"
)



# # --- MP4 保存 ---
# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

# for img_path in images:
#     img = cv2.imread(str(img_path))
#     writer.write(img)

# writer.release()
# print(f"🎥 MP4 を書き出しました: {output_video}")

# # --- GIF 保存 ---
# gif_path = "output.gif"
# frames = [imageio.imread(str(p)) for p in images]
# imageio.mimsave(gif_path, frames, duration=1.0/fps)
# print(f"🎉 GIF も保存しました: {gif_path}")