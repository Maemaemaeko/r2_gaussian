import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact

mode = "pred" # or gt

# 画像が入っているフォルダ
img_dir = Path("/home/maemaeko/imari_lab/r2_gaussian/output/aEupholus_A_CT-38/test/iter_30000/render_test")

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
    path = os.path.join(img_dir, f"{idx:05}_pred.png")
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



# 動画設定
fps = 30  # お好みで (15, 24, 30, 60 など)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

# 各フレーム書き込み
for img_path in images:
    img = cv2.imread(str(img_path))
    writer.write(img)

writer.release()
print(f"✅ 動画を書き出しました: {output_video}")
