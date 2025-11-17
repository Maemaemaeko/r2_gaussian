import cv2
import os
import numpy as np
import imageio


output_dir = "output/teapot_cone_10"  # ← あなたの output フォルダへのパス
output_mp4 = os.path.join(output_dir, "combined_video.mp4")
output_gif = os.path.join(output_dir, "combined_video.gif")
# ==== 入力フォルダの設定 ====
folder1 = os.path.join(output_dir, "renderings_z")  # ← 比較対象の rendering フォルダへのパス
folder2 = os.path.join(output_dir, "visualization")       # ← あなたの rendering フォルダへのパス

# ==== 出力動画ファイル ====
output_path = os.path.join(output_dir, "combined_video.mp4")

# ==== 画像リストを取得 ====
files1 = sorted([f for f in os.listdir(folder1) if f.endswith(".png")])
files2 = sorted([f for f in os.listdir(folder2) if f.endswith(".png")])

# 同じ枚数であることを確認
assert len(files1) == len(files2), "フォルダ内の画像枚数が一致しません！"

# 最初の画像からサイズを取得
img1 = cv2.imread(os.path.join(folder1, files1[0]))
print(img1.shape)
img2 = cv2.imread(os.path.join(folder2, files2[0]))
print(img2.shape)

# サイズが違う場合は合わせる（高さを基準にスケール）
if img1.shape[0] != img2.shape[0]:
    h = min(img1.shape[0], img2.shape[0])
    scale1 = h / img1.shape[0]
    scale2 = h / img2.shape[0]
    img1 = cv2.resize(img1, (int(img1.shape[1] * scale1), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * scale2), h))

# 横方向に連結したときのサイズを計算
height = img1.shape[0]
width = img1.shape[1] + img2.shape[1]

# ==== 動画ライターを準備 ====
fps = 3  # 1秒あたりのフレーム数（必要に応じて変更）
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# ==== GIF用にフレームを蓄積 ====
frames_for_gif = []

# ==== 各フレーム処理 ====
for f1, f2 in zip(files1, files2):
    img1 = cv2.imread(os.path.join(folder1, f1))
    img2 = cv2.imread(os.path.join(folder2, f2))

    if img1.shape[0] != img2.shape[0]:
        h = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
        img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

    # 横に連結
    combined = np.hstack((img2, img1))

    # ---- MP4出力 ----
    writer.write(combined)

    # ---- GIF用にRGB変換して蓄積 ----
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    frames_for_gif.append(combined_rgb)

# ==== 書き出し ====
writer.release()
print(f"✅ MP4動画を保存しました: {output_mp4}")

# ==== GIF保存 ====
# duration: 1フレームあたりの表示秒数 (1/fps 秒)
imageio.mimsave(output_gif, frames_for_gif, duration=2000/fps)
print(f"✅ GIFも保存しました: {output_gif}")