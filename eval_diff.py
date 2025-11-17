import yaml
import numpy as np
import csv

def load(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# 2つのファイルパスを指定
file_A = "/home/maemaeko/imari_lab/r2_gaussian/output/1_pepper_cone-5-smoothness/test/iter_20000/eval2d_render_train.yml"
file_B = "/home/maemaeko/imari_lab/r2_gaussian/output/1_pepper_cone-5/eval_360/iter_10000/eval2d_render_alll.yml"

A = load(file_A)
B = load(file_B)

# ------ Prepare data ------
psnrA = np.array(A["psnr_2d_projs"])
psnrB = np.array(B["psnr_2d_projs"])
diff = psnrB - psnrA

# ------ Write CSV ------
csv_path = "diff.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    # Header
    writer.writerow(["Metric", "A", "B", "Difference", "Flag"])

    # Scalar values（PSNR/SSIM）
    for key in ["psnr_2d", "ssim_2d"]:
        da = A[key]
        db = B[key]
        df = db - da
        flag = "IMPROVED" if df > 0 else ""
        writer.writerow([key, da, db, df, flag])

    writer.writerow([])

    # Projection-level PSNR
    writer.writerow(["Index", "PSNR_A", "PSNR_B", "B - A", "Flag"])

    for i, d in enumerate(diff):
        flag = "IMPROVED" if d > 0 else ""
        writer.writerow([i, psnrA[i], psnrB[i], d, flag])

print(f"Saved: {csv_path}")