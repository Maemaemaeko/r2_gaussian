import yaml
import numpy as np
import csv
import matplotlib.pyplot as plt

def load(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# 2つのファイルパスを指定
file_A = "/home/maemaeko/imari_lab/r2_gaussian/output/synthetic_dataset/cone_ntrain_3_angle_360-smoothness-v2-wo-densification/0_head_cone/eval_360/iter_10000/eval2d_render_all.yml"
file_B = "/home/maemaeko/imari_lab/r2_gaussian/output/head-5-smoothness/eval_360/iter_10000/eval2d_render_all.yml"

A = load(file_A)
B = load(file_B)

# ------ Prepare data ------
psnrA = np.array(A["psnr_2d_projs"])
psnrB = np.array(B["psnr_2d_projs"])
ssimA = np.array(A["ssim_2d_projs"])
ssimB = np.array(B["ssim_2d_projs"])

diff_psnr = psnrB - psnrA
diff_ssim = ssimB - ssimA

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

    for i, d in enumerate(diff_psnr):
        flag = "IMPROVED" if d > 0 else ""
        writer.writerow([i, psnrA[i], psnrB[i], d, flag])

print(f"Saved: {csv_path}")

# ------ Plot per-frame PSNR / SSIM ------

indices = np.arange(len(psnrA))

plt.figure(figsize=(12, 5))

# PSNR per frame
plt.subplot(1, 2, 1)
plt.plot(indices, psnrA, label="PSNR A (wo densif.)")
plt.plot(indices, psnrB, label="PSNR B (smoothness-v2)")
plt.xlabel("Projection index")
plt.ylabel("PSNR [dB]")
plt.title("Per-frame PSNR")
plt.legend()
plt.grid(True, alpha=0.3)

# SSIM per frame
plt.subplot(1, 2, 2)
plt.plot(indices, ssimA, label="SSIM A (wo densif.)")
plt.plot(indices, ssimB, label="SSIM B (smoothness-v2)")
plt.xlabel("Projection index")
plt.ylabel("SSIM")
plt.title("Per-frame SSIM")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("per_frame_psnr_ssim.png", dpi=300)
plt.show()

# （おまけ）差分を見たい場合
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(indices, diff_psnr)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Projection index")
plt.ylabel("ΔPSNR (B - A) [dB]")
plt.title("Per-frame PSNR difference")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(indices, diff_ssim)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Projection index")
plt.ylabel("ΔSSIM (B - A)")
plt.title("Per-frame SSIM difference")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("per_frame_diff_psnr_ssim.png", dpi=300)
plt.show()
