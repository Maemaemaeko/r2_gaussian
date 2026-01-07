import os
import pandas as pd

path_a = "/home/maemaeko/imari_lab/r2_gaussian/output/synthetic_dataset/cone_ntrain_9_angle_360-random-sampling-loss-wo-densification-wo-filter/0_foot_cone/test/iter_10000/per_slice_metrics.csv"
path_b = "/home/maemaeko/imari_lab/r2_gaussian/output/synthetic_dataset/cone_ntrain_9_angle_360-wo-densification/0_foot_cone/test/iter_10000/per_slice_metrics.csv"

A = pd.read_csv(path_a)
B = pd.read_csv(path_b)

# valid列があるなら、両方validのものだけ比較（任意）
if "valid" in A.columns and "valid" in B.columns:
    A = A[A["valid"] == True].copy()
    B = B[B["valid"] == True].copy()

# axis, slice で対応づけ
M = A.merge(B, on=["axis", "slice"], suffixes=("_A", "_B"), how="inner")

M["delta_psnr"] = M["psnr_B"] - M["psnr_A"]
M["delta_ssim"] = M["ssim_B"] - M["ssim_A"]

print(M[["delta_psnr","delta_ssim"]].describe())

out_path = os.path.join(os.path.dirname(path_a), "per_slice_metrics_A_to_B_delta.csv")
M.to_csv(out_path, index=False)
print("saved:", out_path)