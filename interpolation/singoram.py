import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import csv

# ==== 入力/出力パス ====
base_dir = Path("/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone/proj_train")
out_root = base_dir.parent / "proj_train_interp_full"
(out_root / "pred").mkdir(parents=True, exist_ok=True)
(out_root / "gt").mkdir(parents=True, exist_ok=True)
(out_root / "even").mkdir(parents=True, exist_ok=True)
for sub in ["abs", "sq", "signed", "ssim1m"]:
    (out_root / "error_maps" / sub).mkdir(parents=True, exist_ok=True)

# ==== 角度・インデックス ====
n_proj = 75
angles = np.linspace(0, 360, n_proj, endpoint=False)
even_idx = np.arange(0, n_proj, 2)
odd_idx  = np.arange(1, n_proj, 2)

# ==== 全投影ロード ====
projs = [np.load(base_dir / f"proj_train_{i:04d}.npy").astype(np.float32) for i in range(n_proj)]
projs = np.stack(projs, axis=0)  # [75, H, W]
H, W = projs.shape[1:]

# ==== 偶数のみで補間関数 ====
interp_func = interp1d(
    angles[even_idx],
    projs[even_idx],
    axis=0,
    kind="linear",          # "cubic" でも可
    fill_value="extrapolate",
    assume_sorted=True
)

# ==== 偶数番の実測データも保存 ====
for idx in even_idx:
    gt_even = projs[idx]
    np.save(out_root / "even" / f"proj_even_{idx:04d}.npy", gt_even)
    plt.imsave(out_root / "even" / f"proj_even_{idx:04d}.png", gt_even, cmap='gray')

# ==== 結果CSV ====
csv_path = out_root / "odd_interp_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "angle_deg", "MAE", "RMSE", "PSNR_dB", "SSIM", "max_abs", "p95_abs"])

    # ==== 奇数番を補間・保存 ====
    for idx in odd_idx:
        gt   = projs[idx]
        pred = interp_func(angles[idx]).astype(np.float32)

        # 保存 (npy + png)
        np.save(out_root / "gt"   / f"proj_gt_{idx:04d}.npy", gt)
        np.save(out_root / "pred" / f"proj_pred_{idx:04d}.npy", pred)
        plt.imsave(out_root / "gt"   / f"proj_gt_{idx:04d}.png", gt, cmap='gray')
        plt.imsave(out_root / "pred" / f"proj_pred_{idx:04d}.png", pred, cmap='gray')

        # ---- 誤差計算 ----
        diff = pred - gt
        abs_err = np.abs(diff)
        sq_err = diff**2
        dr = float(gt.max() - gt.min()) if gt.max() != gt.min() else 1.0
        ssim_val, ssim_map = ssim(gt, pred, data_range=dr, full=True)
        ssim_err = 1.0 - ssim_map

        # ---- 指標 ----
        mae  = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(sq_err)))
        psnr_val = float(psnr(gt, pred, data_range=dr))
        max_abs = float(abs_err.max())
        p95_abs = float(np.percentile(abs_err, 95))
        writer.writerow([idx, float(angles[idx]), mae, rmse, psnr_val, float(ssim_val), max_abs, p95_abs])

        # ---- .npy保存 ----
        np.save(out_root / "error_maps" / "abs"    / f"err_abs_{idx:04d}.npy", abs_err)
        np.save(out_root / "error_maps" / "sq"     / f"err_sq_{idx:04d}.npy",  sq_err)
        np.save(out_root / "error_maps" / "signed" / f"err_signed_{idx:04d}.npy", diff)
        np.save(out_root / "error_maps" / "ssim1m" / f"err_ssim1m_{idx:04d}.npy", ssim_err)

        # ---- PNG誤差マップ ----
        vmax_abs = np.percentile(abs_err, 99)
        vmax_sq  = np.percentile(sq_err, 99)
        vm = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)))

        # ---- 誤差マップPNG保存（カラーバー付き） ----
        # 絶対誤差
        plt.figure(figsize=(4, 4))
        plt.imshow(abs_err, vmin=0, vmax=vmax_abs, cmap='inferno')
        plt.title(f"ABS Error (idx={idx}, angle={angles[idx]:.1f}°)")
        plt.colorbar(label="|Pred - GT|")
        plt.tight_layout()
        plt.savefig(out_root / "error_maps" / "abs" / f"err_abs_{idx:04d}.png", dpi=150)
        plt.close()

        # 二乗誤差
        plt.figure(figsize=(4, 4))
        plt.imshow(sq_err, vmin=0, vmax=vmax_sq, cmap='inferno')
        plt.title(f"SQ Error (idx={idx}, angle={angles[idx]:.1f}°)")
        plt.colorbar(label="(Pred - GT)^2")
        plt.tight_layout()
        plt.savefig(out_root / "error_maps" / "sq" / f"err_sq_{idx:04d}.png", dpi=150)
        plt.close()

        # 符号付き誤差
        plt.figure(figsize=(4, 4))
        plt.imshow(diff, vmin=-vm, vmax=vm, cmap='seismic')
        plt.title(f"Signed Error (idx={idx}, angle={angles[idx]:.1f}°)")
        plt.colorbar(label="Pred - GT")
        plt.tight_layout()
        plt.savefig(out_root / "error_maps" / "signed" / f"err_signed_{idx:04d}.png", dpi=150)
        plt.close()

        # 1 - SSIM 誤差マップ
        plt.figure(figsize=(4, 4))
        plt.imshow(ssim_err, vmin=0, vmax=1, cmap='magma')
        plt.title(f"1 - SSIM (idx={idx}, angle={angles[idx]:.1f}°)")
        plt.colorbar(label="1 - SSIM")
        plt.tight_layout()
        plt.savefig(out_root / "error_maps" / "ssim1m" / f"err_ssim1m_{idx:04d}.png", dpi=150)
        plt.close()

        print("✅ 完了！")
print(f"出力フォルダ: {out_root}")
print("保存内容:")
print("  pred/ : 補間された奇数投影 (npy/png)")
print("  gt/   : 実測の奇数投影 (npy/png)")
print("  even/ : 実測の偶数投影 (npy/png)")
print("  error_maps/ : 誤差マップ (abs, sq, signed, 1-SSIM)")
print(f"  指標CSV : {csv_path}")


