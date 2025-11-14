# epi_interpolate_npy_cv.py
import numpy as np
import cv2
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

# ---------- I/O ----------
def load_npy_f32(path):
    arr = np.load(path).astype(np.float32)
    return arr

def save_npy_and_png(out_dir, idx, img01, prefix="proj_train"):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{idx:04d}"

    # npy保存（元スケールのまま）
    np.save(out_dir / f"{stem}.npy", img01.astype(np.float32))
    print(img01.min(), img01.max())

    plt.imsave(str(out_dir / f"{stem}.png"), img01, cmap='gray',  vmin=0.0, vmax=1.0)
    

# ---------- 変換 ----------
def to_log_space(I, I0=None):
    if I0 is None:
        # 上位5%平均をフラット場近似に
        flat = np.sort(I.reshape(-1))[-max(100, I.size // 20):].mean()
        I0 = np.full_like(I, flat, dtype=np.float32)
    I0 = np.clip(I0, 1e-6, 1.0)
    L = -np.log(np.clip(I / I0, 1e-6, 1.0))
    return L, I0

def from_log_space(L, I0):
    return np.clip(I0 * np.exp(-np.clip(L, 0, None)), 0.0, 1.0)

def gaussian_lowpass_rows(img, ksize=5, sigma=1.0):
    if ksize <= 1: return img
    return cv2.GaussianBlur(img, (ksize | 1, 1), sigmaX=sigma, sigmaY=0)

# ---------- フロー ----------
def tvl1_flow(L1, L2,
              scales=5,           # ピラミッド段数
              scale_step=0.5,     # 各段の縮小率
              warps=5,
              eps=0.01,
              tau=0.25,
              lam=0.15,
              inner=30,
              outer=10,
              gamma=0.0,
              median_filtering=5):
    """
    OpenCV contrib の Dual TV-L1 を使って L1→L2 のフローを推定
    返り値: flow (H,W,2)  [dx, dy]
    """
    tvl1 = None
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        tvl1.setTau(tau)
        tvl1.setLambda(lam)
        tvl1.setScalesNumber(scales)     # ピラミッド段数
        tvl1.setScaleStep(scale_step)    # 縮小率（例: 0.5）
        tvl1.setWarpingsNumber(warps)
        tvl1.setEpsilon(eps)
        tvl1.setInnerIterations(inner)
        tvl1.setOuterIterations(outer)
        tvl1.setGamma(gamma)
        tvl1.setMedianFiltering(median_filtering)
        return tvl1.calc(L1, L2, None)

    # フォールバック: Farneback（contrib が無い場合）
    flow = cv2.calcOpticalFlowFarneback(
        L1, L2, None,
        pyr_scale=0.5, levels=scales, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0
    )
    return flow

# ---------- ワープ（OpenCV remap） ----------
def remap_with_flow(img, flow_xy):
    H, W = img.shape
    # flow: [dx, dy]（xが列方向、yが行方向）
    dx = flow_xy[..., 0].astype(np.float32)
    dy = flow_xy[..., 1].astype(np.float32)
    # OpenCVは (x, y) = (col, row) マップを要求
    x = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    y = np.tile(np.arange(H, dtype=np.float32).reshape(-1, 1), (1, W))
    map_x = x + dx
    map_y = y + dy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

# ---------- 中間投影 ----------
def interpolate_middle_projection(I1, I2, do_row_lowpass=True):
    # 1) L空間
    # L1, I0 = to_log_space(I1)
    # L2, _  = to_log_space(I2, I0)
    L1 = I1
    L2 = I2

    # 2) 行方向ローパス（ノイズ抑制）
    if do_row_lowpass:
        L1f = gaussian_lowpass_rows(L1, ksize=5, sigma=1.0)
        L2f = gaussian_lowpass_rows(L2, ksize=5, sigma=1.0)
    else:
        L1f, L2f = L1, L2

    # 3) L1→L2 のフロー
    flow = tvl1_flow(L1f, L2f)

    # 4) 中間角度へ半分ワープ（両側）
    L1_mid = remap_with_flow(L1,  0.5 * flow)
    L2_mid = remap_with_flow(L2, -0.5 * flow)

    # 5) L空間で保守的に合成（max）
    Lmid = np.maximum(L1_mid, L2_mid)

    # 6) 画像空間に戻す
    #Imid = from_log_space(Lmid, I0)
    return Lmid

# ---------- 指標（RMSE/PSNR/SSIM） ----------
def rmse(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def psnr(a, b, data_range=1.0):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

def ssim_gray(img1, img2, data_range=1.0, K1=0.01, K2=0.03, win_size=11, sigma=1.5):
    """
    単チャネル SSIM（OpenCV の GaussianBlur でウィンドウ）
    入力は [0,1] を想定
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 近似: ガウシアンで局所平均
    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), sigma) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1 * img2, (win_size, win_size), sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(ssim_map.mean())

# ---------- シーケンス処理 ----------
def interpolate_sequence_npy(input_dir, out_dir, prefix="proj_train"):
    in_dir  = Path(input_dir)
    out_dir = Path(out_dir)
    files = sorted(in_dir.glob("*.npy"))
    if len(files) < 2:
        raise RuntimeError("2枚以上の .npy ファイルが必要です。")

    out_idx = 0
    for i in range(len(files) - 1):
        I1 = load_npy_f32(files[i])
        I2 = load_npy_f32(files[i + 1])

        # もとのフレーム
        save_npy_and_png(out_dir, out_idx, I1, prefix=prefix); out_idx += 1
        # 中間補間
        Imid = interpolate_middle_projection(I1, I2)
        save_npy_and_png(out_dir, out_idx, Imid, prefix=prefix); out_idx += 1

    # 最後のフレーム
    save_npy_and_png(out_dir, out_idx, load_npy_f32(files[-1]), prefix=prefix)
    total = out_idx + 1
    print(f"✅ Interpolation Done. Wrote {total} frames to {out_dir}")
    return total  # 生成枚数（例：入力Nなら 2*N-1）

# ---------- 評価（GT があれば） ----------
def evaluate_against_gt(out_dir, gt_dir, prefix="proj_train"):
    out_dir = Path(out_dir)
    gt_dir  = Path(gt_dir)

    # 出力側の連番（存在するものだけ評価）
    out_npys = sorted(out_dir.glob(f"{prefix}_*.npy"))
    if len(out_npys) == 0:
        print("⚠️ 評価対象が見つかりません。まず出力を生成してください。")
        return

    rows = [("index", "filename", "RMSE", "PSNR", "SSIM")]
    rmse_list, psnr_list, ssim_list = [], [], []

    for p in out_npys:
        name = p.stem  # e.g., "proj_train_0042"
        idx_str = name.split("_")[-1]
        idx = int(idx_str)
        if idx % 2 == 1:

            # GT 側の候補: 同じ接頭辞 or 接頭辞なし
            gt_candidates = [
                gt_dir / f"{prefix}_{idx:04d}.npy",
                gt_dir / f"{idx:04d}.npy",
            ]
            gt_path = next((g for g in gt_candidates if g.exists()), None)
            if gt_path is None:
                # 見つからなければスキップ
                continue

            pred = load_npy_f32(p)
            gt   = load_npy_f32(gt_path)
            if pred.shape != gt.shape:
                # 必要ならリサイズ（最近傍 or bilinear）。ここでは安全に bilinear。
                gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)
                gt = gt_resized

            r = rmse(pred, gt)
            ps = psnr(pred, gt, data_range=1.0)
            ss = ssim_gray(pred, gt, data_range=1.0)

            rows.append((idx, p.name, r, ps, ss))
            rmse_list.append(r); psnr_list.append(ps); ssim_list.append(ss)

    # CSV 保存
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        # 末尾に平均
        if rmse_list:
            writer.writerow([])
            writer.writerow(["mean", "-", np.mean(rmse_list), np.mean(psnr_list), np.mean(ssim_list)])

    # 画面表示
    if rmse_list:
        print(f"📊 Evaluation on {len(rmse_list)} frames")
        print(f"   RMSE: {np.mean(rmse_list):.6f}")
        print(f"   PSNR: {np.mean(psnr_list):.3f} dB")
        print(f"   SSIM: {np.mean(ssim_list):.4f}")
        print(f"   → {csv_path}")
    else:
        print("⚠️ GT と一致するファイル名が見つからず、評価できませんでした。")

# ---------- メイン ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="入力 .npy（疎）を含むフォルダ")
    ap.add_argument("--out_dir", required=True, help="補間後を書き出すフォルダ")
    ap.add_argument("--gt_dir", default=None, help="GT フォルダ（任意）。与えると定量評価を実施")
    ap.add_argument("--prefix", default="proj_pred", help="出力ファイル接頭辞（既定: proj_train）")
    args = ap.parse_args()

    interpolate_sequence_npy(args.in_dir, args.out_dir, prefix=args.prefix)

    if args.gt_dir is not None:
        evaluate_against_gt(args.out_dir, args.gt_dir, prefix=args.prefix)


