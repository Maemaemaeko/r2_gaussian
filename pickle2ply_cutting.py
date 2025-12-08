import pickle
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUIなし環境でもOKにする
import matplotlib.pyplot as plt

def filter_gaussians_by_scale_ratio(
    in_pickle_path,
    out_pickle_path=None,
    ratio_thresh=5.0,
    also_filter_by_opacity=False,
    opacity_min=None,
):
    """
    scale比 (max(scale) / min(scale)) が大きすぎる Gaussian を削除して、
    新しい pickle を保存する。

    Parameters
    ----------
    in_pickle_path : str or Path
        元の point_cloud.pickle パス
    out_pickle_path : str or Path, optional
        出力する pickle パス（未指定なら suffix に `_filtered` を付けて保存）
    ratio_thresh : float
        この値より大きい scale 比を持つ Gaussian を削除する
        例: 5.0 → max(scale) / min(scale) > 5 の Gaussian を除外
    also_filter_by_opacity : bool
        True の場合、opacity_min も使ってさらにフィルタ
    opacity_min : float
        also_filter_by_opacity=True のとき、density(opacity) がこれ未満のものも削除
    """
    in_pickle_path = Path(in_pickle_path)
    if out_pickle_path is None:
        out_pickle_path = in_pickle_path.with_name(in_pickle_path.stem + "_filtered.pickle")
    else:
        out_pickle_path = Path(out_pickle_path)

    # 1) pickle 読み込み
    with open(in_pickle_path, "rb") as f:
        data = pickle.load(f)

    # 必須フィールドの確認
    if "scale" not in data:
        raise KeyError("pickle 内に 'scale' キーがありません。r2_gaussian の point_cloud.pickle を指定していますか？")
    if "xyz" not in data:
        raise KeyError("pickle 内に 'xyz' キーがありません。")

    scale = np.asarray(data["scale"])   # (N, 3) を想定
    N = scale.shape[0]
    print(f"[INFO] 元の Gaussian 数: {N}")

    # 2) scale 比を計算（max / min）
    #    念のため abs を取って、min が 0 に近いときは小さな値でクリップ
    scale_abs = np.abs(scale)
    scale_min = np.min(scale_abs, axis=1)
    scale_max = np.max(scale_abs, axis=1)
    eps = 1e-8
    scale_ratio = scale_max / np.maximum(scale_min, eps)  # shape (N,)

    # 3) マスクを作る（ratio_thresh 以下を残す）
    keep_mask = scale_ratio > ratio_thresh

    print(f"[INFO] ratio_thresh = {ratio_thresh}")
    print(f"[INFO] scale_ratio min/max = {scale_ratio.min():.4f} / {scale_ratio.max():.4f}")
    print(f"[INFO] ratio_thresh を超えて削除される Gaussian 数: {(~keep_mask).sum()}")

    # 追加で opacity でもフィルタしたい場合（任意）
    if also_filter_by_opacity:
        if "density" not in data:
            raise KeyError("also_filter_by_opacity=True ですが 'density' キーがありません。")
        density = np.asarray(data["density"]).reshape(-1)  # (N,) 化
        if opacity_min is None:
            raise ValueError("also_filter_by_opacity=True のときは opacity_min を指定してください。")

        opacity_mask = density >= opacity_min
        print(f"[INFO] opacity_min = {opacity_min}")
        print(f"[INFO] opacity_min を下回って削除される Gaussian 数: {(~opacity_mask).sum()}")

        keep_mask = keep_mask & opacity_mask

    N_new = keep_mask.sum()
    print(f"[INFO] フィルタ後の Gaussian 数: {N_new}")

    # 4) data 内の「N次元先頭が N の配列」を全部マスクして更新
    new_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.shape[0] == N:
                new_data[k] = v[keep_mask]
                # デバッグ用に shape を表示したければ:
                # print(f"[DEBUG] key={k}, shape {v.shape} → {new_data[k].shape}")
            else:
                # そのままコピー（例えば metadata など）
                new_data[k] = v
        else:
            # np.ndarray 以外（list, dict, float など）はそのままコピー
            new_data[k] = v

    # 5) 新しい pickle を保存
    with open(out_pickle_path, "wb") as f:
        pickle.dump(new_data, f)

    print(f"[INFO] 保存しました: {out_pickle_path}")
    return out_pickle_path

def analyze_xy_distribution(
    pickle_path,
    num_bins=64,
    clip_quantile=0.995,
    save_fig_path_counts=None,
    save_fig_path_mean_density=None,
    density_key="density",
):
    """
    xyz の (x, y) 分布が、xy平面上でどこに偏っているかを調べる。
    さらに、Gaussian の density の平均値が xy 平面上でどう偏っているかも調べる。

    - 2D ヒストグラムを作り、
      - 各セルの Gaussian 個数
      - 各セル内の density 平均値
    などを出力する。
    - オプションでヒートマップ画像も保存する。

    Parameters
    ----------
    pickle_path : str or Path
        point_cloud.pickle のパス
    num_bins : int
        x, y 方向それぞれのビンの数
    clip_quantile : float
        外れ値を少し切るためのクオンタイル（例: 0.995）
        これより外側の x, y はヒートマップの範囲外として無視する
    save_fig_path_counts : str or Path, optional
        Gaussian 個数ヒートマップの保存先
    save_fig_path_mean_density : str or Path, optional
        density 平均値ヒートマップの保存先
    density_key : str
        pickle 内で density が格納されているキー名（通常 "density"）
    """
    pickle_path = Path(pickle_path)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if "xyz" not in data:
        raise KeyError("pickle 内に 'xyz' キーがありません。")

    xyz = np.asarray(data["xyz"])
    x = xyz[:, 0]
    y = xyz[:, 1]
    N = xyz.shape[0]
    print(f"[INFO] Gaussian 数 (N): {N}")

    # --- density（あれば） ---
    has_density = density_key in data
    if has_density:
        density = np.asarray(data[density_key]).reshape(-1)
        if density.shape[0] != N:
            raise ValueError(f"'{density_key}' の長さが xyz と一致しません: {density.shape[0]} vs {N}")
        print(f"[INFO] density キー '{density_key}' を使用します。")
    else:
        print(f"[WARN] pickle 内に '{density_key}' がありません。density の解析はスキップします。")
        density = None

    # --- 外れ値対策として、両端を少しクリップ（任意） ---
    x_min, x_max = np.quantile(x, [1 - clip_quantile, clip_quantile])
    y_min, y_max = np.quantile(y, [1 - clip_quantile, clip_quantile])

    print(f"[INFO] x range (clipped): {x_min:.4f} ~ {x_max:.4f}")
    print(f"[INFO] y range (clipped): {y_min:.4f} ~ {y_max:.4f}")

    # --- 2D ヒストグラム（個数） ---
    H_counts, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=num_bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    # H_counts.shape == (num_bins, num_bins)

    counts = H_counts.flatten()
    total_cells = counts.size
    non_empty_cells = np.count_nonzero(counts)

    mean_count_all = counts.mean()
    mean_count_non_empty = counts[counts > 0].mean() if non_empty_cells > 0 else 0.0
    max_count = counts.max()

    print(f"[INFO] 総セル数: {total_cells} (={num_bins}x{num_bins})")
    print(f"[INFO] 非ゼロセル数: {non_empty_cells}")
    print(f"[INFO] 1セルあたり平均カウント（全セル）: {mean_count_all:.2f}")
    print(f"[INFO] 1セルあたり平均カウント（非ゼロセルのみ）: {mean_count_non_empty:.2f}")
    print(f"[INFO] 最大カウント: {max_count:.2f}")
    if mean_count_non_empty > 0:
        print(f"[INFO] max / mean_non_empty = {max_count / mean_count_non_empty:.2f}")

    # --- 上位セル（Gaussian 密集セル）をいくつか表示 ---
    top_k = 10
    top_indices = np.argsort(counts)[::-1][:top_k]
    print(f"[INFO] Gaussian が多い上位 {top_k} セル:")
    for idx in top_indices:
        c = counts[idx]
        if c == 0:
            break
        ix = idx // num_bins
        iy = idx % num_bins
        x0, x1 = x_edges[ix], x_edges[ix + 1]
        y0, y1 = y_edges[iy], y_edges[iy + 1]
        print(
            f"  cell ({ix:2d}, {iy:2d}) "
            f"x∈[{x0:.3f}, {x1:.3f}), y∈[{y0:.3f}, {y1:.3f}) -> count={int(c)}"
        )

    # --- 個数ヒートマップの保存 ---
    if save_fig_path_counts is not None:
        save_fig_path_counts = Path(save_fig_path_counts)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            H_counts.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Gaussian count on xy-plane")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("count")

        fig.tight_layout()
        fig.savefig(save_fig_path_counts, dpi=300)
        plt.close(fig)
        print(f"[INFO] 個数ヒートマップを保存しました: {save_fig_path_counts}")

    # --- density 平均値の 2D 分布 ---
    mean_density_map = None
    if has_density:
        # 各セル内の density 合計
        H_sum_density, _, _ = np.histogram2d(
            x,
            y,
            bins=num_bins,
            range=[[x_min, x_max], [y_min, y_max]],
            weights=density,   # ここがポイント
        )
        # 平均値 = 合計 / 個数（ゼロ割防止）
        eps = 1e-8
        mean_density_map = H_sum_density / (H_counts + eps)

        # 簡単な統計
        valid_mask = H_counts > 0
        mean_density_valid = mean_density_map[valid_mask]
        print(f"[INFO] density 平均値（セル単位, 非ゼロセルのみ）: "
              f"min={mean_density_valid.min():.4f}, "
              f"max={mean_density_valid.max():.4f}, "
              f"mean={mean_density_valid.mean():.4f}")

        # density ヒートマップの保存
        if save_fig_path_mean_density is not None:
            save_fig_path_mean_density = Path(save_fig_path_mean_density)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                mean_density_map.T,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                aspect="equal",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Mean density on xy-plane")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("mean density")

            fig.tight_layout()
            fig.savefig(save_fig_path_mean_density, dpi=300)
            plt.close(fig)
            print(f"[INFO] density 平均値ヒートマップを保存しました: {save_fig_path_mean_density}")

    return {
        "hist_counts": H_counts,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "mean_density_map": mean_density_map,
    }

def visualize_xy_density_from_npy(
    npy_path,
    x_idx=0,
    y_idx=1,
    density_idx=3,
    num_bins=64,
    clip_quantile=0.995,
    save_fig_path_sum="xy_density_sum.png",
    save_fig_path_mean="xy_density_mean.png",
):
    """
    density point cloud (.npy) について、
    xy平面上の「density の分布」を 2D ヒートマップとして可視化する。

    Parameters
    ----------
    npy_path : str or Path
        point cloud の .npy ファイルパス。
        例: shape (N, 4) で [x, y, z, density] が入っているなど。
    x_idx, y_idx, density_idx : int
        配列のどの列を x, y, density とみなすか。
        例: [x, y, z, density] → x_idx=0, y_idx=1, density_idx=3
    num_bins : int
        x, y 方向それぞれのビン数。
    clip_quantile : float
        外れ値除去のためのクオンタイル。
        例: 0.995 → 下位0.5%, 上位0.5%を切り捨てて範囲決定。
    save_fig_path_sum : str or Path
        各セルの density 合計値ヒートマップの保存先。
    save_fig_path_mean : str or Path
        各セルの density 平均値ヒートマップの保存先。
    """
    npy_path = Path(npy_path)
    pts = np.load(npy_path)  # 例: shape (N, D)

    if pts.ndim != 2:
        raise ValueError(f"期待する形は (N, D) ですが、実際は {pts.shape} でした。")

    N, D = pts.shape
    print(f"[INFO] loaded {npy_path}, shape = {pts.shape}")

    if not (0 <= x_idx < D and 0 <= y_idx < D and 0 <= density_idx < D):
        raise ValueError("x_idx, y_idx, density_idx が配列の次元範囲外です。")

    x = pts[:, x_idx]
    y = pts[:, y_idx]
    density = pts[:, density_idx]
    print(f"[INFO] point count N = {N}")

    # --- 外れ値対策として範囲をクリップ ---
    x_min, x_max = np.quantile(x, [1 - clip_quantile, clip_quantile])
    y_min, y_max = np.quantile(y, [1 - clip_quantile, clip_quantile])
    print(f"[INFO] x range (clipped): {x_min:.4f} ~ {x_max:.4f}")
    print(f"[INFO] y range (clipped): {y_min:.4f} ~ {y_max:.4f}")

    # --- 2Dヒストグラム：点の個数 ---
    H_counts, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=num_bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )

    # --- 2Dヒストグラム：density の合計（weights） ---
    H_sum_density, _, _ = np.histogram2d(
        x,
        y,
        bins=num_bins,
        range=[[x_min, x_max], [y_min, y_max]],
        weights=density,
    )

    # --- density 平均値マップ（合計 / 個数） ---
    eps = 1e-8
    mean_density = H_sum_density / (H_counts + eps)

    # 統計をちょっとだけ表示
    valid_mask = H_counts > 0
    if np.any(valid_mask):
        md_valid = mean_density[valid_mask]
        print(
            f"[INFO] mean density per cell (non-empty cells): "
            f"min={md_valid.min():.4f}, max={md_valid.max():.4f}, mean={md_valid.mean():.4f}"
        )
    else:
        print("[WARN] 全セルが空です。ビン数 or 範囲を見直してください。")

    # === 1) density 合計のヒートマップ ===
    if save_fig_path_sum is not None:
        save_fig_path_sum = Path(save_fig_path_sum)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            H_sum_density.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Sum of density on xy-plane")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("sum density")
        fig.tight_layout()
        fig.savefig(save_fig_path_sum, dpi=300)
        plt.close(fig)
        print(f"[INFO] saved sum-density heatmap: {save_fig_path_sum}")

    # === 2) density 平均値のヒートマップ ===
    if save_fig_path_mean is not None:
        save_fig_path_mean = Path(save_fig_path_mean)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            mean_density.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Mean density on xy-plane")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("mean density")
        fig.tight_layout()
        fig.savefig(save_fig_path_mean, dpi=300)
        plt.close(fig)
        print(f"[INFO] saved mean-density heatmap: {save_fig_path_mean}")

    return {
        "H_counts": H_counts,
        "H_sum_density": H_sum_density,
        "mean_density": mean_density,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }

# if __name__ == "__main__":
#     input_pickle = "/home/maemaeko/imari_lab/r2_gaussian/output/synthetic_dataset/cone_ntrain_9_angle_360/1_pepper_cone/point_cloud/iteration_1000/point_cloud.pickle"


#     analyze_xy_distribution(
#         input_pickle,
#         num_bins=64,
#         clip_quantile=0.995,
#         save_fig_path_counts="xy_density_counts.png",
#         save_fig_path_mean_density="xy_mean_density.png",
#         density_key="density",  # r2-gaussian の log-density など
#     )
 

if __name__ == "__main__":
    npy_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_3_angle_360/0_head_cone/init_0_head_cone.npy"

    visualize_xy_density_from_npy(
        npy_path,
        x_idx=0,
        y_idx=1,
        density_idx=3,  # 必要に応じて変える
        num_bins=64,
        clip_quantile=0.995,
        save_fig_path_sum="xy_density_sum.png",
        save_fig_path_mean="xy_density_mean.png",
    )