import pickle
import numpy as np
from pathlib import Path


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


# 使い方例
if __name__ == "__main__":
    input_pickle = "/home/maemaeko/imari_lab/r2_gaussian/output/synthetic_dataset/cone_ntrain_3_angle_360-wo-densification/0_foot_cone/point_cloud/iteration_10000/point_cloud.pickle"

    # スケール比が大きすぎる Gaussian を削除して新しい pickle を生成
    out_pickle = filter_gaussians_by_scale_ratio(
        input_pickle,
        ratio_thresh=1.5,          # ここはお好みで調整
        also_filter_by_opacity=False,  # 必要なら True に
        opacity_min=-5.0,          # density が log 空間ならこのへんを試すとか
    )

    # その後、可視化用に PLY も作るなら：
    # output_ply = out_pickle.with_suffix(".ply")
    # pickle_to_ply(out_pickle, output_ply)
