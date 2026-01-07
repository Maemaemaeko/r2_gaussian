
from pathlib import Path
import struct
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def read_traw3d_ss(path, *, assume_little_endian=True):
    """
    Read a .traw3D_ss file:
      int32 W,H,D
      float64 px,py,pz
      int16 volume (W*H*D)
    Returns:
      vol   : np.ndarray with shape (D, H, W), dtype=float32 (raw values, not normalized)
      pitch : (px, py, pz) in same units asファイル（たぶん mm）
      header: dict with W,H,D and raw bytes read
    """


    path = Path(path)
    with path.open("rb") as f:
        # 読み取りエンディアン
        endian = "<" if assume_little_endian else ">"

        # ヘッダ読む
        W, H, D = struct.unpack(endian + "iii", f.read(12))
        px, py, pz = struct.unpack(endian + "ddd", f.read(8 * 3))

        # 体積データ読む（int16）
        count = W * H * D
        buf = f.read(2 * count)
        if len(buf) != 2 * count:
            raise IOError(f"Unexpected EOF: need {2*count} bytes, got {len(buf)}")

        vol = np.frombuffer(buf, dtype=np.int16)
        # Cの典型的な格納（xが最速）: z * (W*H) + y * W + x
        vol = vol.reshape((D, H, W))  # 形状は(D,H,W)にしておく
        vol = vol.astype(np.float32, copy=False)

    header = {"W": W, "H": H, "D": D, "px": px, "py": py, "pz": pz}
    pitch = (px, py, pz)
    return vol, pitch, header

def safe_minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """min-max正規化。スカラー/定数配列のときはゼロ配列で返す。"""
    a_min = float(arr.min())
    a_max = float(arr.max())
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max == a_min:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - a_min) / (a_max - a_min)


def expand_to_cube(array):
    # Step 1: Find the maximum dimension
    max_dim = max(array.shape)

    # Step 2: Calculate the padding for each dimension
    padding = [(max_dim - s) // 2 for s in array.shape]
    # For odd differences, add an extra padding at the end
    padding = [(pad, max_dim - s - pad) for pad, s in zip(padding, array.shape)]

    # Step 3: Pad the array to get the cubic shape
    cubic_array = np.pad(
        array, padding, mode="constant", constant_values=0
    )  # Using zero padding

    return cubic_array


def crop_to_cube(array):
    # Step 1: Find the minimum dimension
    min_dim = min(array.shape)

    # Step 2: Define the start and end indices for cropping
    start_indices = [(dim_size - min_dim) // 2 for dim_size in array.shape]
    end_indices = [start + min_dim for start in start_indices]

    # Step 3: Crop the array to get the cubic region
    cubic_region = array[
        start_indices[0] : end_indices[0],
        start_indices[1] : end_indices[1],
        start_indices[2] : end_indices[2],
    ]

    return cubic_region


def resample(image, spacing, new_spacing=[1, 1, 1]):
    """Resample to stantard spacing (keep physical scale stable, change pixel numbers)"""
    # .mhd image order : z, y, x
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)

    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = ndimage.zoom(image, real_resize_factor, mode="nearest")
    return image, new_spacing



def resize(scan, target_size):
    """Resize the scan based on given voxel dimension."""
    scan_x, scan_y, scan_z = scan.shape
    zoom_x = target_size / scan_x
    zoom_y = target_size / scan_y
    zoom_z = target_size / scan_z

    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        scan = ndimage.zoom(
            scan,
            (zoom_x, zoom_y, zoom_z),
            mode="nearest",
        )
    return scan


def reshape_vol(image, spacing=[1, 1, 1], target_size=256, mode="expand"):
    """Reshape a CT volume."""

    if mode is not None:
        image, _ = resample(image, spacing, [1, 1, 1])
        if mode == "crop":
            image = crop_to_cube(image)
        elif mode == "expand":
            image = expand_to_cube(image)
        else:
            raise ValueError("Unsupported reshape mode!")

    image_new = resize(image, target_size)
    return image_new



def process_traw3d_ss(case_info, target_size=256):
    """
    .traw3D_ss を読み込んで、process_tif と同等の前処理を行う。

    Parameters
    ----------
    case_info: dict
        {
          "raw_path": str | Path,
          # 下記は .tif 時代のAPI互換のため:
          # "reshape": str | dict  -> reshape_vol に渡すモード
          # "transpose": tuple     -> 例 (2,1,0) など
          # "z_invert": bool       -> TrueならZ反転
        }
        spacing はファイルヘッダ(px,py,pz)を使用するため指定不要
    target_size: tuple[int,int,int]
        reshape_vol の出力ボクセル数 (D,H,W) など

    Returns
    -------
    data: np.ndarray, float32
    """
    # 1) 読み込み
    vol, pitch, _hdr = read_traw3d_ss(case_info)
    spacing = pitch  # (px,py,pz)

    # パーセンタイルで外れ値除去
    low, high = np.percentile(vol, [10, 99])
    data = np.clip(vol, low, high)

    data = vol
  


    data = safe_minmax_normalize(data)
    print(data.min(), data.max())
    print(data.shape)

    data = reshape_vol(
        data
    )

    data = data.transpose([2, 0, 1])
    print(data.min(), data.max())
    print(data.shape)

    data = data.clip(0.0, 1.0)
    


    # plt.figure(figsize=(6, 4))
    # plt.hist(data.flatten(), bins=200, color='steelblue', edgecolor='black', alpha=0.7)
    # plt.title("Voxel Intensity Histogram")
    # plt.xlabel("Normalized Intensity")
    # plt.ylabel("Count")
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # 3) リシェイプ（リサンプリング）
    # ここは既存の reshape_vol をそのまま使う前提
    # 期待する引数: (data, spacing, target_size, mode=str|dict)
    # data = reshape_vol(
    #     data, spacing, target_size, mode=case_info.get("reshape", None)
    # )

    # # 5) 軸入替え（例: (D,H,W)->(H,W,D) など）
    # if "transpose" in case_info and case_info["transpose"] is not None:
    #     data = data.transpose(case_info["transpose"])

    # # 6) Z反転（最後の軸がZだと仮定。transpose後のZがどの軸かに注意）
    # if case_info.get("z_invert", False):
    #     data = data[..., ::-1]

    plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
    plotter.add_volume(data, cmap="viridis", opacity="linear")
    plotter.show()

    return data




if __name__ == "__main__":
    #read_traw3d_ss("/home/maemaeko/imari_lab/r2_gaussian/data_generator/synthetic_dataset/volume_raw/aEupholus_A_CT.traw3D_ss")
    vol_out = process_traw3d_ss("/home/maemaeko/imari_lab/r2_gaussian/data_generator/synthetic_dataset/volume_raw/aEupholus_A_CT.traw3D_ss")
    output_path = "/home/maemaeko/imari_lab/r2_gaussian/data_generator/synthetic_dataset/volume_processed/aEupholus_A_CT.npy"
    np.save(output_path, vol_out.astype(np.float32))
    
