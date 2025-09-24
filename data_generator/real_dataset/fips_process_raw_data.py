import os
import shutil
import numpy as np
import scipy.io as sio  # for saving .mat
from pathlib import Path

from create_ct_project import create_ct_project  # 先ほど作った関数をimport


def fips_process_raw_data(data_path: str, data_save_path: str):
    """
    Python equivalent of fips_process_raw_data.m
    """

    data_path = Path(data_path)
    data_save_path = Path(data_save_path)

    # --- Copy config file (.txt → config.txt)
    data_save_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(data_path) + ".txt", data_save_path / "config.txt")

    # --- Read data using create_ct_project
    CtData = create_ct_project(str(data_path), "3D")
    sinogram = CtData["sinogram"]

    # --- Save data slice by slice
    n_imgs = sinogram.shape[1]
    for i in range(n_imgs):
        img = np.squeeze(sinogram[:, i, :]).T
        img_id = f"{i+1:04d}"  # 0001, 0002, ...
        out_file = data_save_path / f"{img_id}.mat"
        sio.savemat(out_file, {"img": img})
        print(f"Saving image {i+1}/{n_imgs}")


if __name__ == "__main__":
    # === Example usage ===
    # data_path = "FIPS_raw/pine/20201118_pine_cone_"
    # data_save_path = "FIPS_processed/pine"
    # data_path = "FIPS_raw/seashell/20211124_seashell_"
    # data_save_path = "FIPS_processed/seashell"
    # data_path = "FIPS_raw/walnut/20201111_walnut_"
    # data_save_path = "FIPS_processed/walnut"

    # data_path = "/home/maemaeko/imari_lab/r2_gaussian/data_generator/real_dataset/FIPS_raw/pine/20201118_pine_cone_"
    # data_save_path = "/home/maemaeko/imari_lab/r2_gaussian/data_generator/real_dataset/FIPS_processed/pine"
    data_path = "/home/maemaeko/imari_lab/r2_gaussian/data_generator/real_dataset/WebCT_raw/dragon/20250918_dragon_cone_"
    data_save_path = "/home/maemaeko/imari_lab/r2_gaussian/data_generator/real_dataset/WebCT_processed/dragon"

    fips_process_raw_data(data_path, data_save_path)
