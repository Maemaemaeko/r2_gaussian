import os
import os.path as osp
import numpy as np
import sys

sys.path.append("./")
from r2_gaussian.utils.plot_utils import show_one_volume

proj_path = "/home/maemaeko/imari_lab/r2_gaussian/data/real_dataset/cone_ntrain_75_angle_360/teapot/proj_train"
proj_list = sorted(os.listdir(proj_path))

projs = np.stack(
    [np.load(osp.join(proj_path, proj_id)) for proj_id in proj_list], axis=-1
)

show_one_volume(projs)
