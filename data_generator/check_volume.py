import numpy as np
import pyvista as pv

vol_path = "data/real_dataset/cone_ntrain_75_angle_360/teapot/vol_gt.npy"


vol = np.load(vol_path)

plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(vol, cmap="viridis", opacity="linear")
plotter.show()
