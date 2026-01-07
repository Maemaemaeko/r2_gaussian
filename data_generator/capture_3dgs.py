import sys
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
import os.path as osp
import copy
import open3d as o3d
import yaml
from pathlib import Path
import imageio.v3 as iio


sys.path.append("./")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.arguments import PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.plot_utils import create_textured_camera, create_vol_mesh
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.graphics_utils import fov2focal

from argparse import ArgumentParser, Namespace

from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
)

import cv2
from cv2 import aruco
from pathlib import Path

class R2GaussianSceneRenderer:
    def __init__(self, source_path: str = "../data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone", model_path: str = "output/95e359ad-b", data_device: str = "cuda"):
        parser = ArgumentParser(description="Read Gaussian model from file")
        model = ModelParams(parser, sentinel=True)
        self.pipeline = PipelineParams(parser)
        # Namespace で引数を手動指定
        args = Namespace(
            source_path=source_path,
            model_path=model_path,
            data_device=data_device,
            scale_min=0.0005,
            scale_max=0.5,
            eval=True
        )

        dataset = model.extract(args)
        self.source_path = source_path
        self.gaussians = GaussianModel(None)
        self.model_path = model_path
        loaded_iter = initialize_gaussian(self.gaussians, dataset, loaded_iter=-1) 

    def look_at_transform(
            self,
            center,
            lookat,
            up,
        ):
        z = lookat - center

        z = z / np.linalg.norm(z)

        x = np.cross(up, z)
        if np.all(x == 0):
            x = np.cross(up, z + np.array([1e-6, 1e-6, 1e-6]))  # Avoid zero vector
        x = x / np.linalg.norm(x)

        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)
        t = -R.T @ center
        view_matrix = np.eye(4)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = t
        return view_matrix


    # === 軌道生成のコア ===
    def orbit_positions(self,axis: str, n_views: int, radius: float, center):
        axis = axis.lower()
        thetas = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)
        center = np.asarray(center, dtype=float)

        if axis == "z":
            pts = np.stack([radius * np.cos(thetas),
                            radius * np.sin(thetas),
                            np.zeros_like(thetas)], axis=1)
        elif axis == "x":
            pts = np.stack([np.zeros_like(thetas),
                            radius * np.cos(thetas),
                            radius * np.sin(thetas)], axis=1)
        elif axis == "y":
            pts = np.stack([radius * np.cos(thetas),
                            np.zeros_like(thetas),
                            radius * np.sin(thetas)], axis=1)
        else:
            raise ValueError("axis must be one of {'x','y','z'}")
        return center[None, :] + pts

    def orbit_cameras_around_axis(
        self,
        axis: str = "z",
        n_views: int = 60,
        radius: float = 5.0,
        center = (0.0, 0.0, 0.0),
        FoVx: float = 0.5565993180102227,
        FoVy: float = 0.5565993180102227,
        angle: float = 0.8366643629487514,
        image_hw: int = 512,
        colmap_id_start: int = 100,
        uid_start: int = 0,
        mode: int = 1,
        image_name_prefix: str = "orbit",
    ):
        # 軸に合わせた up（ロール安定用）
        up_map = {"x": np.array([1, 0, 0], float),
                "y": np.array([0, 1, 0], float),
                "z": np.array([0, 0, -1], float)}
        axis = axis.lower()
        if axis not in up_map:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        up = up_map[axis]

        center = np.asarray(center, dtype=float)
        positions = self.orbit_positions(axis, n_views, radius, center)

        cams = []
        for k, pos in enumerate(positions):
            # look-at: 注視点は軌道中心
            w2c = self.look_at_transform(center=pos, lookat=center, up=up)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            cams.append(
                Camera(
                    colmap_id=colmap_id_start + k,
                    scanner_cfg=None,
                    R=R,
                    T=T,
                    angle=angle,
                    mode=mode,
                    FoVx=FoVx,
                    FoVy=FoVy,
                    image=torch.zeros((1, image_hw, image_hw)),
                    image_name=f"{image_name_prefix}_{axis}_{k:03d}.png",
                    uid=uid_start + k,
                )
            )
        return cams



    
if __name__ == "__main__":
    # Example usage
    renderer = R2GaussianSceneRenderer(
        source_path="/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/aEupholus_A_CT_cone",
        model_path="/home/maemaeko/imari_lab/r2_gaussian/output/synthetic_dataset/cone_ntrain_9_angle_360-wo-densification/0_foot_cone",
        #model_path="../output/95e359ad-b",
        data_device="cuda"
    )

    cams = renderer.orbit_cameras_around_axis(
        axis="z")
    print(f"Generated {len(cams)} cameras")

    renderings = []
    rendering_output_dir = Path(renderer.model_path) / "renderings_z"
    rendering_output_dir.mkdir(exist_ok=True, parents=True)
    for i, view in enumerate(cams):
        rendering = render(view, renderer.gaussians, renderer.pipeline)["render"][0].detach().cpu().numpy()
        renderings.append(rendering)
        plt.imsave(rendering_output_dir / f"{i:03d}.png", rendering)
    
    # PNGが保存されているフォルダ
    rendering_output_dir = Path(renderer.model_path) / "renderings_z"
    output_path = Path(renderer.model_path) / "simple_orbit_z.gif"

    # フレームを読み込む
    files = sorted(rendering_output_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png files found in {rendering_output_dir}")

    frames = [iio.imread(str(f)) for f in files]

    # # GIFとして保存
    fps = 10       # 再生速度
    iio.imwrite(output_path, frames, duration=1.0/fps, loop=0)

    print(f"✅ Saved simple GIF: {output_path}")
    print(f"frames: {len(frames)}, fps: {fps}")
