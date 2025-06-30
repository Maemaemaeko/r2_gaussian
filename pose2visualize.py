import sys
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
import copy

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.arguments import PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian

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
    def __init__(self, source_path: str = "data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone", model_path: str = "output/95e359ad-b", data_device: str = "cuda"):
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

        self.gaussians = GaussianModel(None)
        loaded_iter = initialize_gaussian(self.gaussians, dataset, -1)

        
    def get_eye_view(self, charuco_tf, charuco_center, eye_position: str = "top"):
        if eye_position == "top":
            _r = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # 上からの視点
            _t = np.array([0, 0, 3])

        elif eye_position == "lookatobject":
            origin = charuco_tf[:3, 3]
            normal_vector = charuco_center
            def project_onto_plane(vec, normal):
                normal = normal / np.linalg.norm(normal)
                return vec - np.dot(vec, normal) * normal
        
            x_axis = project_onto_plane(charuco_tf[:3, 0], normal_vector)
            y_axis = project_onto_plane(charuco_tf[:3, 1], normal_vector)
            z_axis = normal_vector
            _r = np.array([y_axis, x_axis, -z_axis]).T
            _t = origin + z_axis * 4


        return Camera(
            colmap_id = 65,
            scanner_cfg = None,
            R = _r,
            T = _t,
            angle=0.8366643629487514,
            mode=1,
            FoVx=0.5565993180102227,
            FoVy=0.5565993180102227,
            image=torch.zeros((1, 512, 512)),
            image_name="none",
            uid=15,
        )


    def plot_board_transform(self, charuco_tf, charuco_center, colors = ('r', 'g', 'b'), eye_position: str = "top"):
        camera_tf = np.eye(4)  # カメラの変換行列（単位行列）
        #eye_tf= self.get_eye_transform(charuco_tf, charuco_center, eye_position)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*charuco_center, color='orange', s=50, label='ChArUco Center')

        def plot_frame(ax, tf, label, color):
            origin = tf[:3, 3]
            x_axis, y_axis, z_axis = tf[:3, 0], tf[:3, 1], tf[:3, 2]

            ax.quiver(*origin, *x_axis, color=color[0], length=0.1, normalize=True)
            ax.quiver(*origin, *y_axis, color=color[1], length=0.1, normalize=True)
            ax.quiver(*origin, *z_axis, color=color[2], length=0.1, normalize=True)

            ax.text(*(origin + x_axis * 0.25), f'{label}_X', color=color[0])
            ax.text(*(origin + y_axis * 0.25), f'{label}_Y', color=color[1])
            ax.text(*(origin + z_axis * 0.25), f'{label}_Z', color=color[2])

        plot_frame(ax, charuco_tf, 'Charuco', colors)
        plot_frame(ax, camera_tf, 'Camera', colors)
        #plot_frame(ax, eye_tf, 'Eye', colors)

        # 軸ラベルとタイトル
        ax.set_xlabel('X (Eye)')
        ax.set_ylabel('Y (Eye)')
        ax.set_zlabel('Z (Eye)')
        ax.set_title('Camera and Eye Transformations')

        # アスペクト比を揃える
        points = np.vstack((charuco_center, camera_tf[:3, 3]))
        max_range = points.ptp(axis=0).max() / 2.0
        mid = points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # 表示
        plt.show()

    def cut_gaussians_by_plane(self, gaussians, charuco_tf, d=0.1):
        with torch.no_grad():
            origin = charuco_tf[:3, 3]
            normal = charuco_tf[:3, 2]

            # 正規化された法線ベクトル
            normal = normal / np.linalg.norm(normal)
            normal = torch.from_numpy(normal).to(gaussians._xyz.device, dtype=gaussians._xyz.dtype)

            # originもtensorに変換
            origin = torch.from_numpy(origin).to(gaussians._xyz.device, dtype=gaussians._xyz.dtype)

        # 各centerから平面までの符号付き距離を計算
        distance = torch.abs(torch.matmul(gaussians._xyz - origin, normal))

        mask  = distance > d
        gaussians._density[mask] = -100


    def cut_gaussians_beyond_plane(self, gaussians, charuco_tf, d=0.1):
        with torch.no_grad():
            origin = charuco_tf[:3, 3]
            normal = charuco_tf[:3, 2]
            # 正規化された法線ベクトル
            normal = normal / np.linalg.norm(normal)
            normal = torch.from_numpy(normal).to(gaussians._xyz.device, dtype=gaussians._xyz.dtype)

            # originもtensorに変換
            origin = torch.from_numpy(origin).to(gaussians._xyz.device, dtype=gaussians._xyz.dtype)

            # 各centerから平面までの符号付き距離を計算
            distance = torch.matmul(gaussians._xyz - origin, normal)

            mask  = distance < 0
            gaussians._density[mask] = -100

    def render_gaussians(self, charuco_tf, charuco_center, d=0.1, eye_position: str = "top", cut_method: str="by_plane"):
        view = self.get_eye_view(charuco_tf, charuco_center, eye_position)
        with torch.no_grad():
            cut_gaussians = copy.deepcopy(self.gaussians)
            if cut_method == "by_plane":
                self.cut_gaussians_by_plane(cut_gaussians, charuco_tf, d)
            elif cut_method == "beyond_plane":
                self.cut_gaussians_beyond_plane(cut_gaussians, charuco_tf, d)

            rendering = render(view, self.gaussians, self.pipeline)["render"][0].detach().cpu().numpy()
            rendering_cut = render(view, cut_gaussians, self.pipeline)["render"][0].detach().cpu().numpy()

            print("Rendering shape:", rendering.shape)
            print("Cut Rendering shape:", rendering_cut.shape)

            # 画像を表示
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(rendering)
            plt.title("Original Rendering")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(rendering_cut)
            plt.title(f"Rendering with {cut_method} (d={d})")
            plt.axis('off')
            plt.show()

        return rendering, rendering_cut


if __name__ == "__main__":
    # Example usage
    renderer = R2GaussianSceneRenderer(
        source_path="data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
        model_path="output/95e359ad-b",
        data_device="cuda"
    )
    charuco_path =  "volunme_slicing_display/charuco_camera_transformation.npz"
    charuco_tf = np.load(charuco_path)["charuco_tf"]
    charuco_center = np.array([-0.00781352, -0.01640093, 0.4141831])
    #renderer.plot_board_transform(charuco_tf, charuco_center, colors=('r', 'g', 'b'), eye_position="lookatobject")
    eye_position = "lookatobject"  # or "top", "lookatobject"
    cut_method = "beyond_plane"  # or "by_plane"
    d = 0.1

    rendering, rendering_cut = renderer.render_gaussians(charuco_tf, charuco_center, d, eye_position, cut_method)
