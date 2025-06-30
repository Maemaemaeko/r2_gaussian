import sys
import torch
import math
import numpy as np

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
        scene  = Scene(dataset)
        gaussians = GaussianModel(None)
        loaded_iter = initialize_gaussian(gaussians, dataset, -1)
        scene.gaussians = gaussians

        return None
    
    def get_camera_transform(self, charuco_tf, charuco_center, camera_position: str = "top"):
        if camera_position == "top":
            _r = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # 上からの視点
            _t = np.array([0, 0, 3])
        
        elif camera_position == "lookatobject":
            normal_vector = charuco_center
            def project_onto_plane()
        

    def plot_camera_transform(self, charuco_tf, )


