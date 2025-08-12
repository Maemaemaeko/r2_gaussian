from pathlib import Path
import torch
import sys
import os

sys.path.append("./")
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "gaussian-splatting"))  # gaussian-splattingを追加


from scene import Scene

from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import Namespace
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


from scene.cameras import Camera
from PIL import Image
import numpy as np

from image2pose import Image2Pose
import cv2

class GaussianSceneRenderer:
    def __init__(self, source_path, model_path, ply_path, data_device: str = "cuda"):
        parser = ArgumentParser(description="Read Gaussian model from file")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        args = Namespace(
            sh_degree = 3,
            source_path = source_path,
            model_path = model_path,
            images = "images",
            depths = "",
            resolution = -1,
            white_background = False,
            train_test_exp = False,
            eval = False,
            debug = False,
            antialiasing = False,
            convert_SHs_python = False,
            compute_cov3D_python = False
        )

        self.dataset = model.extract(args)
        self.pipeline = pipeline.extract(args)

        bg_color = [1,1,1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=data_device)


        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.gaussians.load_ply(ply_path)

    
    def get_eye_view(self, charuco_tf, charuco_center, eye_position: str = "top", height=4):
        if eye_position == "top":
            _r = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # 上からの視点
            _t = np.array([0, 0, height])

        elif eye_position == "lookatobject":
            origin = charuco_tf[:3, 3]
            normal_vector = charuco_center.ravel()
            def project_onto_plane(vec, normal):
                normal = normal / np.linalg.norm(normal)
                return vec - np.dot(vec, normal) * normal

            x_axis = project_onto_plane(charuco_tf[:3, 0], normal_vector)
            y_axis = project_onto_plane(charuco_tf[:3, 1], normal_vector)
            z_axis = normal_vector / np.linalg.norm(normal_vector)
            _r = np.array([-x_axis, y_axis, -z_axis]).T
            print("Rotation matrix:\n", _r)
            _t = origin + z_axis * height
            print("Look at object position:", _t)

        elif eye_position == "board":
            origin = charuco_tf[:3, 3]
            x_axis = charuco_tf[:3, 0]
            y_axis = charuco_tf[:3, 1]
            z_axis = charuco_tf[:3, 2]
            # z_axisの大きさを正規化
            z_axis = z_axis / np.linalg.norm(z_axis)
            _r = np.array([-x_axis, y_axis, -z_axis]).T
            print("Rotation matrix:\n", _r)
            _t = origin + z_axis * height

        return Camera(
            colmap_id=87,
            R = _r,
            T = _t,
            FoVx=0.5565993180102227,
            FoVy=0.5565993180102227,
            image_name="none",
            uid=15,
            resolution=(512, 512),
            depth_params = None,
            image=Image.new('RGB', (512, 512), (0, 0, 0)),
            invdepthmap=None
        )

    def render_gaussians(self, charuco_tf, charuco_center, height=4, eye_position: str = "top", interactive=False):
        view = self.get_eye_view(charuco_tf, charuco_center, eye_position, height=height)
        print(view.world_view_transform)

        with torch.no_grad():
            rendering = render(view, self.gaussians, self.pipeline, self.background, use_trained_exp=self.dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]


        return rendering

        

if __name__ == "__main__":
    renderer = GaussianSceneRenderer(
        source_path="../gaussian-splatting/data/bigfoot",
        model_path="../gaussian-splatting/output/6bd50db1-7",
        ply_path="../gaussian-splatting/output/6bd50db1-7/point_cloud/iteration_30000/point_cloud_v5.ply"
    )

    image2pose = Image2Pose(Path("C:\\Users\\Maemaeko\\imari_lab\\r2_gaussian\\volunme_slicing_display\\camera_calibration"))
    image2pose.read_intrinsics()
    image = Path(r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\debug\capture_20250726_162456.png")
    image = cv2.imread(str(image))
    marker_corners, marker_ids = image2pose.detect_markers(image)
    rvec, tvec = image2pose.detect_charuco(image, marker_corners, marker_ids)
    tvec -= np.array([[0], [0], [0.5]])
    charuco_tf, _ = image2pose.output_transform(rvec, tvec)
    charuco_center = image2pose.output_charuco_center(rvec, tvec)

    rendering = renderer.render_gaussians(charuco_tf, charuco_center, height=4, eye_position="board")
    cv2.imshow("Rendering", rendering.permute(1, 2, 0).cpu().numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   