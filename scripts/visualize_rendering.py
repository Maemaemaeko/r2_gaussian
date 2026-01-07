import os
import os.path as osp
import torch
import sys
from argparse import ArgumentParser
import numpy as np
import open3d as o3d
import matplotlib


sys.path.append("./")
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.plot_utils import create_textured_camera, create_vol_mesh
from r2_gaussian.utils.graphics_utils import fov2focal
from r2_gaussian.utils.general_utils import t2a

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
        arguments = Namespace(
            source_path=source_path,
            model_path=model_path,
            data_device=data_device,
            scale_min=0.0005,
            scale_max=0.5,
            eval=True
        )

        dataset = model.extract(arguments)
        self.source_path = source_path
        self.gaussians = GaussianModel(None)
        self.model_path = model_path
        loaded_iter = initialize_gaussian(self.gaussians, dataset, -1)

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
        n_views: int =60,
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

def main(dataset: ModelParams, args):
    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    scanner_cfg = scene.scanner_cfg

    vol_mesh = create_vol_mesh(
        np.load(osp.join(dataset.source_path, "vol_gt.npy")),
        np.array(scanner_cfg["offOrigin"]),
        np.array(scanner_cfg["dVoxel"]),
        np.eye(3),
        level=args.mc_thresh,
    )

    vol_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scanner_cfg["sVoxel"][0] / 2,
        origin=scanner_cfg["offOrigin"],
    )
    vol_bbox = o3d.geometry.OrientedBoundingBox(
        center=scanner_cfg["offOrigin"],
        R=np.eye(3),
        extent=scanner_cfg["sVoxel"],
    )
    vol_bbox.color = np.array([0, 0, 1])

    unit_bbox = o3d.geometry.OrientedBoundingBox(
        center=[0, 0, 0], R=np.eye(3), extent=[2, 2, 2]
    )
    unit_bbox.color = np.array([0, 0, 1])

    cams = []
    cmap = matplotlib.colormaps["viridis"]
    cam_scale = args.cam_scale
    n_proj = len(scene.train_cameras)

    renderer = R2GaussianSceneRenderer(
        source_path=dataset.source_path,
        model_path="/home/maemaeko/imari_lab/r2_gaussian/output/teapot_cone_10",
        #model_path="../output/95e359ad-b",
        data_device="cuda"
    )
    cameras = renderer.orbit_cameras_around_axis(
        axis="z")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D Scene",
        width=1080,
        height=1080,
        visible=True
    )
    # 2) 変わらないジオメトリは最初に一度だけ追加
    static_assets = [vol_mesh, vol_bbox, vol_coord, unit_bbox]
    for g in static_assets:
        vis.add_geometry(g)


    ctr = vis.get_view_control()

    # 原点固定＆ズームを最初に決めて「保存」
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(2)  # ← ここで好みのスケールに固定
    vis.poll_events(); vis.update_renderer()

    # ← この“固定視点”を保存しておく
    fixed_params = ctr.convert_to_pinhole_camera_parameters()

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    # 3) カメラ用のジオメトリは毎フレームで追加→撮影→削除
    visualization_dir = os.path.join(renderer.model_path, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)

    for i_proj, camera in enumerate(cameras):
        proj_name = "not_used"
        proj_id = i_proj
        proj = render(camera, renderer.gaussians, renderer.pipeline)["render"][0].detach().cpu().numpy()
        K = np.array(
            [
                [fov2focal(camera.FoVx, proj.shape[1]), 0, proj.shape[1] / 2],
                [0, fov2focal(camera.FoVy, proj.shape[0]), proj.shape[0] / 2],
                [0, 0, 1],
            ]
        )
        w2c = np.eye(4)
        w2c[:3, :3] = t2a(camera.R.T)
        w2c[:3, 3] = t2a(camera.T)
        c2w = np.linalg.inv(w2c)
        DSO = np.linalg.norm(c2w[:3, 3] - np.array(scanner_cfg["offOrigin"]))
        cam = create_textured_camera(
            K,
            w2c,
            cam_scale,
            (0, 0, 1), #cmap(i_proj / n_proj)[:3],
            proj.shape[1],
            proj.shape[0],
            f"{proj_id:03d}",
            proj,
        )
        # 追加（camはList[Geometry]想定）
        for g in cam:
            vis.add_geometry(g)

        # 追加・更新の後に、毎フレーム “同じ視点” を強制的に復元
        ctr.convert_from_pinhole_camera_parameters(fixed_params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

        save_path = os.path.join(visualization_dir, f"{i_proj:03d}.png")
        vis.capture_screen_image(save_path, do_render=True)
        print(f"✅ Screenshot saved to: {save_path}")

        # カメラ用ジオメトリだけを削除（staticは残す）
        for g in cam:
            vis.remove_geometry(g, reset_bounding_box=False)

        # 終了
    vis.destroy_window()


    # vis_assets = cams + [vol_mesh, vol_bbox, vol_coord, unit_bbox]
    # o3d.visualization.draw_geometries(vis_assets, mesh_show_back_face=True)
    # --- ここから変更 ---
    # Visualizerを作成
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(
    #     window_name="Open3D Scene",
    #     width=1920,
    #     height=1080,
    #     visible=True
    # )

    # # すべてのジオメトリを追加
    # for asset in vis_assets:
    #     vis.add_geometry(asset)

    # # 描画を更新
    # vis.poll_events()
    # vis.update_renderer()

    # # スクリーンショット保存パス
    # screenshot_path = os.path.join("scene_screenshot.png")

    # # スクリーンショットを保存
    # vis.capture_screen_image(screenshot_path, do_render=True)
    # print(f"✅ Screenshot saved to: {screenshot_path}")

    # # 終了
    # vis.destroy_window()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    parser.add_argument("--mc_thresh", type=float, default=0.2, help="Threshold of marching cubes for mesh extraction from volume.")
    parser.add_argument("--cam_scale", type=float, default=1.0, help="Size of camera model for visualization")
    args = parser.parse_args(sys.argv[1:])
    main(lp.extract(args), args)
