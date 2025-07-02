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
    def __init__(self, source_path: str = "..data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone", model_path: str = "output/95e359ad-b", data_device: str = "cuda"):
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
        loaded_iter = initialize_gaussian(self.gaussians, dataset, -1)

        #self.scene = Scene(dataset, shuffle=False)


    def get_eye_view(self, charuco_tf, charuco_center, eye_position: str = "top", height=4):
        if eye_position == "top":
            _r = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # 上からの視点
            _t = np.array([0, 0, height])

        elif eye_position == "lookatobject":
            origin = charuco_tf[:3, 3]
            normal_vector = charuco_center
            def project_onto_plane(vec, normal):
                normal = normal / np.linalg.norm(normal)
                return vec - np.dot(vec, normal) * normal
        
            x_axis = project_onto_plane(charuco_tf[:3, 0], normal_vector)
            y_axis = project_onto_plane(charuco_tf[:3, 1], normal_vector)
            z_axis = normal_vector / np.linalg.norm(normal_vector)
            _r = np.array([y_axis, x_axis, -z_axis]).T
            _t = origin + z_axis * height
            print("Look at object position:", _t)

        elif eye_position == "board":
            origin = charuco_tf[:3, 3]
            x_axis = charuco_tf[:3, 0]
            y_axis = charuco_tf[:3, 1]
            z_axis = charuco_tf[:3, 2]
            # z_axisの大きさを正規化
            z_axis = z_axis / np.linalg.norm(z_axis)
            _r = np.array([y_axis, x_axis, -z_axis]).T
            _t = origin + z_axis * height

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

    def render_gaussians(self, charuco_tf, charuco_center, d=0.1, eye_position: str = "top", cut_method: str="by_plane", interactive=False):
        view = self.get_eye_view(charuco_tf, charuco_center, eye_position)
        with torch.no_grad():
            cut_gaussians = copy.deepcopy(self.gaussians)
            if cut_method == "by_plane":
                self.cut_gaussians_by_plane(cut_gaussians, charuco_tf, d)
            elif cut_method == "beyond_plane":
                self.cut_gaussians_beyond_plane(cut_gaussians, charuco_tf, d)

            rendering = render(view, self.gaussians, self.pipeline)["render"][0].detach().cpu().numpy()
            rendering_cut = render(view, cut_gaussians, self.pipeline)["render"][0].detach().cpu().numpy()

            if interactive:
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

    def visualize(self, charuco_tf, charuco_center, scanner_cfg="../data_generator/synthetic_dataset/scanner/cone_beam.yml", eye_position="top", rendering=None, interactive=False):
            camera = self.get_eye_view(charuco_tf, charuco_center, eye_position)
            # w2cをRとTから計算
            w2c = np.eye(4)
            w2c[:3, :3] = camera.R.T
            w2c[:3, 3] = camera.T
            

            with open(scanner_cfg, "r") as handle:
                scanner_cfg = yaml.safe_load(handle)

            vol_mesh = create_vol_mesh(
                np.load(osp.join(self.source_path, "vol_gt.npy")),
                np.array(scanner_cfg["offOrigin"]),
                np.array(scanner_cfg["sVoxel"]) / np.array(scanner_cfg["nVoxel"]),
                np.eye(3),
                level = 0.4
            )


            vol_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size = scanner_cfg["sVoxel"][0] / 2,
                origin = scanner_cfg["offOrigin"],
            )

            vol_bbox = o3d.geometry.OrientedBoundingBox(
                center = scanner_cfg["offOrigin"],
                R = np.eye(3),
                extent = scanner_cfg["sVoxel"],
            )

            vol_bbox.color = np.array([1, 0, 0]) # 赤色

            FoVx = camera.FoVx
            FoVy = camera.FoVy
            W = 512
            H = 512

            K = np.array(
                [
                    [fov2focal(FoVx, W), 0, W / 2],
                    [0, fov2focal(FoVy, H), H / 2],
                    [0, 0, 1],
                ]
            )
            cam = create_textured_camera(
                K,
                w2c,
                2,  # cam_scale
                np.array([0.5, 0.5, 0.5]),  # color
                W,
                H,
                "eye_camera",  # image_name
                rendering
            )
           
            # charucoボードを可視化
            charuco_size = (2, 2)
            w, h = charuco_size
            half_w, half_h = w / 2, h / 2
            local_corners = np.array([
                [-half_w, -half_h, 0],
                [half_w, -half_h, 0],
                [half_w, half_h, 0],
                [-half_w, half_h, 0]
            ])

            local_corners_h = np.hstack([local_corners, np.ones((4, 1))])  # → (4, 4)
            world_corners = (charuco_tf @ local_corners_h.T).T[:, :3]     # → (4, 3)


            lines = [
              [0, 1], [1, 2], [2, 3], [3, 0],  # 外周
                [0, 2], [1, 3]  # 対角線
            ]
            colors = [[0, 0, 1] for _ in range(len(lines))]  # 青色
            # Step 1: 空のLineSetを作る
            line_box = o3d.geometry.LineSet()

            # Step 2: プロパティを個別に代入する（キーワード引数ではなく）
            line_box.points = o3d.utility.Vector3dVector(world_corners)
            line_box.lines = o3d.utility.Vector2iVector(lines)
            line_box.colors = o3d.utility.Vector3dVector(colors)

            vis_assets = [vol_mesh, vol_coord, vol_bbox] + cam + [line_box]

            image = None
            if interactive:
                o3d.visualization.draw_geometries(vis_assets, mesh_show_back_face=True)
                # For debugging
                #self.save_camera_extrinsic(vis_assets)
                #self.test_pose_change(vis_assets)

            else:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=512, height=512)
                for asset in vis_assets:
                    vis.add_geometry(asset)
                vis.get_render_option().mesh_show_back_face = True

                vis.poll_events()
                vis.update_renderer()

                ctr = vis.get_view_control()
                ctr.set_lookat([0.0, 0.0, 0.5])   # 注視点
                ctr.set_front ([-1, 0.0, 0.0])  # カメラから見た前方向ベクトル
                ctr.set_up    ([0.0, 0.0, 1.0])  # カメラの上向きベクトル
                ctr.set_zoom  (0.8)          # ズーム倍率（1.0 で等倍）

       
                vis.poll_events()
                vis.update_renderer()

                image = vis.capture_screen_float_buffer(do_render=True)
                vis.destroy_window()

                # np.asarray(image) を可視化
                image = np.asarray(image)
                # opencvで画像を表示
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.imshow("Rendered Image", image)
                # cv2.waitKey(0)
               
            return image

    def save_camera_extrinsic(self, vis_assets, save_path="camera.json"):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for asset in vis_assets:
            vis.add_geometry(asset)

        # GUIを表示して視点をユーザーが調整
        vis.run()

        # カメラパラメータを取得
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()

        # extrinsic を表示・保存
        extrinsic = cam_param.extrinsic
        print("Extrinsic matrix (world to camera):\n", extrinsic)

        # JSON保存（必要であれば）
        o3d.io.write_pinhole_camera_parameters(save_path, cam_param)
        print(f"Camera parameters saved to: {save_path}")

        vis.destroy_window()

    def test_pose_change(self, vis_assets):
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, width=1920, height=1080)
        for asset in vis_assets:
            vis.add_geometry(asset)

        # 背面ポリゴンも表示
        vis.get_render_option().mesh_show_back_face = True

        # 一度イベントループを回して内部状態を整える
        vis.poll_events()
        vis.update_renderer()

        # ——— カメラパラメータ取得 & 置き換え ———rol()
        ctr = vis.get_view_control()
        ctr.set_lookat([0.0, 0.0, 0.0])   # 注視点
        ctr.set_front ([-1, 0.0, 0.0])  # カメラから見た前方向ベクトル
        ctr.set_up    ([0.0, 0.0, 1.0])  # カメラの上向きベクトル
        ctr.set_zoom  (0.7)               # ズーム倍率（1.0 で等倍）
        # 4) 再レンダリング → キャプチャ
        vis.poll_events()
        vis.update_renderer()

        # ——— キャプチャ & Window 終了 ———
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # NumPy 配列化 → BGR に変換 → OpenCV で表示
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Rendered Image", image)
        cv2.waitKey(0)



if __name__ == "__main__":
    # Example usage
    renderer = R2GaussianSceneRenderer(
        source_path="../data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
        model_path="../output/95e359ad-b",
        data_device="cuda"
    )
    charuco_path =  "charuco_camera_transformation.npz"
    charuco_tf = np.load(charuco_path)["charuco_tf"]
    charuco_center = np.array([-0.00781352, -0.01640093, 0.4141831])
    #renderer.plot_board_transform(charuco_tf, charuco_center, colors=('r', 'g', 'b'), eye_position="lookatobject")
    eye_position = "board"  # or "top", "lookatobject", "board"
    cut_method = "by_plane"  # or "by_plane"
    d = 0.1

    rendering, rendering_cut = renderer.render_gaussians(charuco_tf, charuco_center, d, eye_position, cut_method)

    renderer.visualize(charuco_tf, charuco_center, rendering=rendering_cut, eye_position=eye_position, interactive=False)