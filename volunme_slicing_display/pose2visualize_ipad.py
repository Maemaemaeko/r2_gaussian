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
from image2pose import Image2Pose


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
from PIL import Image


def spherical_to_cartesian(theta, phi, radius):
    x  = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z])


def look_at_transform(
        center,
        lookat,
        up,
    ):
    z = lookat - center

    z = z / np.linalg.norm(z)
    print("z", z)

    x = np.cross(up, z)
    if np.all(x == 0):
        x = np.cross(up, z + np.array([1e-6, 1e-6, 1e-6]))  # Avoid zero vector
    x = x / np.linalg.norm(x)
    print("x", x)

    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    t = -R.T @ center
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t
    return view_matrix

def view_matrix_from_spherical(
        theta,
        phi,
        radius,
        lookat = np.array([0, 0, 0]),
        up = np.array([1, 0, 0])
    ):
    center = spherical_to_cartesian(theta, phi, radius)

    return look_at_transform(center, lookat, up)


import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def build_camera_axes(camera_pos, lookat=np.array([0, 0, 0]), mode="horizontal"):
    """
    mode: "horizontal" -> 左右移動で y_cam = world_x
          "vertical"   -> 上下移動で x_cam = world_y
    """

    # z-axis (forward direction)
    z_cam = normalize(lookat - camera_pos)

    world_x = np.array([1.0, 0.0, 0.0])
    world_y = np.array([0.0, 1.0, 0.0])

    if mode == "horizontal":
        # y_cam は常に world_x と一致
        y_cam = normalize(world_x)

        # z_cam と y_cam が近すぎると外積が死ぬ
        if abs(np.dot(z_cam, y_cam)) > 0.99:
            y_cam = normalize(world_y)

        x_cam = normalize(np.cross(y_cam, z_cam))
        y_cam = np.cross(z_cam, x_cam)

    elif mode == "vertical":
        # x_cam は常に world_y と一致
        x_cam = normalize(world_y)

        if abs(np.dot(z_cam, x_cam)) > 0.99:
            x_cam = normalize(world_x)

        y_cam = normalize(np.cross(z_cam, x_cam))
        x_cam = np.cross(y_cam, z_cam)

    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'")

    # 3×3 行列 R（列が x_cam, y_cam, z_cam）
    R = np.stack([x_cam, y_cam, z_cam], axis=1)  # c2w

    # T は -RC
    T = -R.T @ camera_pos # w2c


    return R, T

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)  # w2c
    return np.float32(Rt)


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
        loaded_iter = initialize_gaussian(self.gaussians, dataset, -1)

    def load_pose_csv(self, csv_path: Path, angle_unit: str = "rad"):
        """
        CSVから time, yaw, pitch, roll を読み込む。
        ヘッダ行に 'time', 'yaw', 'pitch', 'roll' の列がある想定。
        angle_unit: "rad" or "deg"
        """
        import pandas as pd
        # CSV 読み込み
        print(f"[app_recording] Loading pose CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[app_recording] CSV columns: {df.columns.tolist()}")
        t = df["time"].values
        # time_ns → 秒
        df["time_sec"] = df["time"] / 1e9
        # yaw/pitch/roll を取得
        timestamps = df["time_sec"].values
        yaws   = df["yaw"].values
        pitches = df["pitch"].values
        rolls  = df["roll"].values
        
        
        return yaws, rolls



    # def load_pose_csv(self, csv_path: Path, angle_unit: str = "rad"):
    #     """
    #     CSVから time, yaw, pitch, roll を読み込む。
    #     ヘッダ行に 'time', 'yaw', 'pitch', 'roll' の列がある想定。
    #     angle_unit: "rad" or "deg"
    #     """
    #     import pandas as pd
    #     # CSV 読み込み
    #     print(f"[app_recording] Loading pose CSV from: {csv_path}")
        
    #     df = pd.read_csv(csv_path)
    #     print(f"[app_recording] CSV columns: {df.columns.tolist()}")
    #     df.columns = df.columns.str.strip()

    #     yaws   = df["GyroY"].values
    #     rolls  = df["GyroX"].values
        
    #     return yaws, rolls

    def get_camera(self, yaw, roll,
                radius=5.0,
                yaw_gain=1.0,
                roll_gain=1.0,
                up=np.array([1, 0, 0])):
        """
        iPad のジャイロセンサーの yaw / roll から Camera を生成する。
        """

        if abs(yaw) > abs(roll): # 上下
            mode = "vertical"
            print("mode is vertical")
            camera_pos = np.array([yaw*yaw_gain, 0, radius])
        else: # 左右
            mode = "horizontal"
            camera_pos = np.array([0, roll*roll_gain, radius])
        R, T = build_camera_axes(camera_pos, mode=mode)

        return Camera(
            colmap_id=65,
            scanner_cfg=None,
            R = R,
            T = T, 
            angle=0.8366643629487514,
            mode=1,
            FoVx=0.5565993180102227,
            FoVy=0.5565993180102227,
            image=torch.zeros((1, 512, 512)),
            image_name="none",
            uid=15,
        )
    
    def render_gaussians(self, yaw, roll, interactive=False):
        view = self.get_camera(
            yaw=yaw,
            roll=roll,
            radius=5.0,       # 必要なら調整
            yaw_gain=1.0,
            roll_gain=1.0
        )
        # ---- Gaussian レンダリング ----
        rendering = render(
            view,
            renderer.gaussians,
            renderer.pipeline
        )["render"][0].detach().cpu().numpy()  # returned: (H, W, 3)


        # # rendering: (H, W, 3) or (H, W)
        # img = rendering
        # if img.ndim == 3:
        #     img_gray = img.mean(axis=-1)
        # else:
        #     img_gray = img

        # base_color = np.array([1.0, 0.0, 0.0])  # 赤

        # # 濃淡を掛ける
        # colored = img_gray[..., None] * base_color

        # plt.imshow(colored)
        # plt.axis("off")
        # plt.show()

        return rendering

    def visualize(self, yaw, roll, rendering, scanner_cfg="../data_generator/synthetic_dataset/scanner/cone_beam.yml", interactive=True):
            camera = self.get_camera(yaw=yaw,
            roll=roll,
            radius=5.0,       # 必要なら調整
            yaw_gain=1.0,
            roll_gain=1.0
            )
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
                level = 0.2
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
           

            vis_assets = [vol_mesh, vol_coord, vol_bbox] + cam 

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
                ctr.set_front ([0, 1.0, 0.0])  # カメラから見た前方向ベクトル
                ctr.set_up    ([0.0, 0.0, 1.0])  # カメラの上向きベクトル
                ctr.set_zoom  (0.8)          # ズーム倍率（1.0 で等倍）

       
                vis.poll_events()
                vis.update_renderer()

                image = vis.capture_screen_float_buffer(do_render=True)
                vis.destroy_window()

                # np.asarray(image) を可視化
                image = np.asarray(image)
                # opencvで画像を表示
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Rendered Image", image)
                cv2.waitKey(0)

            return image
    

if __name__ == "__main__":
    # Example usage
    renderer = R2GaussianSceneRenderer(
        source_path=Path(r"C:\Users\Maemaeko\imari_lab\r2_gaussian\data\synthetic_dataset\cone_ntrain_75_angle_360\1_beetle_cone"),
        model_path=Path(r"C:\Users\Maemaeko\imari_lab\r2_gaussian\output\chest"),
        #model_path="../output/95e359ad-b",
        data_device="cuda"
    )

    csv_path = "../ipad_data/Orientation.csv"
    
    # デバッグ出力
    print(f"CSV path: {csv_path}")
    # print(f"Exists: {csv_path.exists()}")
    # if not csv_path.exists():
    #     print(f"Parent dir contents:")
    #     print(list(csv_path.parent.glob("*")))
    # else:
    yaws_rad, rolls_rad = renderer.load_pose_csv(csv_path)

    yaws_rad, rolls_rad = renderer.load_pose_csv(csv_path)

    # 初期値
    yaw0  = yaws_rad[0]
    roll0 = rolls_rad[0]

    # 差分に変換
    yaws_rad  = (yaws_rad  - yaw0) * 5
    rolls_rad = (rolls_rad - roll0) * 5
    # yaws_rad = np.deg2rad(yaws_deg)
    # rolls_rad = np.deg2rad(rolls_deg)

    print(rolls_rad.max(), rolls_rad.min())
    print(yaws_rad.max(), yaws_rad.min())


    # ===============================================
    #  1 枚ずつレンダリングして保存
    # ===============================================
    save_dir = Path("./render_from_ipad")
    save_dir.mkdir(exist_ok=True)


    # yaw_deg 
    # プラスに動くと下からのぞき込むかたちになる
    #yaws_rad = np.linspace(0, np.pi/2, 100)
    # # マイナスに動くと上からのぞき込むかたちになる
    # yaws_rad = np.linspace(0, -np.pi/4, 5)

    # yaws_rad = np.linspace(0, 0, 50)


    # # # # roll_deg
    #rolls_rad = np.linspace(0, 0, 50)




    # 1フレームずつ処理
    from itertools import islice
    for i, (yaw, roll) in enumerate(islice(zip(yaws_rad, rolls_rad), 500)):
        

        # yaw  = np.clip(yaw,  -np.pi/3, np.pi/3)
        # roll = np.clip(roll, -np.pi/3, np.pi/3)

        #print(f"[render] Frame {i}: yaw={yaw[i]:.2f} rad, roll={roll[i]:.2f} rad")
        rendering = renderer.render_gaussians(yaw, roll)


        # # 保存
        # plt.imsave(save_dir / f"parallax_{i:03d}.png", rendering)


        #renderer.visualize(yaw, roll, rendering) 

        # ---- 保存 ----
        save_path = save_dir / f"frame_{i:05d}.png"
        img = (rendering * 255).astype(np.uint8)

        # min-max normalize (0〜1)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 0〜255 に変換
        img_norm255 = (img_norm * 255).astype(np.uint8)
        

        # グレースケール → 赤着色
        # shape: (H, W) → (H, W, 3)
        # 赤着色ではなくGチャンネルに入れたい？
        img_red = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img_red[..., 0] = 0                 # R
        img_red[..., 1] = img_norm255       # G
        img_red[..., 2] = 0      

        Image.fromarray(img_red).save(save_path)
    print("[done] All frames rendered and saved!")

    from PIL import Image
import glob

# 連番 PNG を読み込み
frames = []
png_files = sorted(glob.glob(str(save_dir / "frame_*.png")))
for filename in png_files:
    frames.append(Image.open(filename))

# GIF 保存
gif_path = save_dir / "parallax.gif"
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=50,  # 1フレーム50ms = 20fps
    loop=0
)

print(f"[done] GIF saved → {gif_path}")


