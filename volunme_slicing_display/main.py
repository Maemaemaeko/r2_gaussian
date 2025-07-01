from image2pose import Image2Pose
from pose2visualize import R2GaussianSceneRenderer
from pathlib import Path
import os
 

def main(image_folder):
    image2pose = Image2Pose(Path("C:\\Users\\Maemaeko\\imari_lab\\r2_gaussian\\volunme_slicing_display\\camera_calibration"))
    image2pose.read_intrinsics()

    # 画像フォルダ内のすべての画像ファイルを取得
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {image_folder}")

    # renderingの設定
    renderer = R2GaussianSceneRenderer(
        source_path="../data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
        model_path="../output/95e359ad-b",
        data_device="cuda"
    )
    d = 0.1
    eye_position = "top"
    cut_method = "by_plane"

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        marker_corners, marker_ids = image2pose.detect_markers(image_path)
        rvec, tvec = image2pose.detect_charuco(image_path, marker_corners, marker_ids)
        charuco_tf, _ = image2pose.output_transform(rvec, tvec)
        charuco_center = image2pose.output_charuco_center(rvec, tvec)

        # R2GaussianSceneRendererを使用して表示
        rendering, rendering_cut = renderer.render_gaussians(charuco_tf, charuco_center, d, eye_position, cut_method)
        renderer.visualize(charuco_tf, charuco_center, rendering=rendering_cut, eye_position=eye_position)

if __name__ == "__main__":
    image_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\images"
    main(image_folder)