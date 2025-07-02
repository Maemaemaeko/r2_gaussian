from image2pose import Image2Pose
from pose2visualize import R2GaussianSceneRenderer
from pathlib import Path
import os
import numpy as np
from PIL import Image
import cv2
 

def main(image_folder, rendering_folder=None, visualization_folder=None, concat_folder=None, create_gif=False):
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


    # 保存先のフォルダを作成

    if rendering_folder and not os.path.exists(rendering_folder):
        os.makedirs(rendering_folder)
        print(f"Created rendering folder: {rendering_folder}")

    if visualization_folder and not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)
        print(f"Created visualization folder: {visualization_folder}")
    
    if concat_folder and not os.path.exists(concat_folder):
        os.makedirs(concat_folder)
        print(f"Created concat folder: {concat_folder}")

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        marker_corners, marker_ids = image2pose.detect_markers(image_path)


        rvec, tvec = image2pose.detect_charuco(image_path, marker_corners, marker_ids)
        charuco_tf, _ = image2pose.output_transform(rvec, tvec)
        charuco_center = image2pose.output_charuco_center(rvec, tvec)

        # R2GaussianSceneRendererを使用してレンダリング
        _, rendering_cut = renderer.render_gaussians(charuco_tf, charuco_center, d, eye_position, cut_method)
        if rendering_cut is not None:
            rendering_cut /= np.max(rendering_cut)  # 正規化
            save_numpy_image(rendering_cut, os.path.join(rendering_folder, f"rendering_cut_{image_file}"))
            print(f"Saved rendering cut image to: {os.path.join(rendering_folder, f'rendering_cut_{image_file}')}")

        # 位置関係を可視化
        image = renderer.visualize(charuco_tf, charuco_center, rendering=rendering_cut, eye_position=eye_position, interactive=False)
        if image is not None:
            # 画像を保存
            output_image_path = os.path.join(visualization_folder, f"rendered_{image_file}")
            save_numpy_image(image, output_image_path)
            print(f"Saved rendered image to: {output_image_path}")
        else:
            print(f"Failed to render image: {image_file}")

        # concat
        print(image.shape, rendering_cut.shape)
        if rendering_cut.ndim == 2:
            rendering_cut = cv2.cvtColor(rendering_cut, cv2.COLOR_GRAY2BGR)
            print(rendering_cut.shape)

        concat_images = np.concatenate([image, rendering_cut], axis=1)
        if concat_images is not None:
            concat_image_path = os.path.join(concat_folder, f"concat_{image_file}")
            save_numpy_image(concat_images, concat_image_path)
            print(f"Saved concatenated image to: {concat_image_path}")
    


    # GIF作成
    if create_gif and rendering_folder:
        create_gif_from_images(image_folder, os.path.join(image_folder, "output.gif"))
        create_gif_from_images(rendering_folder, os.path.join(rendering_folder, "output.gif"))
        create_gif_from_images(visualization_folder, os.path.join(visualization_folder, "output.gif"))
        create_gif_from_images(concat_folder, os.path.join(concat_folder, "output.gif"))



def save_numpy_image(image, output_path):
    """Save a numpy image to a file."""
    # Convert the float buffer to an 8-bit image
    #image = (image / np.max(image) * 255).astype(np.uint8)
    image = (image * 255).astype(np.uint8)  # Assuming image is normalized between 0 and 1
    img = Image.fromarray(image)
    img.save(output_path)
    print(f"Image saved to {output_path}")

def create_gif_from_images(image_folder, output_path):
    """Create a GIF from images in the specified folder."""
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(image_folder, filename)
            images.append(Image.open(img_path))
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=500, loop=0)
        print(f"GIF saved to {output_path}")
    else:
        print("No images found to create GIF.")

def create_mov_from_images(image_folder, output_path):
    """Create a MOV file from images in the specified folder."""
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    if images:
        height, width, layers = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        for img in images:
            out.write(img)
        out.release()
        print(f"MOV saved to {output_path}")
    else:
        print("No images found to create MOV.")


if __name__ == "__main__":
    image_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\test_images"
    rendering_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\rendering"
    visualization_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\visualization"
    concat_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\concat_images"

    # main(image_folder, rendering_folder=rendering_folder,
    #      visualization_folder=visualization_folder,
    #      concat_folder=concat_folder,
    #      create_gif=True)
    create_gif_from_images(concat_folder, os.path.join(concat_folder, "output.gif"))