import time
from image2pose import Image2Pose
from pose2visualize import R2GaussianSceneRenderer
from pose2visualize_gs import GaussianSceneRenderer
from pathlib import Path
import os
import numpy as np
from PIL import Image
import cv2
 



def main(image_folder=None, rendering_folder=None, visualization_folder=None, concat_folder=None, create_gif=False):
    image2pose = Image2Pose(Path("C:\\Users\\Maemaeko\\imari_lab\\r2_gaussian\\volunme_slicing_display\\camera_calibration"))
    image2pose.read_intrinsics()

    # renderingの設定
    renderer = R2GaussianSceneRenderer(
        source_path="../data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
        model_path="../output/95e359ad-b",
        data_device="cuda"
    )
    renderer_gs = GaussianSceneRenderer(
        source_path="../gaussian-splatting/data/bigfoot",
        model_path="../gaussian-splatting/output/6bd50db1-7",
        ply_path="../gaussian-splatting/output/6bd50db1-7/point_cloud/iteration_30000/point_cloud_v5.ply"
    )

    d = 0.1
    height = 4
    eye_position = "top"
    cut_method = "beyond_plane"
    first = True


    # 保存先のフォルダを作成
    if image_folder and not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Created image folder: {image_folder}")

    if rendering_folder and not os.path.exists(rendering_folder):
        os.makedirs(rendering_folder)
        print(f"Created rendering folder: {rendering_folder}")

    if visualization_folder and not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)
        print(f"Created visualization folder: {visualization_folder}")
    
    if concat_folder and not os.path.exists(concat_folder):
        os.makedirs(concat_folder)
        print(f"Created concat folder: {concat_folder}")

        # webcamから画像を取得
    cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラ

    if not cap.isOpened():
        print(f"Cannot open camera {0}")
        return 
    
    while True:
        ret, image = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # 画像を保存(debug用)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 画像を保存するパスを指定
        output_image_path = os.path.join(image_folder, f"captured_frame_{timestamp}.png")
      

        marker_corners, marker_ids = image2pose.detect_markers(image)
        rvec, tvec = image2pose.detect_charuco(image, marker_corners, marker_ids)
        charuco_tf, _ = image2pose.output_transform(rvec, tvec)
        charuco_center = image2pose.output_charuco_center(rvec, tvec)

        # R2GaussianSceneRendererを使用してレンダリング
        if charuco_tf is None or charuco_center is None:
            print("Failed to compute charuco transformation or center.")
            rendering_cut = np.zeros((480, 640, 3), dtype=np.uint8)  # 空の画像を生成
        else:
            # レンダリングを実行 
            # NOTE: eye_positionはTop固定のほうがい？？
            _, rendering_cut = renderer.render_gaussians(charuco_tf, charuco_center, d, eye_position, cut_method)
            rendering_cut /= np.max(rendering_cut)  # 正規化
            # (480, 640, 3)の形状に変換
            rendering_cut = cv2.resize(rendering_cut, (640, 480))  # サイズを調整
            # 2次元から3次元に変換
            if rendering_cut.ndim == 2:
                rendering_cut = cv2.cvtColor(rendering_cut, cv2.COLOR_GRAY2BGR)

            
            rendering_gs = renderer_gs.render_gaussians(charuco_tf, charuco_center, height, eye_position)
            if rendering_gs is not None:
                rendering_gs = rendering_gs.permute(1, 2, 0).cpu().numpy()  # PyTorch tensorからNumPy配列に変換

        
    
        output_image = rendering_cut if tvec[2] < 1 else rendering_gs
        window_name = "RealTime View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if first:
            first = False
            h,w = output_image.shape[:2]
            # ウィンドウのサイズを調整
            cv2.resizeWindow(window_name, w, h)
        
        
        cv2.imshow(window_name, output_image)
        # 画像を保存
    
        print(f"Saved captured frame to: {output_image_path}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    cap.release()

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
    image_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\capture_images"
    rendering_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\rendering"
    visualization_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\visualization"
    concat_folder = r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\concat_images"

    main(image_folder=image_folder, rendering_folder=rendering_folder,
         visualization_folder=visualization_folder,
         concat_folder=concat_folder,
         create_gif=False)
    #create_gif_from_images(concat_folder, os.path.join(concat_folder, "output.gif"))