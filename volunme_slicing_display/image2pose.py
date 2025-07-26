import numpy as np
import cv2
from cv2 import aruco
from pathlib import Path
from PIL import Image

class Image2Pose:
    def __init__(self, calibration_path):
        self.calibration_path = calibration_path
        #self.cameraMatrix, self.distCoeffs = read_intrinsics(self.path)
        self.square_length = 0.08 # 0.04
        self.marker_length = 0.06 # 0.03
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard((7, 5), self.square_length, self.marker_length, self.dictionary)
        self.cameraMatrix = None
        self.distCoeffs = None


    def read_intrinsics(self):
        """Read camera intrinsics from a file."""
        calibration_data = np.load(self.calibration_path / "calibration.npz")
        self.cameraMatrix = calibration_data['camera_matrix']
        print("Camera Matrix:\n", self.cameraMatrix)
        self.distCoeffs = calibration_data['dist_coeffs']
        print("Distortion Coefficients:\n", self.distCoeffs)

    def detect_markers(self , image, rotation_angle=0):
        """Detect markers in the image and return their poses."""
        #image = cv2.imread(str(image_path))
        if rotation_angle != 0:
            # Rotate the image if a rotation angle is specified
            image = cv2.rotate(image, rotation_angle)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")
        
        detector_params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(self.dictionary, detector_params)
        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(image)
        immarkers = aruco.drawDetectedMarkers(image.copy(), marker_corners, marker_ids)

        objPoints = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        for marker_corner in marker_corners:
            valid, rvec, tvec = cv2.solvePnP(
                objPoints, marker_corner, self.cameraMatrix, self.distCoeffs
            )
            cv2.drawFrameAxes(image, self.cameraMatrix, self.distCoeffs, rvec, tvec, 1)

        return marker_corners, marker_ids
    
    def detect_charuco(self, image, marker_corners=None, marker_ids=None):
        """Detect ChArUco markers in the image and return their poses."""
        #image = cv2.imread(str(image_path))
        
        detector_params = aruco.CharucoParameters()
        detector_params.cameraMatrix = self.cameraMatrix


        charuco_detector = aruco.CharucoDetector(self.board, detector_params)
        charuco_corners, charuco_ids = None, None
        charuco_corners, charuco_ids,  charuco_marker_corners, charuco_marker_ids = charuco_detector.detectBoard(image, charuco_corners, charuco_ids, marker_corners, marker_ids)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
            return None, None

        imcharuco = aruco.drawDetectedCornersCharuco(image.copy(), charuco_corners, charuco_ids)

        # charco axis
        objPoints, imgPoints = self.board.matchImagePoints(charuco_corners, charuco_ids)
        valid, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, self.cameraMatrix, self.distCoeffs)
        cv2.drawFrameAxes(imcharuco, self.cameraMatrix, self.distCoeffs, rvec, tvec, self.square_length * 3)

        print("Translation Vector:\n", tvec)
        

        return rvec, tvec
    

    def output_transform(self, rvec, tvec):
        """Convert rotation vector and translation vector to a transformation matrix."""
        if rvec is None or tvec is None:
            return None, None
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        charuco_tf = np.concatenate((np.concatenate((rotation_matrix, tvec), axis=1), np.array([[0, 0, 0, 1]])))

        camera_position = -np.matrix(rotation_matrix).T * np.matrix(tvec)
        camera_tf = np.concatenate((np.concatenate((rotation_matrix.T, camera_position), axis=1), np.array([[0, 0, 0, 1]])))

        return charuco_tf, camera_tf
    
    def output_charuco_center(self, rvec, tvec):
        if rvec is None or tvec is None:
            return None
        # 回転ベクトルから回転行列に変換
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        # ボードの中心
        charuco_center = np.array([[self.square_length * 3 / 2], [self.square_length * 2 / 2], [0]], dtype=np.float32)
        print("Charuco Center in Object Coordinates:\n", charuco_center)
        # ボードの中心をカメラ座標系に変換
        charuco_center = rotation_matrix @ charuco_center + tvec
        print("Charuco Center in Camera Coordinates:\n", charuco_center)

        return charuco_center


    
if __name__ == "__main__":
    import argparse
    image2pose = Image2Pose(Path("C:\\Users\\Maemaeko\\imari_lab\\r2_gaussian\\volunme_slicing_display\\camera_calibration"))
    image_path = Path(r"C:\Users\Maemaeko\imari_lab\r2_gaussian\volunme_slicing_display\images\capture_20250617_175949.png")
    image2pose.read_intrinsics()
    marker_corners, marker_ids = image2pose.detect_markers(image_path, rotation_angle=cv2.ROTATE_180)
    #marker_corners, marker_ids = image2pose.detect_markers(image_path, rotation_angle=cv2.ROTATE_90_CLOCKWISE)
    rvec, tvec = image2pose.detect_charuco(image_path, marker_corners, marker_ids)
    charuco_tf, _ = image2pose.output_transform(rvec, tvec)
    charuco_center = image2pose.output_charuco_center(rvec, tvec)
    print("ChArUco Transformation Matrix:\n", charuco_tf)
    # save the transformation matrices to a file
    np.savez("charuco_camera_transformation.npz", charuco_tf=charuco_tf, charuco_center=charuco_center)
