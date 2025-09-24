import os
import numpy as np
import imageio.v2 as imageio  # or from PIL import Image

def bin_pixels(image: np.ndarray, binning: int) -> np.ndarray:
    """Bin pixels by factor (average pooling)."""
    if binning == 1:
        return image
    # reshape and mean over blocks
    sh = (image.shape[0] // binning, binning,
          image.shape[1] // binning, binning)
    return image.reshape(sh).mean(-1).mean(1)


def read_ct_scan_parameters(parameter_file: str) -> dict:
    """Read CT scan parameter file (key=value per line)."""
    parameters = {}
    with open(parameter_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" in line:
                key, value = [s.strip() for s in line.split("=", 1)]
                key = key.lower()  # 小文字化
                parameters[key] = value

    # 型変換（あれば）
    if "numberimages" in parameters:
        parameters["numberImages"] = int(parameters["numberimages"])
    if "pixelsize" in parameters:
        parameters["pixelSize"] = float(parameters["pixelsize"])
    if "geometricmagnification" in parameters:
        parameters["geometricMagnification"] = float(parameters["geometricmagnification"])
    if "detectortype" in parameters:
        parameters["detectorType"] = parameters["detectortype"]

    return parameters



def create_ct_project(project_name: str,
                      recon_mode: str,
                      free_ray=(1, 128, 1, 128),
                      cor_fix=0,
                      binning=1,
                      save_to_disk=False,
                      output_filename="") -> dict:
    """
    Python reimplementation of MATLAB create_ct_project.
    project_name: path prefix of files (e.g., 'FIPS_raw/pine/20201118_pine_cone_')
    recon_mode: '2D' or '3D'
    """
    print(f"Creating CT project '{project_name}'")

    # Load parameter file
    param_file = project_name + ".txt"
    parameters = read_ct_scan_parameters(param_file)
    print(f"Loaded parameters from {parameters}")

    detector_type = parameters.get("detectorType", "").upper()
    if detector_type not in ("EID", "PCD"):
        raise ValueError(f"Unknown detector type {detector_type}")

    if detector_type == "EID":
        print("Found CT scan collected with an energy-integrating X-ray detector.")
    else:
        print("Found CT scan collected with a photon-counting X-ray detector.")

    # Determine size from first projection
    #first_img_file = f"{project_name}0001.tif"
    first_img_file = "/home/maemaeko/imari_lab/r2_gaussian/data_generator/real_dataset/WebCT_raw/dragon/projections-0000.tiff"
    I0 = imageio.imread(first_img_file).astype(np.float64)
    rows, cols = I0.shape

    parameters["detectorRows"] = rows
    parameters["detectorCols"] = cols
    parameters["freeRayAtDetector"] = free_ray
    parameters["binningPost"] = binning
    parameters["pixelSizePost"] = parameters["pixelSize"] * binning
    if "geometricMagnification" in parameters:
        parameters["effectivePixelSizePost"] = (
            parameters["pixelSizePost"] / parameters["geometricMagnification"]
        )

    if recon_mode.upper() == "2D":
        parameters["numDetectorsPost"] = cols // binning
        sinogram = np.zeros((parameters["numberImages"],
                             parameters["numDetectorsPost"]))
        center_row = (rows // binning) // 2
    elif recon_mode.upper() == "3D":
        parameters["projectionRows"] = rows // binning
        parameters["projectionCols"] = cols // binning
        sinogram = np.zeros((parameters["projectionCols"],
                             parameters["numberImages"],
                             parameters["projectionRows"]))
    else:
        raise ValueError("recon_mode must be '2D' or '3D'")

    # Default logTransformed
    log_transformed = parameters.get("logTransformed", "NO").upper()

    print(f"Creating {recon_mode} sinogram...")
    for iii in range(0, parameters["numberImages"]):
        print(f"Processing image {iii}/{parameters['numberImages']} ...", end=" ")

        #filename = f"{project_name}{iii:04d}.tif"
        filename = f"/home/maemaeko/imari_lab/r2_gaussian/data_generator/real_dataset/WebCT_raw/dragon/projections-{iii:04d}.tiff"  # --- EDITED LINE ---
        I = imageio.imread(filename).astype(np.float64)

        # Extract background if not log transformed
        if log_transformed != "YES":
            r1, r2, c1, c2 = free_ray
            background = I[r1-1:r2, c1-1:c2]  # MATLAB 1-index → Python 0-index
        else:
            background = None

        # Apply center of rotation correction
        if cor_fix != 0:
            I = np.roll(I, shift=cor_fix, axis=1)

        # Apply binning
        if binning != 1:
            I = bin_pixels(I, binning)
            if background is not None:
                background = bin_pixels(background, binning)

        # Log transform if needed
        if log_transformed != "YES":
            bkgd_intensity = np.mean(background)
            I = -np.log(I / (bkgd_intensity + 1e-12))

        # Insert into sinogram
        if recon_mode.upper() == "2D":
            detector_data = I[center_row, :]
            sinogram[iii-1, :] = detector_data
        else:  # 3D
            I = I.T
            sinogram[:, iii-1, :] = I

        print("done.")

    print("Sinogram completed.")

    CtData = {
        "type": recon_mode,
        "sinogram": sinogram,
        "parameters": parameters
    }

    if save_to_disk:
        if not output_filename:
            output_filename = project_name + "_ctdata.npz"
        np.savez_compressed(output_filename, **CtData)
        print(f"Saved CT project to {output_filename}")

    print("CT project creation completed.")
    return CtData


