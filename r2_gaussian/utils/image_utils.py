#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
import numpy as np
import torch

sys.path.append("./")
from r2_gaussian.utils.loss_utils import ssim


def mse(img1, img2, mask=None):
    """MSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse


def rmse(img1, img2, mask=None):
    """RMSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    rmse = mse_out**0.5
    return rmse


@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out


@torch.no_grad()
def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
        return psnr_out.item(), None
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None])
                    count += 1
                else:
                    result = 0
                results.append(result)
            results = torch.tensor(results)
            mean_results = torch.sum(results) / count
            ssims.append(mean_results.item())
        return float(np.mean(ssims)), ssims


@torch.no_grad()
def metric_proj(img1, img2, metric="psnr", axis=2, pixel_max=1.0):
    """Metrics for projection

    Args:
        img1 (_type_): [x, y, z]
        img2 (_type_): [x, y, z]
        pixel_max (float, optional): _description_. Defaults to 1.0.
    """
    assert axis in [0, 1, 2, None]
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    n_slice = img1.shape[axis]

    results = []
    count = 0
    for i in range(n_slice):
        if axis == 0:
            slice1 = img1[i, :, :]
            slice2 = img2[i, :, :]
        elif axis == 1:
            slice1 = img1[:, i, :]
            slice2 = img2[:, i, :]
        elif axis == 2:
            slice1 = img1[:, :, i]
            slice2 = img2[:, :, i]
        else:
            raise NotImplementedError
        if slice1.max() > 0:
            slice1 = slice1 / slice1.max()
            slice2 = slice2 / slice2.max()
            if metric == "psnr":
                result = psnr(
                    slice1[None, None], slice2[None, None], pixel_max=pixel_max
                )
            elif metric == "ssim":
                result = ssim(slice1[None, None], slice2[None, None])
            else:
                raise NotImplementedError
            count += 1
        else:
            result = 0
        results.append(result)
    results = torch.tensor(results)
    mean_results = torch.sum(results) / count
    return mean_results.item(), results.tolist()


def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
        return psnr_out.item(), None
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None])
                    count += 1
                else:
                    result = 0
                results.append(result)
            results = torch.tensor(results)
            mean_results = torch.sum(results) / count
            ssims.append(mean_results.item())
        return float(np.mean(ssims)), ssims


import numpy as np
import torch

# ssim はあなたの環境のやつを使う前提
# from some_lib import ssim

def _to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy())
    return x

def _get_slice(vol, axis, i):
    if axis == 0:
        return vol[i, :, :]
    elif axis == 1:
        return vol[:, i, :]
    elif axis == 2:
        return vol[:, :, i]
    else:
        raise ValueError(f"axis must be 0/1/2, got {axis}")

def metric_vol_per_slice(img1, img2, pixel_max=1.0, skip_zero_gt=True):
    """
    img1: GT volume, shape (D,H,W) or similar 3D
    img2: Pred volume, same shape
    Returns:
      summary: dict (mean psnr/ssim etc.)
      per_slice: dict with keys axis=0/1/2, each contains:
        - psnr: list[float] length n_slice (nan if skipped)
        - ssim: list[float] length n_slice (nan if skipped)
        - valid_mask: list[bool]
    """
    img1 = _to_torch(img1).float()
    img2 = _to_torch(img2).float()

    if pixel_max is None:
        pixel_max = float(img1.max().item())

    per_slice = {}
    all_psnr_vals = []
    all_ssim_vals = []

    for axis in [0, 1, 2]:
        n_slice = img1.shape[axis]
        psnr_list = []
        ssim_list = []
        valid_mask = []

        for i in range(n_slice):
            slice1 = _get_slice(img1, axis, i)
            slice2 = _get_slice(img2, axis, i)

            # valid判定（GTが真っ黒スライスは除外したい、という元コードの流儀）
            valid = True
            if skip_zero_gt and slice1.max() <= 0:
                valid = False

            if valid:
                mse = torch.mean((slice1 - slice2) ** 2)
                # mse=0 だと inf になるので安全策（必要なら）
                if mse.item() == 0:
                    psnr = float("inf")
                else:
                    psnr = (10.0 * torch.log10((pixel_max ** 2) / mse)).item()

                s = ssim(slice1[None, None], slice2[None, None])
                # ssimがtensorならfloatへ
                s = float(s.item()) if hasattr(s, "item") else float(s)

                all_psnr_vals.append(psnr)
                all_ssim_vals.append(s)
            else:
                psnr = float("nan")
                s = float("nan")

            psnr_list.append(psnr)
            ssim_list.append(s)
            valid_mask.append(valid)

        per_slice[axis] = {
            "psnr": psnr_list,
            "ssim": ssim_list,
            "valid_mask": valid_mask,
        }

    # 集計（nan除外）
    def _nanmean(xs):
        xs = np.array(xs, dtype=np.float64)
        return float(np.nanmean(xs)) if np.any(~np.isnan(xs)) else float("nan")

    summary = {
        "mean_psnr_over_all_valid_slices": _nanmean(all_psnr_vals),
        "mean_ssim_over_all_valid_slices": _nanmean(all_ssim_vals),
        "mean_psnr_by_axis": {a: _nanmean(per_slice[a]["psnr"]) for a in [0, 1, 2]},
        "mean_ssim_by_axis": {a: _nanmean(per_slice[a]["ssim"]) for a in [0, 1, 2]},
    }

    return summary, per_slice
