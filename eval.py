import os
import os.path as osp
import sys
import torch
from tqdm import tqdm, trange
import torchvision
from time import time
import numpy as np
import concurrent.futures
import yaml
from argparse import ArgumentParser
from random import randint
import SimpleITK as sitk

sys.path.append("./")
from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
)
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.image_utils import metric_vol, metric_proj

from types import SimpleNamespace



def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
    skip_recon: bool,
):  
    # Set up dataset
    scene = Scene(
        dataset,
        shuffle=False,
    )

    # Set up Gaussians
    gaussians = GaussianModel(None)  # scale_bound will be loaded later
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    scene.gaussians = gaussians


    scene_gt_path = "/home/maemaeko/imari_lab/r2_gaussian/data/synthetic_dataset/cone_ntrain_360_angle_360/1_pepper_cone"


    # ModelParams を使わず、必要な属性だけ持った args を作る
    args_gt = SimpleNamespace(
        source_path=scene_gt_path,
        model_path="/home/maemaeko/imari_lab/r2_gaussian/output/pepper_gt_debug",
        data_device="cuda",
        ply_path="",
        scale_min=0.0005,
        scale_max=0.5,
        eval=False,
    )

    scene_gt = Scene(args_gt)


    save_path = osp.join(
        dataset.model_path,
        "eval_360",
        "iter_{}".format(loaded_iter),
    )

    # Evaluate projection train
    
    evaluate_render(
        save_path,
        "render_train",
        scene.getTrainCameras(),
        gaussians,
        pipeline,
    )

    
    evaluate_render(
        save_path,
        "render_all",
        scene_gt.getTrainCameras(),
        gaussians,
        pipeline,
    )




def evaluate_render(save_path, name, views, gaussians, pipeline):
    """Evaluate projection rendering."""
    proj_save_path = osp.join(save_path, name)

    # If already rendered, skip.
    if osp.exists(osp.join(save_path, "eval.yml")):
        print("{} in {} already rendered. Skip.".format(name, save_path))
        return
    os.makedirs(proj_save_path, exist_ok=True)

    gt_list = []
    render_list = []
    for view in tqdm(views, desc="render {}".format(name), leave=False):
        rendering = render(view, gaussians, pipeline)["render"]
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)
        render_list.append(rendering)
    multithread_write(gt_list, proj_save_path, "_gt")
    multithread_write(render_list, proj_save_path, "_pred")

    images = torch.concat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)
    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
    eval_dict = {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
    }
    with open(osp.join(save_path, f"eval2d_{name}.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
    print(
        f"{name} complete. psnr_2d: {eval_dict['psnr_2d']}, ssim_2d: {eval_dict['ssim_2d']}."
    )


def multithread_write(image_list, path, suffix):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, osp.join(path, "{0:05d}".format(count) + "{}.png".format(suffix))
            )
            np.save(
                osp.join(path, "{0:05d}".format(count) + "{}.npy".format(suffix)),
                image.cpu().numpy()[0],
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)
    parser.add_argument("--skip_recon", action="store_true", default=False)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
            args.skip_recon,
        )
