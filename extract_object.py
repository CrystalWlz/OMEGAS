# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from gaussian_splatting.scene import Scene
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_splatting.scene.gaussian_model import GaussianModel

import json

from edit_object_removal import points_inside_convex_hull


def extract(dataset : ModelParams, iteration : int,  opt : OptimizationParams, obj_id : int, removal_thresh : float):
    # 1. load gaussian checkpoint
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(os.path.join(dataset.model_path,"point_cloud","iteration_" + str(iteration),"point_cloud.ply"))
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(iteration),"classifier.pth")))

        obj_id = torch.tensor(obj_id).cuda()
    
        logits3d = classifier(gaussians._objects.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[obj_id, :, :] > removal_thresh
        mask3d = mask.any(dim=0).squeeze()

        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d,mask3d_convex)
        mask3d = mask3d.float()[:,None,None]
        
    gaussians.extract_setup(opt, mask3d)
    point_cloud_path = os.path.join(dataset.model_path, "object_{}/point_cloud/iteration_{}".format(int(obj_id[0]),iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    print("Done")
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=7000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="configs/object_removal/bear.json", help="Path to the configuration file")


    args = get_combined_args(parser)
    print("Extracting " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])

    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    extract(model.extract(args), args.iteration, opt.extract(args), args.select_obj_id, args.removal_thresh)