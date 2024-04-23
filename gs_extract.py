# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco
import os
from os import makedirs
import torch
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import colorsys
import cv2
import torchvision
from argparse import ArgumentParser
from gaussian_splatting.scene import Scene
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
from ext.grounded_sam import grouned_sam_output, load_model_hf
from segment_anything import sam_model_registry, SamPredictor
from render import feature_to_rgb, visualize_obj

def select_id(classification_maps, mask, ioa_thresh=0.7):
    unique_classes = classification_maps.unique()
    classes_above_threshold = []

    for class_id in unique_classes:
        class_mask = (classification_maps == class_id).byte()
        class_area = class_mask.sum()
        intersection = torch.sum(class_mask * mask)
        ioa = intersection / class_area if class_area != 0 else 0  # Avoid division by zero
        
        if ioa > ioa_thresh:
            classes_above_threshold.append(class_id)

    return classes_above_threshold

def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask


def visualize_gt(objects,target_obj):
    obj_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        if id in target_obj:
            obj_mask[objects == id] = 1
        else:
            obj_mask[objects == id] = 0
    return obj_mask

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, target_obj):
    render_path = os.path.join(model_path, "object_{}".format(target_obj[0]), "renders")
    gts_path = os.path.join(model_path, "object_{}".format(target_obj[0]),  "images")
    # colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    # gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path,"object_{}".format(target_obj[0]), "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # makedirs(colormask_path, exist_ok=True)
    # makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        

        gt_rgb_mask = visualize_gt(pred_obj.cpu().numpy().astype(np.uint8), target_obj)
        gt_rgb_mask = torch.from_numpy(gt_rgb_mask)
        gt_mask = torch.permute(gt_rgb_mask,(2,0,1)).to('cuda')
        # rgb_mask = feature_to_rgb(rendering_obj)
        # Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        # Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, view.image_name + ".jpg"))
        gt = view.original_image[0:3, :, :]
        
        obj_gt = gt * gt_mask
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".jpg"))
        torchvision.utils.save_image(obj_gt, os.path.join(gts_path, view.image_name + ".jpg"))

    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    size = (gt.shape[-1]*5,gt.shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4v'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path,file_name)))
        rgb = np.array(Image.open(os.path.join(render_path,file_name)))
        # gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
        # render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

        result = np.hstack([gt,rgb,pred_obj])
        result = result.astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path,file_name))
        writer.write(result[:,:,::-1])

    writer.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, target_obj):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, target_obj)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, target_obj)

def choose_id(dataset : ModelParams, iteration : int,  opt : OptimizationParams, prompt, threshold=0.5, choose_view = 0):
    id = []
    
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # grounding-dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        # sam-hq
        sam_checkpoint = 'Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)
        positives = prompt.split(";")
        print("Text prompts:    ", positives)
        
        for TEXT_PROMPT in positives:
            views = scene.getTrainCameras()
            # Use Grounded-SAM on the first frame
            results0 = render(views[choose_view], gaussians, pipeline, background)
            rendering0 = results0["render"]
            rendering_obj0 = results0["render_object"]
            logits = classifier(rendering_obj0)
            pred_obj = torch.argmax(logits,dim=0)

            image = (rendering0.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
            text_mask, annotated_frame_with_mask = grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image)
            selected_obj_ids = select_id(pred_obj, text_mask, threshold)
            if selected_obj_ids not in id:
                print("Found object ID:", selected_obj_ids)
                id.append(selected_obj_ids)
    
    return id

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

    args.num_classes = config.get("num_classes")
    args.removal_thresh = config.get("removal_thresh")
    args.select_obj_id = config.get("select_obj_id")
    args.prompt = config.get("prompt")
    args.grounding_thresh = config.get("grounding_thresh")

    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.select_obj_id:
        extract(model.extract(args), args.iteration, opt.extract(args), args.select_obj_id, args.removal_thresh)
    else:
        id = choose_id(model.extract(args), args.iteration, opt.extract(args), args.prompt, args.grounding_thresh)
        extract(model.extract(args), args.iteration, opt.extract(args), id, args.removal_thresh)
        
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.select_obj_id)