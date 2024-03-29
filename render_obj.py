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
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from tqdm import tqdm
from os import makedirs
from gaussian_splatting.gaussian_renderer import render
import torchvision
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import json

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def visualize_gt(objects,target_obj):
    obj_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        if id in target_obj:
            obj_mask[objects == id] = 1
        else:
            obj_mask[objects == id] = 0
    return obj_mask

# def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier,target_obj):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "images")
#     # colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
#     # gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
#     pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
#     # makedirs(colormask_path, exist_ok=True)
#     # makedirs(gt_colormask_path, exist_ok=True)
#     makedirs(pred_obj_path, exist_ok=True)

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         results = render(view, gaussians, pipeline, background)
#         rendering = results["render"]
#         rendering_obj = results["render_object"]
        
#         logits = classifier(rendering_obj)
#         pred_obj = torch.argmax(logits,dim=0)
#         pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        

#         gt_objects = view.objects
#         gt_rgb_mask = visualize_gt(gt_objects.cpu().numpy().astype(np.uint8), target_obj)
#         gt_rgb_mask = torch.from_numpy(gt_rgb_mask)
#         gt_mask = torch.permute(gt_rgb_mask,(2,0,1)).to('cuda')
#         # rgb_mask = feature_to_rgb(rendering_obj)
#         # Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
#         # Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
#         Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, view.image_name + ".jpg"))
#         gt = view.original_image[0:3, :, :]
        
#         obj_gt = gt * gt_mask
#         torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".jpg"))
#         torchvision.utils.save_image(obj_gt, os.path.join(gts_path, view.image_name + ".jpg"))

#     out_path = os.path.join(render_path[:-8],'concat')
#     makedirs(out_path,exist_ok=True)
#     fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
#     size = (gt.shape[-1]*5,gt.shape[-2])
#     fps = float(5) if 'train' in out_path else float(1)
#     writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4v'), fourcc, fps, size)

#     for file_name in sorted(os.listdir(gts_path)):
#         gt = np.array(Image.open(os.path.join(gts_path,file_name)))
#         rgb = np.array(Image.open(os.path.join(render_path,file_name)))
#         # gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
#         # render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
#         pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

#         result = np.hstack([gt,rgb,pred_obj])
#         result = result.astype('uint8')

#         Image.fromarray(result).save(os.path.join(out_path,file_name))
#         writer.write(result[:,:,::-1])

#     writer.release()
    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier,target_obj):
    render_path = os.path.join(model_path, "renders")
    gts_path = os.path.join(model_path, "images")
    # colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    # gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path,"objects_pred")
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
        

        gt_objects = view.objects
        gt_rgb_mask = visualize_gt(gt_objects.cpu().numpy().astype(np.uint8), target_obj)
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config_file", type=str, default="configs/gaussian_dataset/truck.json", help="Path to the configuration file")
    args = get_combined_args(parser)
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)
    args.select_obj_id = config.get("select_obj_id", [102])
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.select_obj_id)