import os
from os import makedirs
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
import time
import tqdm
import numpy as np
# import dearpygui.dearpygui as dpg
from omegaconf import OmegaConf
import torchvision
from argparse import ArgumentParser, Namespace
import torch
import torch.nn.functional as F
import json
import rembg
from PIL import Image

from dreamgaussian.cam_utils import orbit_camera, OrbitCamera, save_camera
from dreamgaussian.gs_renderer_2D import Renderer, MiniCam
from pytorch3d.renderer import (AmbientLights, MeshRenderer, MeshRasterizer, RasterizationSettings,SoftPhongShader,TexturesVertex)
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer.blending import BlendParams
import open3d as o3d
from extract_object import visualize_gt
from dreamgaussian.cam_utils import convert_camera_from_orbit_to_pytorch3d
from gaussian_splatting.utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix

# from dreamgaussian.grid_put import mipmap_linear_grid_put_2d
# from dreamgaussian.mesh import Mesh, safe_normalize

class GSCamera(torch.nn.Module):
    """Class to store Gaussian Splatting camera parameters.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_height=None, image_width=None,
                 ):
        """
        Args:
            colmap_id (int): ID of the camera in the COLMAP reconstruction.
            R (np.array): Rotation matrix.
            T (np.array): Translation vector.
            FoVx (float): Field of view in the x direction.
            FoVy (float): Field of view in the y direction.
            image (np.array): GT image.
            gt_alpha_mask (_type_): _description_
            image_name (_type_): _description_
            uid (_type_): _description_
            trans (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
            scale (float, optional): _description_. Defaults to 1.0.
            data_device (str, optional): _description_. Defaults to "cuda".
            image_height (_type_, optional): _description_. Defaults to None.
            image_width (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(GSCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is None:
            if image_height is None or image_width is None:
                raise ValueError("Either image or image_height and image_width must be specified")
            else:
                self.image_height = image_height
                self.image_width = image_width
        else:        
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    @property
    def device(self):
        return self.world_view_transform.device
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt._W
        self.H = opt._H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy, fovx = opt.fovx)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        # torch.cuda.set_device(0)
        self.device = torch.device('cuda')
        
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        # self.renderer = Renderer(opt).to(self.device)
        self.gaussain_scale_factor = 1
        

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.render_mesh = self.opt.render_mesh
        
        
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
            
        if self.render_mesh:
            mesh_path = os.path.join(opt.load,"train","ours_"+self.opt.iteration,"fuse_post.ply")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
            verts = torch.FloatTensor(np.array(mesh.vertices))
            pcd = o3d.io.read_point_cloud(mesh_path)
            verts_rgb = torch.FloatTensor(np.array(pcd.colors)).unsqueeze(0)
            faces = torch.FloatTensor(np.array(mesh.triangles))
            textures = TexturesVertex(verts_features=verts_rgb)
            self.mesh = Meshes([verts], [faces], textures=textures).to(self.device)
        else:
            print("no mesh, render from gaussian")
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            print("Loading iteration:",{self.opt.iteration})
            self.renderer.initialize(input = os.path.join(self.opt.load, "point_cloud","iteration_"+self.opt.iteration,"point_cloud.ply"), depth_ratio = self.opt.depth_ratio)            
            self.render_path = os.path.join(self.opt.load, "refined", "random")
            self.sd_path = os.path.join(self.opt.load, "refined", "sd")
            self.mask_path = os.path.join(self.opt.load, "refined", "mask")
            makedirs(self.render_path, exist_ok=True)
            makedirs(self.sd_path, exist_ok=True)
            makedirs(self.mask_path, exist_ok=True)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        # if self.gui:
        #     dpg.create_context()
        #     self.register_dpg()
        #     self.test_step()

    # def __del__(self):
    #     if self.gui:
    #         dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        
        # # default camera
        # pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        # self.fixed_cam = MiniCam(
        #     pose,
        #     self.opt.ref_size,
        #     self.opt.ref_size,
        #     self.cam.fovy,
        #     self.cam.fovx,
        #     self.cam.near,
        #     self.cam.far,
        # )
        self.target = torch.mean(self.renderer.gaussians._xyz, dim=0).detach().cpu().numpy()
        
        variance = torch.var(self.renderer.gaussians._xyz, unbiased=True).detach().cpu().numpy()
        self.radius_c = np.sqrt(variance)
        
        print("target:", self.target)
        print("radius:", self.radius_c)
        self.radius = float(self.radius_c)+self.opt.radius
        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        
        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None
        # self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.prompt != ""

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from dreamgaussian.guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from dreamgaussian.guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from dreamgaussian.guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last = self.input_img_torch[0].permute(1,2,0).contiguous()

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)
        
        
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for iter in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            # ### known view
            # if self.input_img_torch is not None:
            #     cur_cam = self.fixed_cam
            #     out = self.renderer.render(cur_cam)

            #     # rgb loss
            #     image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            #     loss = loss + 10000 * step_ratio * F.mse_loss(image, self.input_img_torch)

            #     # mask loss
            #     mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
            #     loss = loss + 1000 * step_ratio * F.mse_loss(mask, self.input_mask_torch)

            ### novel view (manual batch)
            # render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            render_resolution = 800
            # render_resolution = 256 if step_ratio < 0.3 else (512 if step_ratio < 0.6 else 800)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            # min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            # max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
            min_ver = self.opt.min_ver
            max_ver = self.opt.max_ver
            radius = self.radius
            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, radius, target= self.target)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, radius)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            torchvision.utils.save_image(images, os.path.join(self.render_path, str(self.step) + ".jpg"))
            
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # import kiui
            # print(hor, ver)
            # kiui.vis.plot_image(images)

            # guidance loss
            strength = step_ratio * 0.15 + 0.8
            if self.enable_sd:
                if self.step < self.opt.sdsiter:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images,  step_ratio)
                    # refined_images = self.guidance_sd.refine(images, poses, strength=strength).float()
                    # refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    # loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)
                else:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
                    refined_images = self.guidance_sd.refine(images, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    torchvision.utils.save_image(refined_images, os.path.join(self.sd_path, str(self.step) + ".jpg"))
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)
                    

            if self.enable_zero123:
                # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.opt.elevation).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)
                
                torchvision.utils.save_image(refined_images, os.path.join(self.sd_path, str(self.step) + ".jpg"))
                # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        with torch.no_grad():
            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.05, extent=radius, max_screen_size=10)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        # if self.gui:
        #     dpg.set_value("_log_train_time", f"{t:.4f}ms")
        #     dpg.set_value(
        #         "_log_train_log",
        #         f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
        #     )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    def random_render(self):
        render_W = self.W
        render_H = self.H
        random_render_path = os.path.join(self.opt.load, "refined", "images")
        makedirs(random_render_path, exist_ok=True)
        poses = []
        vers, hors, radii = [], [], []
        
        radius = self.radius
        random_n = self.opt.random_n
        
        radii.append(radius)
        min_ver = self.opt.min_ver
        max_ver = self.opt.max_ver
        print("random render:",random_n)
        for cam_id in range(random_n):
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            vers.append(ver)
            hors.append(hor)
            pose = orbit_camera(ver, hor, radius, target= self.target, opengl=False)
            poses.append(pose)
            
            cur_cam = MiniCam(pose, render_W, render_H, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            torchvision.utils.save_image(image, os.path.join(random_render_path, "random_"+str(cam_id) + ".jpg"))
            
        with open(os.path.join(self.opt.load, "cameras.json"), "r") as f:
            cameras_data = json.load(f)
            if "ramdom" in cameras_data[-1]["img_name"]:
                new_data = cameras_data[:-random_n]
            else:
                new_data = cameras_data
            l = len(new_data)

            
            for cam_id in range(random_n):
                po, ro = save_camera(vers[cam_id], hors[cam_id], radius = radius, target= self.target, opengl=False)
                cam = {"id": cam_id +l,
                    "img_name": "random_"+str(cam_id),
                    "width": cameras_data[0]["width"],
                    "height": new_data[0]["height"],
                    "position": po.tolist(),
                    "rotation": ro.tolist(),
                    "fy": cameras_data[0]["fy"],
                    "fx": cameras_data[0]["fx"]
                }
                new_data.append(cam)

            # Save the updated data back to cameras.json
        with open(os.path.join(self.opt.load, "refined", "cameras.json"), "w") as file:
            json.dump(new_data, file)

        print("New data added to cameras.json successfully.")
    
    def random_render_mask(self):
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer
        
        
        num_classes = 256
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(self.renderer.gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(self.opt.load,"point_cloud","iteration_"+str(self.opt.iteration),"classifier.pth")))

        self.target = torch.mean(self.renderer.gaussians._xyz, dim=0).detach().cpu().numpy()
        
        variance = torch.var(self.renderer.gaussians._xyz, unbiased=True).detach().cpu().numpy()
        self.radius_c = np.sqrt(variance)
        
        print("target:", self.target)
        
        # self.radius = float(self.radius_c)+self.opt.radius
        self.radius = float(self.radius_c)+self.opt.radius
        print("radius:", self.radius)
        
        
        print(f"[INFO] loading SD...")
        from dreamgaussian.guidance.sd_utils import StableDiffusion
        self.guidance_sd = StableDiffusion(self.device, hf_key= self.opt.sd)
        # self.guidance_sd = StableDiffusion(self.device)
        print(f"[INFO] loaded SD!")
        
        # starter = torch.cuda.Event(enable_timing=True)
        # ender = torch.cuda.Event(enable_timing=True)
        # starter.record()
        
        render_W = self.opt.W
        
        render_H = self.opt.H
        print(render_W,render_H)
        
        target_obj = self.opt.select_obj_id
        random_render_mesh_path = os.path.join(self.opt.load, "refined", "render_mesh")
        random_render_gs_path = os.path.join(self.opt.load, "refined", "render_gs")
        random_obj_path = os.path.join(self.opt.load, "refined", "render_obj")
        random_inpainting_path = os.path.join(self.opt.load, "refined", "images")
        makedirs(random_render_mesh_path, exist_ok=True)
        makedirs(random_render_gs_path, exist_ok=True)
        makedirs(random_obj_path, exist_ok=True)
        makedirs(random_inpainting_path, exist_ok=True)
        poses = []
        vers, hors, radii = [], [], []
        pon, ron = [], []
        
        radius = self.radius
        random_n = self.opt.random_n
        # random_n = 20
        
        radii.append(radius)
        min_ver = self.opt.min_ver
        max_ver = self.opt.max_ver
        print("random render:",random_n)
        
        if self.render_mesh:
        
            faces_per_pixel = 1
            max_faces_per_bin = 5000_000

            mesh_raster_settings = RasterizationSettings(
                image_size=(render_H, render_W),
                blur_radius=0.0, 
                faces_per_pixel=faces_per_pixel,
                max_faces_per_bin=max_faces_per_bin,
                max_faces_opengl= 10000,
            )
            lights = AmbientLights(device=self.device)
            
        
        
        for cam_id in range(random_n):
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            vers.append(ver)
            hors.append(hor)
            pose = orbit_camera(ver, hor, radius, target= self.target, opengl=False)
            poses.append(pose)
            po, ro = save_camera(ver, hor, radius = radius, target= self.target, opengl=False)
            pon.append(po)
            ron.append(ro)
            
            
        
        with open(os.path.join(self.opt.load, "cameras.json"), "r") as f:
            cameras_data = json.load(f)
            if "ramdom" in cameras_data[-1]["img_name"]:
                new_data = cameras_data[:-random_n]
            else:
                new_data = cameras_data
            l = len(new_data)

            
            for cam_id in range(random_n):
                # po, ro = save_camera(vers[cam_id], hors[cam_id], radius = radius, target= self.target, opengl=False)
                cam = {"id": cam_id +l,
                    "img_name": "random_"+str(cam_id),
                    "width": cameras_data[0]["width"],
                    "height": new_data[0]["height"],
                    "position": pon[cam_id].tolist(),
                    "rotation": ron[cam_id].tolist(),
                    "fy": cameras_data[0]["fy"],
                    "fx": cameras_data[0]["fx"]
                }
                new_data.append(cam)

            # Save the updated data back to cameras.json
        with open(os.path.join(self.opt.load, "refined", "cameras.json"), "w") as file:
            json.dump(new_data, file)

        print("New data added to cameras.json successfully.")
        
        with open(os.path.join(self.opt.load,  "refined", "cameras.json"), "r") as f:
            unsorted_camera_transforms = json.load(f)
        camera_transforms = sorted(unsorted_camera_transforms.copy(), key = lambda x : x['id'])
        cam_list = []
        for cam_idx in range(len(camera_transforms)):
            camera_transform = camera_transforms[cam_idx]
            
            # Extrinsics
            rot = np.array(camera_transform['rotation'])
            pos = np.array(camera_transform['position'])
            
            W2C = np.zeros((4,4))
            W2C[:3, :3] = rot
            W2C[:3, 3] = pos
            W2C[3,3] = 1
            
            Rt = np.linalg.inv(W2C)
            T = Rt[:3, 3]
            R = Rt[:3, :3].transpose()
            
            # Intrinsics
            width = camera_transform['width']
            height = camera_transform['height']
            fy = camera_transform['fy']
            fx = camera_transform['fx']
            fov_y = focal2fov(fy, height)
            fov_x = focal2fov(fx, width)
            id = camera_transform['id']
            name = camera_transform['img_name']
            gs_camera = GSCamera(
                colmap_id=id, image=None, gt_alpha_mask=None,
                R=R, T=T, FoVx=fov_x, FoVy=fov_y,
                image_name=name, uid=id,
                image_height=height, image_width=width,)
        
            cam_list.append(gs_camera)
            
        if self.render_mesh:
            p3d_cam = convert_camera_from_orbit_to_pytorch3d(cam_list[-random_n:])
        for cam_id in range(random_n): 
            # print(cam_id)
            cur_cam = MiniCam(poses[cam_id], render_W, render_H, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)
            if not self.render_mesh:  
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                torchvision.utils.save_image(image, os.path.join(random_render_gs_path, "random_"+str(cam_id) + ".jpg"))
            else:
                rasterizer = MeshRasterizer(
                    cameras=p3d_cam[cam_id], 
                    raster_settings=mesh_raster_settings,
                )
                renderer = MeshRenderer(
                    rasterizer=rasterizer,
                    shader=SoftPhongShader(
                        device=self.device, 
                        cameras=p3d_cam[cam_id],
                        lights=lights,
                        blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
                        # blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
                    )
                )
                # print(image.shape)
                img  = renderer(self.mesh, cameras=p3d_cam[cam_id])
                # print(img.shape)
                image = img.permute(0, 3, 1, 2)[:,:3,:,:]
                image_gs = out["image"].unsqueeze(0)
                torchvision.utils.save_image(image_gs, os.path.join(random_render_gs_path, "random_"+str(cam_id) + ".jpg"))
                torchvision.utils.save_image(image, os.path.join(random_render_mesh_path, "random_"+str(cam_id) + ".jpg"))
            
            
            
            rendering_obj = out["render_object"]
            logits = classifier(rendering_obj)
            pred_obj = torch.argmax(logits,dim=0)
            gt_rgb_mask = visualize_gt(pred_obj.cpu().numpy().astype(np.uint8), target_obj)
            
            gray_image = cv2.cvtColor(gt_rgb_mask, cv2.COLOR_RGB2GRAY)
            gray_image = (gray_image * 255).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            eroded_image = cv2.erode(gray_image, kernel, iterations=2)

            eroded_image = eroded_image.astype(np.float32) / 255.0
            eroded_image_color = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2RGB)
            
            
            gt_rgb_mask = eroded_image_color
            
            
            
            gt_mask = (gt_rgb_mask).astype(np.uint8)
            # print(save_gt_mask.shape)
            save_gt_mask = Image.fromarray(gt_mask*255)
            save_gt_mask.save(os.path.join(random_obj_path, "random_"+str(cam_id) + ".jpg"))
            
            threshold = self.opt.threshold
            mask = self.guidance_sd.sd_mask(image, prompt = [self.prompt], render_H=render_H, render_W=render_W, th = threshold)
            # me_mask = cv2.medianBlur(mask, 5)
            # smoothed_mask = cv2.GaussianBlur(mask, (5, 5), 0)
            # smoothed_mask = cv2.dilate(mask, kernel, iterations=3)
            opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算去除小点
            closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
            
            mask_t = (np.stack((closed_mask,) * 3, axis=-1)* gt_mask).astype(np.uint8)*255

            
            # mask_t = np.minimum(np.stack((mask*255,) * 3, axis=-1), gt_mask)
            # mask_t = (np.stack((mask,) * 3, axis=-1)* gt_mask).astype(np.uint8)*255

            pil_images = Image.fromarray(mask_t)
            pil_images.save(os.path.join(self.mask_path, "random_"+str(cam_id) + ".jpg"))

            print("Inpainting: ",cam_id+1,"/",random_n)
            prompt = self.prompt

            refined_image = self.guidance_sd.inpaint(prompt, image.to('cpu'), pil_images)
            resized_image = refined_image.resize((render_W, render_H),Image.BILINEAR)
            resized_image.save(os.path.join(random_inpainting_path, "random_"+str(cam_id) + ".jpg"))


    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        # if self.gui:
        #     dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
        #     dpg.set_value(
        #         "_texture", self.buffer_image
        #     )  # buffer must be contiguous, else seg fault!

    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        # file_prompt = file.replace("_rgba.png", "_caption.txt")
        # if os.path.exists(file_prompt):
        #     print(f'[INFO] load prompt from {file_prompt}...')
        #     with open(file_prompt, "r") as f:
        #         self.prompt = f.read().strip()

    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            # self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
            path = os.path.join(self.opt.load,"refined","point_cloud","iteration_"+self.opt.iteration, 'point_cloud.ply')
            self.renderer.gaussians.save_ply(path)
        
        self.random_render()
        
        
        # save
        # self.save_model(mode='model')
        # self.save_model(mode='geo+tex')
        

if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", required=True, default="")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    # gui.train(opt.iters)
    gui.random_render_mask()
