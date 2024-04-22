import os
from os import makedirs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from dreamgaussian.cam_utils import orbit_camera, OrbitCamera, save_camera
from dreamgaussian.gs_renderer import Renderer, MiniCam

# from dreamgaussian.grid_put import mipmap_linear_grid_put_2d
# from dreamgaussian.mesh import Mesh, safe_normalize

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt._W
        self.H = opt._H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt._fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
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
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(os.path.join(self.opt.load,"point_cloud","iteration_7000","point_cloud.ply"))            
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
        pose = orbit_camera(self.opt.elevation, 0, self.radius_c)
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
        self.render_path = os.path.join(self.opt.load, "refined_sdssde", "random")
        self.sd_path = os.path.join(self.opt.load, "refined_sdssde", "sd")
        makedirs(self.render_path, exist_ok=True)
        makedirs(self.sd_path, exist_ok=True)
        
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

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
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
        render_W = self.opt.W
        render_H = self.opt.H
        random_render_path = os.path.join(self.opt.load, "images")
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
            
            cur_cam = MiniCam(pose, render_W, render_H, self.opt.fovy, self.opt.fovx, self.cam.near, self.cam.far)
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
        with open(os.path.join(self.opt.load, "refined_sdssde", "cameras.json"), "w") as file:
            json.dump(new_data, file)

        print("New data added to cameras.json successfully.")
            

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
            path = os.path.join(self.opt.load,"refined_sdssde","point_cloud","iteration_7000", 'point_cloud.ply')
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

    gui.train(opt.iters)
