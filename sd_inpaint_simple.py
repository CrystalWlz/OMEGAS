import sys
import cv2
import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from pathlib import Path
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)

def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

# sampler = initialize_model('configs/v2-inpainting-inference.yaml', 'spin_nerf/stablediffusion/checkpoints/512-inpainting-ema.ckpt')
def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [Image.fromarray(img.astype(np.uint8)) for img in result]



if __name__=='__main__':
    # 定义文件夹路径
    image_folder = "sd_test_images/image"
    label_folder = "sd_test_images/label"
    save_folder = "sd_test_images/output"
    prompt = "grass"
    ddim_steps = 45
    num_samples = 1
    scale = 30
    seed = 5032
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 对文件夹里的文件名进行排序，确保顺序一致
    image_files = sorted([f for f in os.listdir(image_folder) if not f.startswith('.')])
    label_files = sorted([f for f in os.listdir(label_folder) if not f.startswith('.')])

    # 检查图像和标签数量是否一致
    assert len(image_files) == len(label_files), "Number of images and labels must match"

    # 遍历图像和标签
    for image_file, label_file in zip(image_files, label_files):
        print("reding")
        # 读取图像和标签
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)
        image = Image.open(image_path)
        label = Image.open(label_path)

        # 检查读取是否成功
        if image is None or label is None:
            print(f"Error reading image or label for {image_file}")
            continue
        init_image = pad_image(image.convert("RGB"))
        init_mask = pad_image(label.convert("RGB"))
        width, height = init_image.size
        print("Inpainting...", width, height)
        sampler = initialize_model('configs/v2-inpainting-inference.yaml', 'spin_nerf/stablediffusion/checkpoints/512-inpainting-ema.ckpt')
        # 将遮罩后的图像传递给predict函数
        processed_image = inpaint(sampler, init_image, init_mask,prompt, ddim_steps, num_samples, scale, seed)
        
        # 保存预测结果
        print("saving")
        output_path = os.path.join(save_folder, image_file)
        cv2.imwrite(output_path, processed_image)
        torch.cuda.empty_cache()

