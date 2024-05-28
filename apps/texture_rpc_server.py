import zerorpc
from PIL import Image
import numpy as np
import cv2
from compel import Compel
import argparse
import logging 
import math
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import json
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,DDPMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler
)


class ModelZoo():

    def __init__(self, args, device):


        self.current_model = 'empty'
        self.target_model = 'empty'

        self.device = device
        self.weight_dtype = torch.float32

        self.size = 1024
        self.tokenizer = None
        self.text_encoder_cls = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.controlnet = None
        self.pipeline = None
        self.compel_proc = None

        self.modelzoos = []

        self.uv_texture_mask = np.asarray(Image.open("./data/masks/uv_texture_mask_dilate.png").convert("RGB").resize((self.size, self.size))).astype(float) /255.


    def load_diffusion_model(self, pretrained_model, pretrained_control):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        def load_text_encoder(pretrained_path):
            text_encoder_config = PretrainedConfig.from_pretrained(
                pretrained_path,
                subfolder="text_encoder",
                revision=None,
            )
            model_class = text_encoder_config.architectures[0]

            if model_class == "CLIPTextModel":
                from transformers import CLIPTextModel

                return CLIPTextModel
            elif model_class == "RobertaSeriesModelWithTransformation":
                from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

                return RobertaSeriesModelWithTransformation
            else:
                raise ValueError(f"{model_class} is not supported.")

        self.text_encoder_cls = load_text_encoder(pretrained_model)
        self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
        self.text_encoder = self.text_encoder_cls.from_pretrained(pretrained_model, subfolder="text_encoder", revision=None)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=None)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet", revision=None)
        self.controlnet = ControlNetModel.from_pretrained(pretrained_control)

        self.modelzoos.append(self.vae.to(self.device, dtype=self.weight_dtype))
        self.modelzoos.append(self.unet.to(self.device, dtype=self.weight_dtype))
        self.modelzoos.append(self.text_encoder.to(self.device, dtype=self.weight_dtype))
        self.modelzoos.append(self.controlnet.to(self.device, dtype=self.weight_dtype))

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.requires_grad_(False)


    def load_global_inpaint_model(self):

        if self.current_model == 'global':
            return
        elif self.current_model == 'local':
            self.offline_current_model()

        self.load_diffusion_model(args.global_pretrained_path, args.global_controlnet_path)

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.global_pretrained_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            safety_checker=None,
            revision=None,
            torch_dtype=self.weight_dtype,
        )

        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.compel_proc = Compel(tokenizer=self.pipeline.tokenizer, text_encoder=self.pipeline.text_encoder)

        self.current_model = 'global'
        
        logging.info(f'Load StableDiffusion from {args.global_pretrained_path}')
        logging.info(f'Load GlobalInpaintControlNet ckpt from {args.global_controlnet_path}')



    def global_inpaint(self, validation_image, validation_prompt, num_validation_images, n_prompt, ddim_steps, scale, control_scale, validation_seed):

        generator = torch.Generator(device=self.device).manual_seed(validation_seed)
        self.set_seed(validation_seed)

        validation_image = Image.fromarray(validation_image)

        images = []

        for i in range(num_validation_images):
            with torch.autocast("cuda"):
                prompt_embeds = self.compel_proc(validation_prompt)
                image = self.pipeline(
                    image=validation_image, num_inference_steps=ddim_steps, generator=generator, prompt_embeds=prompt_embeds,
                    controlnet_conditioning_scale=control_scale, guidance_scale=scale, negative_prompt=n_prompt
                ).images[0]

            images.append(np.array(image))

        images = np.asarray(images)
        images = images.tobytes()

        torch.cuda.empty_cache()

        return images

    def load_local_inpaint_model(self):

        if self.current_model == 'local':
            return
        elif self.current_model == 'global':
            self.offline_current_model()

        self.load_diffusion_model(args.local_pretrained_path, args.local_controlnet_path)

        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            args.local_pretrained_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            safety_checker=None,
            revision=None,
            torch_dtype=self.weight_dtype,
        )
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

        self.current_model = 'local'
        
        logging.info(f'Load StableDiffusion from {args.local_pretrained_path}')
        logging.info(f'Load LocalInpaintControlNet ckpt from {args.local_controlnet_path}')

    def local_inpaint(self, validation_image, validation_mask, validation_prompt, num_validation_images, n_prompt, ddim_steps, scale, control_scale, validation_seed):

        generator = torch.Generator(device=self.device).manual_seed(validation_seed)
        self.set_seed(validation_seed)

        def make_inpaint_condition(image, image_mask):
            image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

            assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
            image[image_mask > 0.5] = -1.  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image


        def dilate_mask(mask):
            partial_tex_mask = np.asarray(mask).sum(axis=-1, keepdims=True)
            partial_tex_mask[partial_tex_mask != 0.] = 255.
            partial_tex_mask = np.repeat(partial_tex_mask, 3, axis=-1)
            partial_tex_mask = partial_tex_mask * self.uv_texture_mask
            # partial_tex_mask = 255. - partial_tex_mask

            partial_tex_mask = cv2.cvtColor(partial_tex_mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
            partial_tex_mask = cv2.dilate(partial_tex_mask.astype(np.uint8), kernel, iterations=1)
            partial_tex_mask = cv2.cvtColor(partial_tex_mask, cv2.COLOR_BGR2RGB)
            partial_tex_mask = partial_tex_mask * self.uv_texture_mask

            # partial_tex_mask = 255. - partial_tex_mask
            return Image.fromarray(partial_tex_mask.astype(np.uint8))

        validation_image = np.asarray(validation_image).astype(float)
        validation_mask = np.asarray(dilate_mask(validation_mask)).astype(float)

        validation_mask[validation_mask != 0.] = 255.

        validation_image[validation_mask == 255.] = 0.
        validation_image = validation_image * self.uv_texture_mask # only keep project texture
        validation_image = Image.fromarray(validation_image.astype(np.uint8))

        # validation_mask[uv_texture_mask == 0.] = 0. # keep outer parts of uv_texture_mask
        validation_mask = Image.fromarray(validation_mask.astype(np.uint8))
        control_image = make_inpaint_condition(validation_image, validation_mask)


        images = []

        for i in range(num_validation_images):
            with torch.autocast("cuda"):
                image = self.pipeline(
                    validation_prompt, image=validation_image, mask_image=validation_mask, control_image=control_image,
                    num_inference_steps=ddim_steps, generator=generator, strength=1, controlnet_conditioning_scale=control_scale,
                    guidance_scale=scale, negative_prompt=n_prompt
                ).images[0]

            images.append(np.array(image))

        images = np.asarray(images)
        images = images.tobytes()

        torch.cuda.empty_cache()

        return images


    def offline_current_model(self):

        for model in self.modelzoos:
            model.cpu()
            del model

        self.modelzoos = []

        torch.cuda.empty_cache()

        self.tokenizer = None
        self.text_encoder_cls = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.controlnet = None
        self.pipeline = None
        self.compel_proc = None

        self.current_model = 'empty'

    def set_seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class InpaintServer(object):

    def __init__(self, model_zoo):

        self.model_zoo = model_zoo
        

    def global_inpaint(self, image_bytes, num_samples, prompt, n_prompt, ddim_steps, scale, control_scale, seed, size):

        logging.info(f'New tasks for global inpainting: {num_samples}')

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_array = image_array.reshape((size, size, 3))
        
        self.model_zoo.load_global_inpaint_model()

        image_bytes = self.model_zoo.global_inpaint(image_array, prompt, num_samples, n_prompt, ddim_steps, scale, control_scale, seed)

        # image_array = np.fliplr(image_array)
        # image_bytes = image_array.tobytes()

        return image_bytes

    def local_inpaint(self, image_bytes, mask_bytes, num_samples, prompt, n_prompt, ddim_steps, scale, control_scale, seed, size):

        logging.info(f'New tasks for local inpainting: {num_samples}')

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_array = image_array.reshape((size, size, 3))

        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask_array = mask_array.reshape((size, size, 3))
        
        self.model_zoo.load_local_inpaint_model()

        image_bytes = self.model_zoo.local_inpaint(image_array, mask_array, prompt, num_samples, n_prompt, ddim_steps, scale, control_scale, seed)

        return image_bytes

    def inpaint(self, image_bytes, mask_bytes, num_samples, prompt, n_prompt, ddim_steps, scale, control_scale, seed, size):

        current_stat = self.model_zoo.target_model if self.model_zoo.target_model != 'empty' else self.model_zoo.current_model

        if current_stat == 'local':
            results = self.local_inpaint(image_bytes, mask_bytes, num_samples, prompt, n_prompt, ddim_steps, scale, control_scale, seed, size)
        else:
            results = self.global_inpaint(image_bytes, num_samples, prompt, n_prompt, ddim_steps, scale, control_scale, seed, size)

        logging.info('Done')

        return results


    def switch_model(self, model_name):

        if self.model_zoo.current_model == model_name:
            self.model_zoo.target_model = 'empty'
        else:
            self.model_zoo.target_model = model_name

        # model_zoo.offline_current_model()

        if model_name == 'global':
            logging.info('Switch to global mode')
            # model_zoo.load_global_inpaint_model()
        
        if model_name == 'local':
            logging.info('Switch to local mode')
            # model_zoo.load_local_inpaint_model()

        return
        
def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, default=0, help='The GPU device to be used.')
    parser.add_argument('-i', '--host', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=4242)
    parser.add_argument('--global_pretrained_path', type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument('--global_controlnet_path', type=str, default="./save/ckpt/texture_global")
    parser.add_argument('--local_pretrained_path', type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument('--local_controlnet_path', type=str, default="./save/ckpt/texture_local")

    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)

    args = parse()

    logging.info(f'Server start: {args.host}:{args.port}')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model_zoo = ModelZoo(args, device)
    
    s = zerorpc.Server(InpaintServer(model_zoo))

    s.bind(f"tcp://{args.host}:{args.port}")

    s.run()