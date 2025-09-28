import torch
import os
import gc
from PIL import Image
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
    AutoencoderKL
)
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from RealESRGAN import RealESRGAN
from InstantID import IPAttnProcessor, FaceDetailer
from PIL import Image
from utils import (
    load_face_app, 
    prepare_face_embeddings, 
    set_controlnet,
    set_ip_adapter_processors,
    set_feature_proj_model 
)

class PHOTOBOOTH(object):
    def __init__(self,
                 device="cuda",
                 seed=777,
                 style="disney",
                 model_dir="root/photobooth/ckpt"):
        self.image_enhancer = RealESRGAN(device, scale=4)
        self.image_enhancer.load_weights(os.path.join(model_dir, "RealESRGAN_x4.pth"), download=False)

        self.face_app = load_face_app(
            name='antelopev2',
            root=os.path.join(model_dir, 'models/antelopev2/antelopev2')
        )

        pipe = StableDiffusionPipeline.from_single_file(
            os.path.join(model_dir, 'disneyPixarCartoon_v10.safetensors'),
            torch_dtype=torch.float16
        ).to(device)
        pipe.load_textual_inversion("Eugeoter/badhandv4")
        '''
        pipe.load_textual_inversion("EvilEngine/easynegative",
                                    weight_name="easynegative.safetensors",
                                    token="easynegative")
        pipe.load_textual_inversion("ckpt/verybadimagenegative_v1.3.pt", token="<verybad>")
        '''

        scheduler = DDIMScheduler(**{
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'linear',
            'steps_offset': 1,
            'clip_sample': False,
            })
        pipe.scheduler = UniPCMultistepScheduler.from_config(scheduler.config)

        controlnet_face, controlnet_tile, controlnet_depth = set_controlnet(
            unet=pipe.unet, 
            face_ckpt_path=os.path.join(model_dir, 'controlnet_face.ckpt'),
            tile_ckpt_path=os.path.join(model_dir, 'controlnet_tile.pth'),
            depth_ckpt_path=os.path.join(model_dir, 'controlnet_depth.pth'),
            device=device, dtype=pipe.unet.dtype
        )

        pipe.unet = set_ip_adapter_processors(
            unet=pipe.unet,
            ckpt_path=os.path.join(model_dir, 'ip-state.ckpt'),
            num_tokens=16,
            scale=0.5,
            ignore_motion=True,
            device=device, dtype=pipe.unet.dtype
        )

        face_embedding_proj = set_feature_proj_model(
            ckpt_path=os.path.join(model_dir, "image_proj.ckpt"),
            image_emb_dim=512,
            num_tokens=16,
            device=device, dtype=pipe.unet.dtype
        )

        self.pipe, self.controlnet_face, self.controlnet_tile, self.controlnet_depth, self.face_embedding_proj = (
            pipe, controlnet_face, controlnet_tile, controlnet_depth, face_embedding_proj
        )

        face_detailer_vae = AutoencoderKL.from_single_file("ckpt/vae.safetensors", torch_dtype=pipe.unet.dtype).to(device)
        self.face_detailer = FaceDetailer(
            device=device,
            yolo_url="ckpt/yolov8n-face.pt",
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            vae=face_detailer_vae,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=None
        )

        del pipe, controlnet_face, controlnet_tile, controlnet_depth, face_embedding_proj, face_detailer_vae
        gc.collect()
        torch.cuda.empty_cache()

        self.preprocess = transforms.ToTensor()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.depth_estimator = pipeline('depth-estimation')
        self.device = device
        self.dtype = self.pipe.unet.dtype

        self.positive_text_embedding, self.negative_text_embedding = self.process_text(style=style)

    def process_image(self, 
                      image,
                      shape=(544, 680)):
        src = image.resize(shape)

        depth = self.depth_estimator(src)['depth']
        depth = np.array(depth)[:, :, None]
        depth = np.concatenate([depth] * 3, axis=2)
        depth = Image.fromarray(depth)

        src = self.image_enhancer.predict(src).resize((src.size))

        face_embedding, face_keypoints_image = prepare_face_embeddings(
            source_image=src, 
            face_app=self.face_app,
            force_zero=False
        )

        face_embedding = torch.tensor(face_embedding).reshape(1, -1, 512)
        controlnet_image_face = self.preprocess(face_keypoints_image).unsqueeze(0)
        controlnet_image_tile = self.preprocess(src).unsqueeze(0)
        controlnet_image_depth = self.preprocess(depth).unsqueeze(0)
        src = self.preprocess(src).unsqueeze(0) * 2.0 - 1.0

        return face_embedding, controlnet_image_face, controlnet_image_tile, controlnet_image_depth, src
    
    def process_text(self,
                     positive_prompt = "masterpiece, best quality, high resolution, cartoon, character, beautiful, adorable, cute, perfect face, soft smiling face, perfect eyes, natural eye proportions",
                     negative_prompt = "<badhandv4>, lowres, low quality, worst quality, deformed face, glitch, deformed, mutated, cross-eyed, misalinged eyes, wide-spaced pupils, ugly, disfigured, extra limb",
                     style = "disney"):
        if style == "disney":
            positive_prompt += "Disney, Disney style, Disney character, Disney animation, Disney background"
        if style == "ghibli":
            positive_prompt += "ghibli style, ghibli animation, ghibli background, san \(mononoke hime\), howl \(howl no ugoku shiro\)"

        with torch.no_grad():
            positive_text_embedding, negative_text_embedding = self.pipe.encode_prompt(
                prompt=positive_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                clip_skip=2
            )
            gc.collect()
            torch.cuda.empty_cache()
        return positive_text_embedding, negative_text_embedding
    
    def process_tensor(self,
                       tensor,
                       value_range=(-1, 1)):
        
        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        norm_range(tensor, value_range)
        return tensor
    
    def set_ip_adapter_scale(self, 
                             unet, 
                             scale=1.0):
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
        
    def make_batch(self, 
                   batch):
        face_embeddings = []
        controlnet_images_face = []
        controlnet_images_tile = []
        controlnet_images_depth = []
        srcs = []

        for img in batch:
            face_embedding, controlnet_image_face, controlnet_image_tile, controlnet_image_depth, src = self.process_image(img)
            face_embedding = face_embedding.to(device=self.device, dtype=self.dtype, non_blocking=True)
            controlnet_image_face = controlnet_image_face.to(device=self.device, dtype=self.dtype, non_blocking=True)
            controlnet_image_tile = controlnet_image_tile.to(device=self.device, dtype=self.dtype, non_blocking=True)
            controlnet_image_depth = controlnet_image_depth.to(device=self.device, dtype=self.dtype, non_blocking=True)
            src = src.to(device=self.device, dtype=self.dtype, non_blocking=True)
        
            face_embeddings.append(face_embedding)
            controlnet_images_face.append(controlnet_image_face)
            controlnet_images_tile.append(controlnet_image_tile)
            controlnet_images_depth.append(controlnet_image_depth)
            srcs.append(src)

        face_embeddings = torch.cat(face_embeddings, dim=0)
        with torch.no_grad():
            face_embeddings = self.face_embedding_proj(face_embeddings)
            gc.collect()
            torch.cuda.empty_cache()
        controlnet_images_face = torch.cat(controlnet_images_face, dim=0)
        controlnet_images_tile = torch.cat(controlnet_images_tile, dim=0)
        controlnet_images_depth = torch.cat(controlnet_images_depth, dim=0)
        srcs = torch.cat(srcs, dim=0)
        with torch.no_grad():
            latents = self.pipe.vae.encode(srcs).latent_dist.sample(generator=self.generator) * self.pipe.vae.config.scaling_factor
            gc.collect()
            torch.cuda.empty_cache()
        positive_text_embeddings = self.positive_text_embedding.repeat(len(batch), 1, 1)
        negative_text_embeddings = self.negative_text_embedding.repeat(len(batch), 1, 1)
        return (face_embeddings, controlnet_images_face, controlnet_images_tile, 
                controlnet_images_depth, latents, positive_text_embeddings, negative_text_embeddings)

    def generate(self, 
                 face_embeddings,
                 controlnet_images_face,
                 controlnet_images_tile,
                 controlnet_images_depth,
                 latents,
                 positive_text_embeddings,
                 negative_text_embeddings,
                 num_inference_steps,
                 strength=0.7,
                 cfg_scale=7.5):
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        start = num_inference_steps - int(num_inference_steps * strength)
        timesteps = self.pipe.scheduler.timesteps[start * self.pipe.scheduler.order:]
        init_timesteps = timesteps[0].repeat(len(latents))

        prompt_embeds = torch.cat([negative_text_embeddings, positive_text_embeddings], dim=0)
        face_embeds = torch.cat([torch.zeros_like(face_embeddings), face_embeddings], dim=0)
        controlnet_images_face = torch.cat([controlnet_images_face] * 2, dim=0)
        controlnet_images_tile = torch.cat([controlnet_images_tile] * 2, dim=0)
        controlnet_images_depth = torch.cat([controlnet_images_depth] * 2, dim=0)

        noise = torch.randn(latents.shape, generator=self.generator, device=self.device, dtype=self.dtype)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, init_timesteps)

        self.set_ip_adapter_scale(self.pipe.unet, scale=0.5)
        pbar = tqdm(total=len(timesteps), desc="Generating Images")
        with torch.no_grad():
            for idx, time in enumerate(timesteps):
                time = time.to(self.device)
                noisy_latents_input = torch.cat([noisy_latents] * 2, dim=0)
                noisy_latents_input = self.pipe.scheduler.scale_model_input(noisy_latents_input, time)

                down_block_res_samples_face, mid_block_res_samples_face = self.controlnet_face(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=face_embeds,
                    controlnet_cond=controlnet_images_face,
                    conditioning_scale=0.5,
                    return_dict=False
                )
                down_block_res_samples_tile, mid_block_res_samples_tile = self.controlnet_tile(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_images_tile,
                    conditioning_scale=0.8,
                    return_dict=False
                )
                down_block_res_samples_depth, mid_block_res_samples_depth = self.controlnet_depth(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_images_depth,
                    conditioning_scale=0.3,
                    return_dict=False
                )

                down_block_res_samples = [
                    face_res + tile_res + depth_res
                    for face_res, tile_res, depth_res in zip(down_block_res_samples_face, down_block_res_samples_tile, down_block_res_samples_depth)
                ]
                mid_block_res_samples = (mid_block_res_samples_face + mid_block_res_samples_tile + mid_block_res_samples_depth)

                noise_pred = self.pipe.unet(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=torch.cat([prompt_embeds, face_embeds], dim=1),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_samples,
                    return_dict=False
                )[0]

                noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)
                noisy_latents = self.pipe.scheduler.step(noise_pred, time.to(self.device), noisy_latents, return_dict=False)[0]
                pbar.update(1)
            pbar.close()
            generated_tensors = self.pipe.vae.decode(noisy_latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

        generated_tensors = generated_tensors.float().cpu()
        generated_images = []
        for tensor in generated_tensors:
            tensor = self.process_tensor(tensor, value_range=(-1, 1))
            ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            img = Image.fromarray(ndarr)
            generated_images.append(img)
        with torch.no_grad():
            generated_images = self.face_detailer.generate_retouch(generated_images)
        gc.collect()
        torch.cuda.empty_cache()
        return generated_images

    def inference(self, images):
        (face_embeddings, controlnet_images_face, controlnet_images_tile, 
         controlnet_images_depth, latents, positive_text_embeddings, negative_text_embeddings) = self.make_batch(images)

        generated_images = self.generate(
            face_embeddings=face_embeddings,
            controlnet_images_face=controlnet_images_face,
            controlnet_images_tile=controlnet_images_tile,
            controlnet_images_depth=controlnet_images_depth,
            latents=latents,
            positive_text_embeddings=positive_text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            num_inference_steps=30,
            strength=0.7,
            cfg_scale=7.5
        )
        return generated_images
        
if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    photobooth = PHOTOBOOTH(device=device, seed=111, style="disney", model_dir="/root/photobooth/ckpt")
    os.makedirs("generated", exist_ok=True)
    image_paths = os.listdir("samples")
    images = [Image.open(os.path.join("samples", path)).convert("RGB") for path in image_paths]
    generated_images = photobooth.inference(images)
    for idx, gen_img in enumerate(generated_images):
        gen_img.save(f'generated/generated{idx}.png')
    print(1)    