import torch
import os
import gc
import sys
from PIL import Image

instantid_parent = "/home/ubuntu/chiki-ai"
if instantid_parent not in sys.path:
    sys.path.insert(0, instantid_parent)

import InstantID
import InstantID.utils
import InstantID.ip_adapter
import InstantID.face_detailer

sys.modules["instantid"] = InstantID
sys.modules["instantid.utils"] = InstantID.utils
sys.modules["instantid.ip_adapter"] = InstantID.ip_adapter
sys.modules["instantid.face_detailer"] = InstantID.face_detailer

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

from InstantID.ip_adapter.attention_processor import IPAttnProcessor
from InstantID.face_detailer.detailer_pipeline import FaceDetailer

from InstantID.utils.load_components import (
    load_face_app,
    prepare_face_embeddings
)

from utils.instantid_utils import (
    set_controlnet,
    set_ip_adapter_processors,
    set_feature_proj_model
)


class PHOTOBOOTH(object):

    def __init__(
        self,
        device="cuda",
        seed=777,
        style="disney",
        model_dir="/home/ubuntu/chiki-ai/models/PHOTOBOOTH_MODEL",
        use_config=None,
    ):

        self.style = style
        self.device = device

        self.model_root = model_dir
        self.sd_root = os.path.join(model_dir, "sd_15")

        print(f"\n{'='*60}")
        print(f"[INIT] Initializing PHOTOBOOTH - {style.upper()} style")
        print(f"{'='*60}")

        # ---------------------------------------------------------
        # Real-ESRGAN
        # ---------------------------------------------------------
        self.image_enhancer = RealESRGAN(device, scale=4)
        esrgan_path = os.path.join(self.model_root, "real_esrgan", "RealESRGAN_x4.pth")
        self.image_enhancer.load_weights(esrgan_path, download=False)

        # ---------------------------------------------------------
        # Face app
        # ---------------------------------------------------------
        face_app_root = os.path.join(
            self.model_root,
            "models",
            "antelopev2",
            "antelopev2"
        )

        self.face_app = load_face_app(
            name="antelopev2",
            root=face_app_root
        )

        # ---------------------------------------------------------
        # Base model
        # ---------------------------------------------------------
        base_ckpt = os.path.join(
            self.sd_root,
            "anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"
        )

        pipe = StableDiffusionPipeline.from_single_file(
            base_ckpt,
            torch_dtype=torch.float16
        ).to(device)

        # ---------------------------------------------------------
        # Style LoRAs / checkpoints
        # ---------------------------------------------------------
        if style == "ghibli":
            lora_path = os.path.join(
                self.sd_root,
                "ghibli_style_offset.safetensors"
            )
            if os.path.exists(lora_path):
                try:
                    pipe.load_lora_weights(lora_path)
                except Exception:
                    pass

        elif style == "dreamshaper":
            dreamshaper_path = os.path.join(
                self.sd_root,
                "dreamshaper_v8.safetensors"
            )
            if os.path.exists(dreamshaper_path):
                try:
                    pipe.load_lora_weights(dreamshaper_path)
                except Exception:
                    del pipe
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        pipe = StableDiffusionPipeline.from_single_file(
                            dreamshaper_path,
                            torch_dtype=torch.float16
                        ).to(device)
                    except Exception:
                        pipe = StableDiffusionPipeline.from_single_file(
                            base_ckpt,
                            torch_dtype=torch.float16
                        ).to(device)

        # ---------------------------------------------------------
        # Textual inversions
        # ---------------------------------------------------------
        try:
            pipe.load_textual_inversion("Eugeoter/badhandv4")
        except Exception:
            pass

        easy_neg = os.path.join(self.sd_root, "EasyNegative.safetensors")
        if os.path.exists(easy_neg):
            pipe.load_textual_inversion(easy_neg)

        # ---------------------------------------------------------
        # Scheduler
        # ---------------------------------------------------------
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            steps_offset=1,
            clip_sample=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(scheduler.config)

        # ---------------------------------------------------------
        # ControlNets
        # ---------------------------------------------------------
        controlnet_face_path = os.path.join(self.sd_root, "controlnet.ckpt")
        controlnet_tile_path = os.path.join(self.sd_root, "controlnet_tile.pth")
        controlnet_depth_path = os.path.join(self.sd_root, "controlnet_depth.pth")

        controlnet_face, controlnet_tile, controlnet_depth = set_controlnet(
            unet=pipe.unet,
            face_ckpt_path=controlnet_face_path,
            tile_ckpt_path=controlnet_tile_path,
            depth_ckpt_path=controlnet_depth_path,
            device=device,
            dtype=pipe.unet.dtype
        )

        # ---------------------------------------------------------
        # IP-Adapter
        # ---------------------------------------------------------
        ip_adapter_path = os.path.join(self.sd_root, "ip-state.ckpt")

        pipe.unet = set_ip_adapter_processors(
            unet=pipe.unet,
            ckpt_path=ip_adapter_path,
            num_tokens=16,
            scale=0.5,
            ignore_motion=True,
            device=device,
            dtype=pipe.unet.dtype
        )

        # ---------------------------------------------------------
        # Image projection
        # ---------------------------------------------------------
        image_proj_path = os.path.join(self.sd_root, "image_proj.ckpt")

        face_embedding_proj = set_feature_proj_model(
            ckpt_path=image_proj_path,
            image_emb_dim=512,
            num_tokens=16,
            device=device,
            dtype=pipe.unet.dtype
        )

        self.pipe = pipe
        self.controlnet_face = controlnet_face
        self.controlnet_tile = controlnet_tile
        self.controlnet_depth = controlnet_depth
        self.face_embedding_proj = face_embedding_proj

        # ---------------------------------------------------------
        # VAE for face detailer
        # ---------------------------------------------------------
        vae_path = os.path.join(
            self.sd_root,
            "vaeFtMse840000EmaPruned_vaeFtMse840k.safetensors"
        )

        face_detailer_vae = AutoencoderKL.from_single_file(
            vae_path,
            torch_dtype=pipe.unet.dtype
        ).to(device)

        # ---------------------------------------------------------
        # Face detailer
        # ---------------------------------------------------------
        yolo_path = os.path.join(
            self.model_root,
            "yolo",
            "yolov8n-face.pt"
        )

        self.face_detailer = FaceDetailer(
            device=device,
            yolo_url=yolo_path,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            vae=face_detailer_vae,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=None
        )

        del pipe, controlnet_face, controlnet_tile, controlnet_depth
        del face_embedding_proj, face_detailer_vae

        gc.collect()
        torch.cuda.empty_cache()

        self.preprocess = transforms.ToTensor()
        self.generator = torch.Generator(device=device).manual_seed(seed)

        self.depth_estimator = pipeline("depth-estimation")

        self.dtype = self.pipe.unet.dtype

        self.positive_text_embedding, self.negative_text_embedding = \
            self.process_text(style=style)

    # ---------------------------------------------------------

    def process_image(self, image, shape=(544, 680)):

        if image.mode != "RGB":
            image = image.convert("RGB")

        src = image.resize(shape)

        # ---------------- FIXED DEPTH ----------------
        depth = self.depth_estimator(src)["depth"]
        depth = np.array(depth)

        depth = depth - depth.min()
        depth = depth / (depth.max() + 1e-8)
        depth = (depth * 255).astype(np.uint8)

        depth = np.stack([depth] * 3, axis=-1)
        depth = Image.fromarray(depth, mode="RGB")

        # ---------------------------------------------

        enhanced = self.image_enhancer.predict(src)

        if isinstance(enhanced, Image.Image):
            if enhanced.mode != "RGB":
                enhanced = enhanced.convert("RGB")
            src = enhanced.resize(src.size)
        else:
            if len(enhanced.shape) == 2:
                enhanced = np.stack([enhanced] * 3, axis=-1)
            src = Image.fromarray(enhanced.astype(np.uint8)).resize(src.size)
            if src.mode != "RGB":
                src = src.convert("RGB")

        face_embedding, _ = prepare_face_embeddings(
            source_image=src,
            face_app=self.face_app
        )

        face_embedding = torch.tensor(face_embedding).reshape(1, -1, 512)

        controlnet_image_face = self.preprocess(src).unsqueeze(0)
        controlnet_image_tile = self.preprocess(src).unsqueeze(0)
        controlnet_image_depth = self.preprocess(depth).unsqueeze(0)

        src = self.preprocess(src).unsqueeze(0) * 2.0 - 1.0

        return (
            face_embedding,
            controlnet_image_face,
            controlnet_image_tile,
            controlnet_image_depth,
            src
        )

    # ---------------------------------------------------------

    def process_text(
        self,
        positive_prompt="masterpiece, best quality, high resolution, cartoon, character, beautiful, adorable, cute, perfect face, soft smiling face, perfect eyes, natural eye proportions",
        negative_prompt="<badhandv4>, lowres, low quality, worst quality, deformed face, glitch, deformed, mutated, cross-eyed, misalinged eyes, wide-spaced pupils, ugly, disfigured, extra limb",
        style="disney"
    ):

        if style == "disney":
            positive_prompt += ", Disney, Disney style, Disney character, Disney animation, Disney background"
        elif style == "ghibli":
            positive_prompt += ", ghibli style, ghibli animation, ghibli background, Studio Ghibli"
        elif style == "dreamshaper":
            positive_prompt += ", dreamy, fantasy art, digital painting, highly detailed, cinematic lighting, vibrant colors"

        with torch.no_grad():
            pos, neg = self.pipe.encode_prompt(
                prompt=positive_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                clip_skip=2
            )

        return pos, neg

    # ---------------------------------------------------------

    def process_tensor(self, tensor, value_range=(-1, 1)):
        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        if value_range is not None:
            norm_ip(tensor, value_range[0], value_range[1])
        else:
            norm_ip(tensor, float(tensor.min()), float(tensor.max()))

        return tensor

    # ---------------------------------------------------------

    def set_ip_adapter_scale(self, unet, scale=1.0):
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    # ---------------------------------------------------------

    def make_batch(self, batch):

        face_embeddings = []
        controlnet_images_face = []
        controlnet_images_tile = []
        controlnet_images_depth = []
        srcs = []

        for img in batch:
            fe, cif, cit, cid, src = self.process_image(img)

            face_embeddings.append(fe.to(self.device, self.dtype, non_blocking=True))
            controlnet_images_face.append(cif.to(self.device, self.dtype, non_blocking=True))
            controlnet_images_tile.append(cit.to(self.device, self.dtype, non_blocking=True))
            controlnet_images_depth.append(cid.to(self.device, self.dtype, non_blocking=True))
            srcs.append(src.to(self.device, self.dtype, non_blocking=True))

        face_embeddings = torch.cat(face_embeddings, dim=0)

        with torch.no_grad():
            face_embeddings = self.face_embedding_proj(face_embeddings)

        controlnet_images_face = torch.cat(controlnet_images_face, dim=0)
        controlnet_images_tile = torch.cat(controlnet_images_tile, dim=0)
        controlnet_images_depth = torch.cat(controlnet_images_depth, dim=0)
        srcs = torch.cat(srcs, dim=0)

        with torch.no_grad():
            latents = self.pipe.vae.encode(srcs).latent_dist.sample(
                generator=self.generator
            ) * self.pipe.vae.config.scaling_factor

        pos = self.positive_text_embedding.repeat(len(batch), 1, 1)
        neg = self.negative_text_embedding.repeat(len(batch), 1, 1)

        return (
            face_embeddings,
            controlnet_images_face,
            controlnet_images_tile,
            controlnet_images_depth,
            latents,
            pos,
            neg
        )

    # ---------------------------------------------------------

    def generate(
        self,
        face_embeddings,
        controlnet_images_face,
        controlnet_images_tile,
        controlnet_images_depth,
        latents,
        positive_text_embeddings,
        negative_text_embeddings,
        num_inference_steps,
        strength=0.7,
        cfg_scale=7.5
    ):

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        start = num_inference_steps - int(num_inference_steps * strength)
        timesteps = self.pipe.scheduler.timesteps[start * self.pipe.scheduler.order:]
        init_timesteps = timesteps[0].repeat(len(latents))

        prompt_embeds = torch.cat(
            [negative_text_embeddings, positive_text_embeddings], dim=0
        )

        face_embeds = torch.cat(
            [torch.zeros_like(face_embeddings), face_embeddings], dim=0
        )

        controlnet_images_face = torch.cat([controlnet_images_face] * 2, dim=0)
        controlnet_images_tile = torch.cat([controlnet_images_tile] * 2, dim=0)
        controlnet_images_depth = torch.cat([controlnet_images_depth] * 2, dim=0)

        noise = torch.randn(
            latents.shape,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype
        )

        noisy_latents = self.pipe.scheduler.add_noise(
            latents, noise, init_timesteps
        )

        self.set_ip_adapter_scale(self.pipe.unet, scale=0.5)

        with torch.no_grad():

            for time in tqdm(timesteps, desc="Generating"):

                noisy_latents_input = torch.cat([noisy_latents] * 2, dim=0)
                noisy_latents_input = self.pipe.scheduler.scale_model_input(
                    noisy_latents_input, time
                )

                down_face, mid_face = self.controlnet_face(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=face_embeds,
                    controlnet_cond=controlnet_images_face,
                    conditioning_scale=0.5,
                    return_dict=False
                )

                down_tile, mid_tile = self.controlnet_tile(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_images_tile,
                    conditioning_scale=0.8,
                    return_dict=False
                )

                down_depth, mid_depth = self.controlnet_depth(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_images_depth,
                    conditioning_scale=0.3,
                    return_dict=False
                )

                down = [
                    a + b + c
                    for a, b, c in zip(down_face, down_tile, down_depth)
                ]

                mid = mid_face + mid_tile + mid_depth

                # -------- FIXED UNET CALL (NO CONCAT WITH FACE TOKENS) --------
                noise_pred = self.pipe.unet(
                    noisy_latents_input,
                    time,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                    return_dict=False
                )[0]
                # ----------------------------------------------------------------

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

                noisy_latents = self.pipe.scheduler.step(
                    noise_pred,
                    time,
                    noisy_latents,
                    return_dict=False
                )[0]

            decoded = self.pipe.vae.decode(
                noisy_latents / self.pipe.vae.config.scaling_factor,
                return_dict=False
            )[0]

        decoded = decoded.float().cpu()

        images = []

        for t in decoded:
            t = self.process_tensor(t, (-1, 1))
            arr = (
                t.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to(torch.uint8)
                .numpy()
            )
            images.append(Image.fromarray(arr))

        # ---------------------------------------------------------
        # Face detailer
        # ---------------------------------------------------------
        with torch.no_grad():

            detailed_images = self.face_detailer.generate_retouch(images)

            final_images = []

            if isinstance(detailed_images, list):
                for item in detailed_images:
                    if isinstance(item, list):
                        if len(item) > 0 and isinstance(item[0], Image.Image):
                            final_images.append(item[0])
                    elif isinstance(item, Image.Image):
                        final_images.append(item)

                if not final_images:
                    final_images = images

                # -------- FIX: ALWAYS RETURN ONE IMAGE --------
                images = final_images[:1]
            else:
                images = images[:1]

        return images

    # ---------------------------------------------------------

    def inference(self, images):

        batch = self.make_batch(images)

        generated = self.generate(
            *batch,
            num_inference_steps=30,
            strength=0.7,
            cfg_scale=7.5
        )

        return generated
