import torch
import os
import io
import uuid
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, DDIMScheduler
from ip_adapter import IPAdapterXL
from utils import resize_image, empty_cache

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_PATH = "xinsir/controlnet-tile-sdxl-1.0"
IP_ADAPTER_EXTRACTOR = "IP-Adapter/sdxl_models/image_encoder"
IP_ADAPTER_MODULE = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# STYLE DROPDOWN (Swagger Enum)
# ------------------------------------------------------------

class StyleName(str, Enum):
    disney = "disney"
    anime = "anime"
    gauguin = "gauguin"
    vangogh = "vangogh"


STYLE_CONFIG = {

    "disney": {
        "style_id": 112,
        "prompt": "family-friendly animated film look, appealing character design, soft shading, bright colors, clean linework, warm lighting",
        "negative": "photorealistic, gritty, horror, harsh shadows"
    },

    "anime": {
        "style_id": 119,
        "prompt": "anime style illustration, clean lineart, cel shading, vibrant colors, detailed eyes, smooth gradients",
        "negative": "photorealistic, painterly impasto, heavy texture"
    },

    "gauguin": {
        "style_id": 120,
        "prompt": "post-impressionist painting, flat color areas, bold contours, warm saturated palette, textured brushwork",
        "negative": "photorealistic, hyperreal, glossy, plastic skin"
    },

    "vangogh": {
        "style_id": 103,
        "prompt": "oil painting, expressive swirling brushstrokes, thick impasto texture, vibrant color, painterly",
        "negative": "photorealistic, smooth shading, airbrush, plastic"
    }
}


BASE_PROMPT = "masterpiece, best quality, high quality"
BASE_NEGATIVE = "text, watermark, logo, lowres, blurry, worst quality, deformed, bad anatomy"

# ------------------------------------------------------------
# APP
# ------------------------------------------------------------

app = FastAPI(
    title="Multi-Style SDXL IP-Adapter API",
    description="SDXL + ControlNet + IP-Adapter (with style dropdown + image URLs)"
)

# Serve generated images
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

MODEL = None

# ------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------

@app.on_event("startup")
def load_models():

    global MODEL

    print(f"Loading models on {DEVICE.upper()}...")

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_PATH,
        torch_dtype=torch.float16
    ).to(DEVICE)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

    target_blocks = [
        "down_blocks.0.attentions.0",
        "down_blocks.1.attentions.0",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
        "up_blocks.1.attentions.0",
        "up_blocks.2.attentions.0",
    ]

    MODEL = IPAdapterXL(
        pipe,
        IP_ADAPTER_EXTRACTOR,
        IP_ADAPTER_MODULE,
        DEVICE,
        target_blocks=target_blocks
    )

    empty_cache()
    print("API ready.")


# ------------------------------------------------------------
# GENERATE ENDPOINT
# ------------------------------------------------------------

@app.post("/generate")
async def generate(
    request: Request,
    style: StyleName = Form(...),
    scale: float = Form(1.6),
    controlnet_scale: float = Form(0.35),
    steps: int = Form(35),
    seed: int = Form(42),
    content_file: UploadFile = File(...)
):

    try:

        style_key = style.value
        cfg = STYLE_CONFIG[style_key]

        style_id = cfg["style_id"]
        style_path = f"data/style/{style_id}.jpg"

        if not os.path.exists(style_path):
            raise HTTPException(
                status_code=404,
                detail=f"Style image not found: {style_path}"
            )

        # -------------------------
        # Load images
        # -------------------------

        style_image = Image.open(style_path).convert("RGB")

        content_bytes = await content_file.read()
        content_image = Image.open(io.BytesIO(content_bytes)).convert("RGB")

        W, H = content_image.size

        style_image = resize_image(style_image, short=768)
        controlnet_cond_image = resize_image(content_image, short=1024)

        prompt = f"{BASE_PROMPT}, {cfg['prompt']}"
        negative_prompt = f"{BASE_NEGATIVE}, {cfg['negative']}"

        # -------------------------
        # Generate
        # -------------------------

        with torch.no_grad():
            images = MODEL.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                scale=scale,
                guidance_scale=4.5,
                num_samples=1,
                num_inference_steps=steps,
                seed=seed,
                controlnet_conditioning_scale=controlnet_scale,
                pil_image=style_image,
                image=controlnet_cond_image
            )

        out = images[0].resize((W, H), Image.Resampling.LANCZOS)

        filename = f"api_{style_key}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        out.save(filepath)

        # ✅ build public URL dynamically (no hard-coded IP)
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/results/{filename}"

        return {
            "status": "success",
            "style": style_key,
            "image_url": image_url
        }

    except HTTPException:
        raise
    except Exception as e:
        print("API error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# LOCAL RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
